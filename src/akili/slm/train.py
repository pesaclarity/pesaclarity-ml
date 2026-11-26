# ruff: noqa: PLR0915
"""Fine-tunes a SmolLM2 instruct model on the phone model's synthetic data with LoRA, then merges.

CPU only (GitHub's runners). The loss is on the reply alone (the JSON line, the answer and the
end-of-turn token); the system text and question are read, not learnt. The prompt is the model's
own chat template, which is what rayo renders on the phone (rayo-convert stores it in the file).
No DP budget: the data is synthetic, made-up people only (docs/SLM.md).

    python -m akili.slm.train MODEL_DIR DATA_DIR OUT_DIR [--epochs 1] [--rank 16]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def prompt_ids(tok, system: str, question: str) -> list[int]:  # type: ignore[no-untyped-def]
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=True,
    )


def encode(tok, row: dict) -> tuple[list[int], list[int]]:  # type: ignore[no-untyped-def,type-arg]
    p = prompt_ids(tok, row["system"], row["question"])
    r = tok(row["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + r, [-100] * len(p) + r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--json-weight", type=float, default=1.0, help="loss weight of the chart line's tokens"
    )
    ap.add_argument("--micro", type=int, default=2, help="rows per forward pass")
    ap.add_argument("--limit", type=int, default=0, help="train on this many rows only")
    ap.add_argument("--hours", type=float, default=4.5, help="stop after this long")
    a = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    # Leave a core for the runner itself: starved of CPU, it loses touch with GitHub.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    model = get_peft_model(
        model,
        LoraConfig(
            r=a.rank,
            lora_alpha=2 * a.rank,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    # Memory: recompute activations in the backward pass, and score only the reply's tokens
    # (the full vocabulary's logits for every prompt token would not fit a runner).
    inner = model.base_model.model
    inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inner.enable_input_require_grads()
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    def reply_loss(ids, mask, labels, weights):  # type: ignore[no-untyped-def]
        hidden = inner.model(input_ids=ids, attention_mask=mask).last_hidden_state
        target = labels[:, 1:]
        keep = target != -100
        logits = inner.lm_head(hidden[:, :-1][keep])
        w = weights[:, 1:][keep]
        return (ce(logits.float(), target[keep]) * w).sum(), int(keep.sum())

    rows = [json.loads(x) for x in Path(a.data, "train.jsonl").read_text().splitlines()]
    if a.limit:
        rows = rows[: a.limit]
    data = [encode(tok, r) for r in rows]
    # The chart line's tokens (the first line of the reply), weighted by --json-weight.
    heads = [
        len(tok(r["target"].split("\n")[0], add_special_tokens=False)["input_ids"]) for r in rows
    ]
    print(f"{len(data)} rows, mean {sum(len(d[0]) for d in data) / len(data):.0f} tokens")

    # Batches of similar length, in random order.
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    batches = [order[i : i + a.batch] for i in range(0, len(order), a.batch)]
    steps = math.ceil(len(batches) * a.epochs)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / steps)
    )
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.train()
    start = time.time()
    step = 0
    while step < steps:
        random.shuffle(batches)
        for b in batches:
            if step >= steps:
                break
            tokens = sum(sum(1 for t in data[i][1][1:] if t != -100) for i in b)
            total = 0.0
            for k in range(0, len(b), a.micro):
                part = b[k : k + a.micro]
                n = max(len(data[i][0]) for i in part)
                ids = torch.full((len(part), n), pad)
                labels = torch.full((len(part), n), -100)
                mask = torch.zeros((len(part), n), dtype=torch.long)
                weights = torch.ones((len(part), n))
                for j, i in enumerate(part):
                    x, y = data[i]
                    first = sum(1 for t in y if t == -100)
                    weights[j, first : first + heads[i]] = a.json_weight
                    ids[j, : len(x)] = torch.tensor(x)
                    labels[j, : len(y)] = torch.tensor(y)
                    mask[j, : len(x)] = 1
                summed, _ = reply_loss(ids, mask, labels, weights)
                (summed / tokens).backward()
                total += summed.item()
            loss = torch.tensor(total / tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == 1:
                el = time.time() - start
                print(f"step {step}/{steps} loss {loss.item():.4f} {el / 60:.1f} min", flush=True)
            if time.time() - start > a.hours * 3600:
                print("time limit: stopping early", flush=True)
                steps = step
                break
    minutes = (time.time() - start) / 60
    merged = model.merge_and_unload()
    out = Path(a.out)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    # The base model's generation config, so greedy decoding and stop tokens stay the same.
    gen = Path(a.model, "generation_config.json")
    if gen.exists():
        (out / "generation_config.json").write_bytes(gen.read_bytes())
    (out / "TRAINING.json").write_text(
        json.dumps(
            {
                "rows": len(data),
                "steps": steps,
                "epochs": a.epochs,
                "rank": a.rank,
                "lr": a.lr,
                "batch": a.batch,
                "minutes": round(minutes, 1),
                "last_loss": round(loss.item(), 4),
            },
            indent=1,
        )
    )
    print(f"trained in {minutes:.1f} min")


if __name__ == "__main__":
    main()
