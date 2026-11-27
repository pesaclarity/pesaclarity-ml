# ruff: noqa: E501
"""Scores a model on the held-out people: greedy replies, read as the app reads them.

Metrics (docs/SLM.md): JSON valid (the first line is exactly the schema), chart and data choice
accuracy (both right), numbers grounded (every number in the answer is in the fact sheet),
answer length, and filler (answers with a sentence that quotes an amount of zero or says
nothing, such as "You also paid KES 0 in fees"). Writes every reply, for the rayo check and the report's samples.

    python -m akili.slm.evaluate MODEL_DIR TEST_JSONL OUT_JSON [--n 200] [--name label]
    python -m akili.slm.evaluate --score REPLIES_JSON...   (scores replies made elsewhere, e.g. on rayo)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from akili.slm.answer import filler_sentences, parse, ungrounded_numbers

MAX_NEW = 160


def score(rows: list[dict], replies: list[str]) -> dict:  # type: ignore[type-arg]
    valid = chart = choice = grounded = filler = 0
    words: list[int] = []
    sentences: list[int] = []
    for r, reply in zip(rows, replies, strict=True):
        p = parse(reply)
        gold = r["choice"]
        if p.choice is not None:
            valid += 1
            chart += p.choice.chart == gold["chart"]
            choice += p.choice.chart == gold["chart"] and p.choice.data == gold["data"]
        grounded += not ungrounded_numbers(p.text, r["facts"])
        filler += bool(filler_sentences(p.text))
        words.append(len(p.text.split()))
        sentences.append(len([s for s in re.split(r"(?<=[.!?])\s+", p.text) if s.strip()]))
    n = len(rows)
    return {
        "n": n,
        "json_valid": round(100 * valid / n, 1),
        "chart_choice": round(100 * chart / n, 1),
        "chart_and_data": round(100 * choice / n, 1),
        "numbers_grounded": round(100 * grounded / n, 1),
        "filler": round(100 * filler / n, 1),
        "words_mean": round(statistics.mean(words), 1),
        "sentences_mean": round(statistics.mean(sentences), 1),
    }


def generate(model_dir: str, rows: list[dict]) -> tuple[list[str], float]:  # type: ignore[type-arg]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from akili.slm.train import prompt_ids

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    end = tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    start = time.time()
    with torch.inference_mode():
        for i, r in enumerate(rows):
            ids = torch.tensor([prompt_ids(tok, r["system"], r["question"])])
            g = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=MAX_NEW,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=end,
                pad_token_id=end,
            )
            out.append(tok.decode(g[0, ids.shape[1] :], skip_special_tokens=True).strip())
            if i % 25 == 0:
                print(f"{i}/{len(rows)} {time.time() - start:.0f} s", flush=True)
    return out, time.time() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?")
    ap.add_argument("test", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.test).read_text().splitlines()][: a.n]
    replies, seconds = generate(a.model, rows)
    result = {
        "name": a.name or a.model,
        "metrics": score(rows, replies),
        "seconds": round(seconds),
        "replies": [
            {"question": r["question"], "kind": r["kind"], "gold": r["choice"], "reply": x}
            for r, x in zip(rows, replies, strict=True)
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"name": result["name"], **result["metrics"], "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
