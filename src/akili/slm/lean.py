# ruff: noqa: PLR0915
"""Makes the fine-tuned phone model leaner: vocabulary trimming and layer pruning (docs/SLM.md).

``trim``: keeps only the tokens the task's text uses (every fact sheet, question and reply of the
synthetic data through the chat template, plus extra Swahili and Sheng words), every single-byte
token (so any other text still tokenises, byte by byte, and nothing breaks), the special tokens,
the most common tokens as a safety margin, and every token a kept token is merged from (so BPE
reaches the same tokens). Merges whose parts or result are dropped are removed. The embedding and
the (tied) output rows are gathered to the kept ids, which keep their old order.

``score``: each layer's importance, as 1 - the mean cosine similarity of the hidden state before
and after it (a layer that changes little matters little), on training rows.

``prune``: drops the ``k`` least important layers (never the first or the last), renumbering the
rest; a short healing fine-tune follows (``akili.slm.train``).

``parity``: the trimmed tokenizer gives the same tokens as the full one on the test rows.

    python -m akili.slm.lean trim MODEL DATA OUT [--margin 4000]
    python -m akili.slm.lean score MODEL DATA OUT_JSON [--rows 32]
    python -m akili.slm.lean prune MODEL SCORES_JSON K OUT
    python -m akili.slm.lean parity FULL TRIMMED DATA OUT_JSON
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: Words people may type that the synthetic questions do not all have.
EXTRA_TEXT = (
    "Pesa yangu imeenda wapi? Nimetumia pesa ngapi kwa chakula, fare, kodi, nyumba, stima, maji, "
    "shule, karo, hospitali, mkopo, Fuliza, M-Shwari, chama, sadaka, kanisa, mshahara, bajeti, "
    "akiba, matumizi, wiki hii, mwezi huu, mwezi uliopita, leo, jana, kesho. Doh, ganji, mulla, "
    "munde, nimechomeka, niko broke, nisaidie, punguza, ongeza, ngapi, gani, wapi, lini, kwa nini. "
    "How much, how many, why, when, which, what, where, who, spend, spent, save, saved, saving, "
    "budget, rent, food, fare, transport, airtime, data, bundles, fees, charges, loan, salary, "
    "income, payday, bills, balance, account, savings, chart, week, month, year, today. "
    "0123456789 KES Ksh , . ? ! : ; - ( ) / % ' \" { } [ ]"
)


def _corpus(data: Path, tok) -> list[str]:  # type: ignore[no-untyped-def]
    texts = []
    for name in ("train.jsonl", "test.jsonl"):
        for line in (data / name).read_text().splitlines():
            r = json.loads(line)
            texts.append(
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": r["system"]},
                        {"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": r["target"]},
                    ],
                    tokenize=False,
                )
            )
    texts.append(EXTRA_TEXT)
    texts.append(EXTRA_TEXT.lower())
    texts.append(EXTRA_TEXT.upper())
    return texts


def trim(model_dir: str, data: str, out: str, margin: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src, dst = Path(model_dir), Path(out)
    tok = AutoTokenizer.from_pretrained(src)
    tj = json.loads((src / "tokenizer.json").read_text())
    vocab: dict[str, int] = tj["model"]["vocab"]
    merges_raw = tj["model"]["merges"]
    merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in merges_raw]
    specials = {t["id"] for t in tj.get("added_tokens", [])}

    used: set[int] = set(specials)
    for text in _corpus(Path(data), tok):
        used.update(tok(text, add_special_tokens=False)["input_ids"])
    single = {i for t, i in vocab.items() if len(t) == 1}  # every byte
    common = set(sorted(vocab.values())[: margin + len(single)])
    keep_ids = used | single | common
    # Every token a kept token is made from, so BPE still reaches it.
    made_from: dict[str, list[str]] = {}
    for a, b in merges:
        made_from.setdefault(a + b, []).extend((a, b))
    inv = {i: t for t, i in vocab.items()}
    stack = [inv[i] for i in keep_ids if i in inv]
    keep_tokens = set(stack)
    while stack:
        t = stack.pop()
        if t in made_from:
            for part in made_from[t]:
                if part not in keep_tokens:
                    keep_tokens.add(part)
                    stack.append(part)
    keep_ids |= {vocab[t] for t in keep_tokens}
    order = sorted(keep_ids)
    new_id = {old: new for new, old in enumerate(order)}

    tj["model"]["vocab"] = {t: new_id[i] for t, i in vocab.items() if i in new_id}
    kept_merges = [
        m
        for m, (a, b) in zip(merges_raw, merges, strict=True)
        if a in keep_tokens and b in keep_tokens and (a + b) in keep_tokens
    ]
    tj["model"]["merges"] = kept_merges
    for t in tj.get("added_tokens", []):
        t["id"] = new_id[t["id"]]
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    tc = json.loads((src / "tokenizer_config.json").read_text())
    if "added_tokens_decoder" in tc:
        tc["added_tokens_decoder"] = {
            str(new_id[int(k)]): v for k, v in tc["added_tokens_decoder"].items()
        }
    (dst / "tokenizer_config.json").write_text(json.dumps(tc, ensure_ascii=False, indent=1))
    for f in ("special_tokens_map.json", "SOURCES", "REVISION"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)

    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
    idx = torch.tensor(order)
    emb = model.get_input_embeddings().weight.data[idx].clone()
    tied = bool(model.config.tie_word_embeddings)
    head = None if tied else model.get_output_embeddings().weight.data[idx].clone()
    model.model.embed_tokens = torch.nn.Embedding.from_pretrained(emb, freeze=False)
    model.lm_head = torch.nn.Linear(emb.shape[1], len(order), bias=False)
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        model.lm_head.weight = torch.nn.Parameter(head)
    model.config.vocab_size = len(order)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = getattr(model.config, key, None)
        if isinstance(v, int):
            setattr(model.config, key, new_id[v])
    model.save_pretrained(dst, safe_serialization=True)
    gen = json.loads((src / "generation_config.json").read_text())
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        v = gen.get(key)
        if isinstance(v, int):
            gen[key] = new_id[v]
        elif isinstance(v, list):
            gen[key] = [new_id[x] for x in v]
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=1))
    info = {
        "vocab_before": len(vocab),
        "vocab_after": len(order),
        "merges_before": len(merges),
        "merges_after": len(kept_merges),
        "used_by_corpus": len(used),
        "margin": margin,
    }
    (dst / "TRIM.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


def parity(full: str, trimmed: str, data: str, out: str) -> None:
    from transformers import AutoTokenizer

    a = AutoTokenizer.from_pretrained(full)
    b = AutoTokenizer.from_pretrained(trimmed)
    rows = [json.loads(x) for x in Path(data, "test.jsonl").read_text().splitlines()]
    texts = [r["system"] + "\n" + r["question"] + "\n" + r["target"] for r in rows]
    # Text the corpus never had: still tokenises (by bytes), decodes back unchanged.
    odd = ["Zawadi ya ndugu 🎉 €12.30 ½ ñandú Ωμέγα 漢字", "xqzvjk wpfrtl 0x7f3a"]
    same = sum(
        a.convert_ids_to_tokens(a(t, add_special_tokens=False)["input_ids"])
        == b.convert_ids_to_tokens(b(t, add_special_tokens=False)["input_ids"])
        for t in texts
    )
    roundtrip = all(b.decode(b(t, add_special_tokens=False)["input_ids"]) == t for t in odd)
    res = {
        "rows": len(texts),
        "same_tokens_pct": round(100 * same / len(texts), 2),
        "unseen_text_roundtrips": roundtrip,
        "tokens_full_unseen": [len(a(t)["input_ids"]) for t in odd],
        "tokens_trimmed_unseen": [len(b(t)["input_ids"]) for t in odd],
    }
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    if same != len(texts) or not roundtrip:
        raise SystemExit("trimmed tokenizer differs")


def score(model_dir: str, data: str, out: str, rows: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from akili.slm.train import encode

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    lines = Path(data, "train.jsonl").read_text().splitlines()[:rows]
    n = model.config.num_hidden_layers
    sims = torch.zeros(n)
    with torch.inference_mode():
        for line in lines:
            ids, _ = encode(tok, json.loads(line))
            hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
            for i in range(n):
                sims[i] += torch.nn.functional.cosine_similarity(
                    hs[i][0], hs[i + 1][0], dim=-1
                ).mean()
    importance = (1 - sims / len(lines)).tolist()
    Path(out).write_text(json.dumps({"importance": importance}, indent=1))
    print(json.dumps([round(x, 4) for x in importance]))


def prune(model_dir: str, scores: str, k: int, out: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    imp = json.loads(Path(scores).read_text())["importance"]
    n = len(imp)
    candidates = sorted(range(1, n - 1), key=lambda i: imp[i])[:k]
    keep = [i for i in range(n) if i not in candidates]
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    for j, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = j
    model.config.num_hidden_layers = len(keep)
    dst = Path(out)
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(model_dir).save_pretrained(dst)
    for f in ("generation_config.json", "SOURCES", "REVISION"):
        if Path(model_dir, f).exists():
            shutil.copy(Path(model_dir, f), dst / f)
    (dst / "PRUNE.json").write_text(
        json.dumps({"dropped": sorted(candidates), "layers": len(keep)})
    )
    print(f"dropped layers {sorted(candidates)}; {len(keep)} left")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trim")
    t.add_argument("model")
    t.add_argument("data")
    t.add_argument("out")
    t.add_argument("--margin", type=int, default=4000)
    s = sub.add_parser("score")
    s.add_argument("model")
    s.add_argument("data")
    s.add_argument("out")
    s.add_argument("--rows", type=int, default=32)
    p = sub.add_parser("prune")
    p.add_argument("model")
    p.add_argument("scores")
    p.add_argument("k", type=int)
    p.add_argument("out")
    q = sub.add_parser("parity")
    q.add_argument("full")
    q.add_argument("trimmed")
    q.add_argument("data")
    q.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "trim":
        trim(a.model, a.data, a.out, a.margin)
    elif a.cmd == "score":
        score(a.model, a.data, a.out, a.rows)
    elif a.cmd == "prune":
        prune(a.model, a.scores, a.k, a.out)
    else:
        parity(a.full, a.trimmed, a.data, a.out)


if __name__ == "__main__":
    main()
