# ruff: noqa: E501
"""Two evaluations of the same held-out rows side by side: the scores, then sample answers.

python tool/slm/compare.py BEFORE_JSON AFTER_JSON [--samples 20]
"""

import argparse
import json
from pathlib import Path


def text(reply: str) -> str:
    lines = reply.strip().split("\n", 1)
    rest = lines[1] if len(lines) > 1 and lines[0].lstrip().startswith("{") else reply
    return " ".join(rest.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--samples", type=int, default=20)
    a = ap.parse_args()
    runs = [json.loads(Path(p).read_text()) for p in (a.before, a.after)]
    keys = (
        "json_valid",
        "chart_choice",
        "chart_and_data",
        "numbers_grounded",
        "filler",
        "words_mean",
        "sentences_mean",
    )
    print("## Before and after, same held-out people\n")
    print(
        "| model | n | JSON valid % | chart % | chart+data % | grounded % | filler % | words | sentences |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        m = r["metrics"]
        print(f"| {r['name']} | {m['n']} | " + " | ".join(str(m.get(k, "")) for k in keys) + " |")
    print(f"\n## {a.samples} sample answers, before and after\n")
    b, n = runs[0]["replies"], runs[1]["replies"]
    step = max(1, len(n) // a.samples)
    for i in range(0, min(len(n), step * a.samples), step):
        print(f"{i // step + 1}. **{n[i]['question']}** ({n[i]['kind']})  ")
        print(f"   before: {text(b[i]['reply'])}  ")
        print(f"   after: {text(n[i]['reply'])}")


if __name__ == "__main__":
    main()
