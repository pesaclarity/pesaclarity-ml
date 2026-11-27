# ruff: noqa: E501
"""The evaluation report: a table of every model's scores and 20 sample answers each, as
Markdown (for the job summary and docs/SLM-EVAL.md).

    python tool/slm/report.py RESULTS_DIR > report.md
"""

import json
import sys
from pathlib import Path


def main() -> None:
    d = Path(sys.argv[1])
    torch_runs = sorted(p for p in d.glob("eval-*.json"))
    rayo_runs = sorted(p for p in d.glob("rayo-*.json"))
    print("## Torch, greedy, held-out people\n")
    print(
        "| model | n | JSON valid % | chart % | chart+data % | numbers grounded % | filler % | words | sentences | seconds |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    runs = [json.loads(p.read_text()) for p in torch_runs]
    for r in runs:
        m = r["metrics"]
        print(
            f"| {r['name']} | {m['n']} | {m['json_valid']} | {m['chart_choice']} | {m['chart_and_data']} "
            f"| {m['numbers_grounded']} | {m.get('filler', '')} | {m['words_mean']} | {m['sentences_mean']} | {r['seconds']} |"
        )
    for p in sorted(d.glob("train-log-*.txt")):
        lines = [x for x in p.read_text().splitlines() if x.strip()]
        print(f"\n{p.stem} (last lines):\n\n```\n" + "\n".join(lines[-6:]) + "\n```")
    for p in sorted(d.glob("train-*.json")):
        t = json.loads(p.read_text())
        print(f"\n{p.stem}: {t}")
    if rayo_runs:
        print("\n## rayo (converted), greedy\n")
        print(
            "| file | MB | kv | n | same as torch % | same JSON line % | JSON valid % | chart+data % | grounded % | seconds |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for p in rayo_runs:
            r = json.loads(p.read_text())
            m = r["metrics"]
            print(
                f"| {r['name']} | {r['size_bytes'] / 1e6:.1f} | {r['kv']} | {m['n']} | {r['same_as_torch']} "
                f"| {r['same_json_line_as_torch']} | {m['json_valid']} | {m['chart_and_data']} "
                f"| {m['numbers_grounded']} | {r['seconds']} |"
            )
    print("\n## Chart choice by kind of question (right/asked)\n")
    kinds = sorted({x["kind"] for r in runs for x in r["replies"]})
    print("| model | " + " | ".join(kinds) + " |")
    print("|---|" + "---|" * len(kinds))
    for r in runs:
        cells = []
        for k in kinds:
            xs = [x for x in r["replies"] if x["kind"] == k]
            ok = 0
            for x in xs:
                try:
                    c = json.loads(x["reply"].lstrip().split("\n")[0])
                    ok += (
                        c.get("chart") == x["gold"]["chart"] and c.get("data") == x["gold"]["data"]
                    )
                except (ValueError, AttributeError):
                    pass
            cells.append(f"{ok}/{len(xs)}")
        print(f"| {r['name']} | " + " | ".join(cells) + " |")
    for r in runs:
        print(f"\n## Samples: {r['name']}\n")
        for i, x in enumerate(r["replies"][:20], 1):
            reply = x["reply"].replace("\n", " ⏎ ")
            print(
                f"{i}. **{x['question']}** ({x['gold']['chart']}/{x['gold']['data']})  \n   {reply}"
            )


if __name__ == "__main__":
    main()
