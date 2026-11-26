"""Runs a converted model on rayo (its chat program, greedy) on held-out rows, compares the
replies with the torch model's greedy replies and scores them as the evaluation does.

    python tool/slm/rayo_check.py CHAT_BIN MODEL.weights TEST_JSONL TORCH_JSON OUT_JSON \
        [--n 50] [--kv int8] [--name label]
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from akili.slm.evaluate import MAX_NEW, score


def main() -> None:
    ap = argparse.ArgumentParser()
    for x in ("chat", "weights", "test", "torch", "out"):
        ap.add_argument(x)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kv", default="int8")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.test).read_text().splitlines()][: a.n]
    torch_replies = [r["reply"] for r in json.loads(Path(a.torch).read_text())["replies"]][: a.n]
    replies = []
    start = time.time()
    for r in rows:
        p = subprocess.run(
            [
                a.chat,
                a.weights,
                "--system",
                r["system"],
                "--temp",
                "0",
                "--repeat-penalty",
                "1",
                "--max-tokens",
                str(MAX_NEW),
                "--context",
                "2048",
                "--kv",
                a.kv,
            ],
            input=r["question"] + "\n",
            capture_output=True,
            text=True,
            timeout=600,
            check=True,
        )
        replies.append(p.stdout.strip())
    same = sum(x.strip() == y.strip() for x, y in zip(replies, torch_replies, strict=True))
    first_line = sum(
        x.strip().split("\n")[0] == y.strip().split("\n")[0]
        for x, y in zip(replies, torch_replies, strict=True)
    )
    result = {
        "name": a.name or Path(a.weights).name,
        "size_bytes": Path(a.weights).stat().st_size,
        "kv": a.kv,
        "metrics": score(rows, replies),
        "same_as_torch": round(100 * same / len(rows), 1),
        "same_json_line_as_torch": round(100 * first_line / len(rows), 1),
        "seconds": round(time.time() - start),
        "replies": [
            {"question": r["question"], "reply": x, "torch": y}
            for r, x, y in zip(rows, replies, torch_replies, strict=True)
        ],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "replies"}))


if __name__ == "__main__":
    main()
