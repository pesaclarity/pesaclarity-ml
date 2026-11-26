"""Downloads a Hugging Face model at a fixed revision and checks every file against the hashes
Hugging Face lists for that revision (LFS files by SHA-256, small ones by their git blob hash),
as the app's slm-model.yml does. Writes SOURCES (sha256  name) next to the files.

    python tool/slm/fetch_model.py HuggingFaceTB/SmolLM2-135M-Instruct REVISION OUT_DIR
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model.safetensors",
)


def get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=600) as r:
        return r.read()


def main() -> None:
    model, rev, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    if rev == "main":  # only to find the hash to pin: it is printed, then used
        rev = json.loads(get(f"https://huggingface.co/api/models/{model}/revision/main"))["sha"]
        print(f"main is revision {rev}: pin it")
    if len(rev) != 40:
        sys.exit(f"pin a full revision hash, not {rev!r}")
    out.mkdir(parents=True, exist_ok=True)
    tree = {
        e["path"]: e
        for e in json.loads(get(f"https://huggingface.co/api/models/{model}/tree/{rev}"))
    }
    sums = []
    for name in FILES:
        data = get(f"https://huggingface.co/{model}/resolve/{rev}/{name}")
        entry = tree[name]
        sha256 = hashlib.sha256(data).hexdigest()
        if "lfs" in entry:
            want, got = entry["lfs"]["oid"], sha256
        else:
            want, got = entry["oid"], hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        if want != got:
            sys.exit(f"{name}: expected {want}, got {got}")
        (out / name).write_bytes(data)
        sums.append(f"{sha256}  {name}")
        print(f"{sha256}  {name}")
    (out / "SOURCES").write_text("\n".join(sums) + "\n")
    (out / "REVISION").write_text(rev + "\n")


if __name__ == "__main__":
    main()
