"""Exit 0 when an evaluation passes the thresholds to ship (docs/SLM.md), else 1.

python tool/slm/passes.py EVAL_JSON [--strict]

``--strict`` is the bar for replacing a model that already ships (no regression on the current
Light model): JSON valid 100%, chart choice >= 90%, numbers grounded 100%, filler <= 0.5%.
"""

import json
import sys

THRESHOLDS = {"json_valid": 98.0, "chart_choice": 90.0, "numbers_grounded": 98.0}
STRICT = {"json_valid": 100.0, "chart_choice": 90.0, "numbers_grounded": 100.0}
MAX_FILLER = 0.5

with open(sys.argv[1]) as f:
    m = json.load(f)["metrics"]
strict = "--strict" in sys.argv[2:]
failed = {k: m[k] for k, v in (STRICT if strict else THRESHOLDS).items() if m[k] < v}
if strict and m.get("filler", 100.0) > MAX_FILLER:
    failed["filler"] = m.get("filler")
print("passes" if not failed else f"misses: {failed}")
sys.exit(1 if failed else 0)
