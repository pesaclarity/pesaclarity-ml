"""Exit 0 when an evaluation passes the thresholds to ship (docs/SLM.md), else 1.

python tool/slm/passes.py EVAL_JSON
"""

import json
import sys

THRESHOLDS = {"json_valid": 98.0, "chart_choice": 90.0, "numbers_grounded": 98.0}

with open(sys.argv[1]) as f:
    m = json.load(f)["metrics"]
failed = {k: m[k] for k, v in THRESHOLDS.items() if m[k] < v}
print("passes" if not failed else f"misses: {failed}")
sys.exit(1 if failed else 0)
