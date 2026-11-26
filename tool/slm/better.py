"""Exit 0 when evaluation A scores at least as well as B (chart choice first), else 1.

    python tool/slm/better.py A_JSON B_JSON
"""

import json
import sys


def key(path: str) -> tuple[float, float, float]:
    with open(path) as f:
        m = json.load(f)["metrics"]
    return (m["chart_choice"] + m["chart_and_data"], m["json_valid"], m["numbers_grounded"])


a, b = key(sys.argv[1]), key(sys.argv[2])
print(f"{sys.argv[1]} {a} vs {sys.argv[2]} {b}")
sys.exit(0 if a >= b else 1)
