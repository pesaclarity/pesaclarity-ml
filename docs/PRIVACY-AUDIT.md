# Membership-inference audit

Before any model leaves the server, the spec requires that "a membership-inference audit
passes". This page says what the audit does, how to run it and how to read its numbers. The
code is in [`privacy/audit.py`](../src/akili/privacy/audit.py) (metrics, attacks, limits; numpy
only) and [`privacy/mia.py`](../src/akili/privacy/mia.py) (shadow training; needs `[train]`).

Everything runs on synthetic data from [`synthetic.py`](../src/akili/synthetic.py). Never point
the audit at real messages; when opted-in evaluation data exists it will be audited on the
server, not in this repository.

## What it asks

*Given the released model and a transaction, can an attacker tell whether that transaction was
in the training set?* If they can, the model leaks something about the people who trained it.
DP-SGD bounds this in theory; the audit measures it in practice, against the actual model file.

## How it works

1. **Candidates.** Up to `audit.max_candidates` (default 2,000) held-out validation examples, the
   non-members, and as many random training examples, the members.
2. **Target scores.** The exported TFLite file, the one that would ship, gives each candidate the
   probability of its true category. (`audit.score_exported: false` uses the Keras model.)
3. **Shadow models.** `audit.shadow_models` (default 16) models trained exactly like the target,
   with the same config, DP-SGD included, on as many examples as the target saw. Each candidate
   is in the training set of exactly half of them; the rest is filled from the target's other
   data. This models a strong attacker who knows the training procedure and most of the data,
   which is what an audit should assume.
4. **Attacks.** Three, all scored from the true-label probability:

   | Attack | Idea | Needs |
   | --- | --- | --- |
   | `loss` | Members have lower loss (Yeom et al. 2018). Same ranking as a confidence threshold. | Target only |
   | `lira-online` | Likelihood ratio of the target's logit-scaled confidence under Gaussians fitted to IN and OUT shadow models (LiRA, Carlini et al. 2022). | IN and OUT shadows |
   | `lira-offline` | One-sided test against the OUT shadows only: how unusually confident is the target on this example? | OUT shadows |

   With fewer than 32 shadows LiRA uses one variance pooled over all examples, as the paper
   recommends for small shadow counts.

5. **Limits.** Every attack is checked against the track's limits; the strongest attack decides.

## Running it

```sh
uv sync --all-extras

# Quick: small data, 4 shadow models, under a minute for the pair on a laptop CPU.
# Too few examples to ever pass; use it to check the attack runs and moves the right way.
uv run akili audit --quick --config configs/audit-no-dp.yaml --config configs/audit-dp-sgd.yaml

# Full: 2,000 + 2,000 candidates, 16 shadow models each. About 5 minutes for the pair on 4 CPU
# cores; the DP-SGD half takes longest, since every shadow is trained with DP-SGD.
uv run akili audit --config configs/audit-no-dp.yaml --config configs/audit-dp-sgd.yaml

# Any centralised config can be audited, and any config with `audit.enabled: true` is audited
# at the end of `akili train`.
uv run akili audit --config configs/dp-sgd.yaml --shadow-models 8
```

`akili audit` exits 1 if any audited model fails its limits (`--report-only` always exits 0; CI
uses it for the quick run). Each run writes `artefacts/<name>/audit-report.json` and logs to
MLflow:

| MLflow | Meaning |
| --- | --- |
| `audit.<attack>.auc`, `.advantage`, `.tpr_at_1pct_fpr`, `.tpr_at_0_1pct_fpr` | Per-attack metrics |
| `audit.passed` (metric 0/1 and tag) | Whether every attack is within the limits |
| `audit.limits_provisional` (tag) | The limits still await review |
| `audit-report.json` (artefact) | The full report |

For a release, pass the report to the gate with the rest of the evidence:

```sh
uv run akili gate --evidence release-evidence.json --audit artefacts/<name>/audit-report.json
```

## Reading the numbers

A full run of the audit pair (2,000 + 2,000 candidates, 16 shadow models, synthetic data):

```text
audit-no-dp (no DP)
  attack           AUC      AUC 95% CI    adv  TPR@0.1%   TPR@1%           n
  loss           0.543     0.525-0.561  0.080     0.00%    0.95%  2000+2000
  lira-online    0.562     0.544-0.580  0.088     2.15%    3.75%  2000+2000
  lira-offline   0.575     0.558-0.593  0.122     1.45%    2.75%  2000+2000
  FAIL
audit-dp-sgd (DP-SGD, epsilon 1.83)
  attack           AUC      AUC 95% CI    adv  TPR@0.1%   TPR@1%           n
  loss           0.499     0.481-0.517  0.015     0.00%    0.00%  2000+2000
  lira-online    0.498     0.480-0.516  0.018     0.10%    0.75%  2000+2000
  lira-offline   0.496     0.478-0.514  0.023     0.15%    0.70%  2000+2000
  PASS
```

Without DP the loss attack barely beats chance, but LiRA picks out 2.15% of members at a 0.1%
false-positive rate, about 21 times a random guess: the model has memorised some rare
transactions. With DP-SGD every attack is at chance.

* **AUC** is the chance that a random member scores above a random non-member. 0.5 is a coin
  toss. It summarises the whole curve, so it can look harmless while a few examples are exposed.
* **AUC 95% CI** (Hanley and McNeil) shows how much of the AUC is sampling noise. With 2,000 per
  side it is about plus or minus 0.02; with the quick run's 300, about 0.05. It is for reading
  only; the gate uses the point estimate.
* **adv** (advantage) is the best TPR minus FPR over all thresholds (Yeom et al.), in [0, 1]: how
  much better than guessing the attacker does at their best operating point. Even a perfectly
  private model shows some advantage from sampling noise: at 2,000 per side about 0.03 on
  average, 0.043 at the 95th percentile; at the quick run's 300 per side, 0.07 and 0.11.
* **TPR@0.1%** and **TPR@1%** are the share of members the attacker identifies while wrongly
  accusing at most 0.1% or 1% of non-members. A random guess scores exactly the FPR (0.10% and
  1.00%), so compare against those. This is the number that matters most (Carlini et al.): an
  attacker who is confidently right about a few people is a real privacy failure even if the AUC
  is 0.51. With *n* non-members the smallest non-zero FPR is 1/*n*, which is why the gate wants at
  least `min_audit_examples` of each; below 1,000, TPR@0.1% is the TPR at zero false positives.
* **n** is members + non-members attacked.

Rules of thumb: DP-SGD with a small epsilon should bring every attack close to chance (AUC
within its interval of 0.5, TPR near the FPR). The no-DP baseline shows what the attacks can
find when nothing protects the data; if it is also at chance, the audit setup (data, attacker)
is too weak to say anything, and the DP result should not be trusted as evidence either.
`payee_style: varied` exists so the synthetic data has the long tail of rare merchants and
people that real statements have and a model can memorise.

## Limits, and why they are provisional

The limits live with the other track thresholds in
[`resources/tracks.yaml`](../src/akili/resources/tracks.yaml):

| Limit | App | Commercial |
| --- | --- | --- |
| `max_attack_auc` | 0.60 | 0.55 |
| `max_attack_advantage` | 0.10 | 0.05 |
| `max_attack_tpr_at_1pct_fpr` | 0.05 | 0.03 |
| `max_attack_tpr_at_0_1pct_fpr` | 0.01 | 0.005 |
| `min_audit_examples` (members and non-members each) | 1,000 | 2,000 |

**All of these are provisional** until the review in
[pesaclarity-ml#3](https://github.com/pesaclarity/pesaclarity-ml/issues/3) signs them off; the
`provisional: true` flag makes the gate and every audit report say so. Points for that review:

* The commercial limits sit close to the sampling noise of a perfectly private model at 2,000
  per side: advantage 0.05 against a null 95th percentile of 0.043 (99th: 0.050), and TPR@0.1%
  0.005 against a null 95th percentile of 0.004. Three attacks are checked, which adds multiple
  comparisons. A clean model can fail by chance; either raise `min_audit_examples` (the noise
  shrinks with the square root of *n*) or relax those limits.
* DP theory bounds TPR by e^epsilon x FPR + delta, which is vacuous at the app track's epsilon of
  8, so the TPR limits are empirical, not derived from the budget.
* The attacker model (knows the procedure and all but the target example) is deliberately
  strong. Agree it, and whether to add attacks (for example user-level inference for federated
  models), before the first release.

A config's `audit:` section can tighten any limit for one experiment (`max_attack_auc`,
`max_attack_advantage`, `max_attack_tpr_at_1pct_fpr`, `max_attack_tpr_at_0_1pct_fpr`,
`min_audit_examples`); looser values there are ignored. Loosening a track limit needs a reason in
the PR and a maintainer's approval (see [CONTRIBUTING](../CONTRIBUTING.md)).

## Scope

The audit covers centralised training (`akili train`, `akili audit`). Federated runs are not
audited yet: a user-level attack against DP-FedAvg needs per-user shadow federations, which is
follow-up work. The gate still refuses any model without an audit.

## References

* Yeom, Giacomelli, Fredrikson, Jha. *Privacy Risk in Machine Learning: Analyzing the Connection
  to Overfitting.* CSF 2018.
* Carlini, Chien, Nasr, Song, Terzis, Tramer. *Membership Inference Attacks From First
  Principles.* IEEE S&P 2022.
* Hanley, McNeil. *The meaning and use of the area under a ROC curve.* Radiology 1982.
