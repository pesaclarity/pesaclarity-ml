# Contributing to akili

Thank you for helping. The code of conduct, security policy and general contribution terms are
the organisation defaults in [pesaclarity/.github](https://github.com/pesaclarity/.github);
please read them first. Report security or privacy issues privately as described there, not in
public issues.

## Setup

```sh
uv sync                 # core only, fast
uv sync --all-extras    # with TensorFlow, MLflow and Flower
```

## Before you push

```sh
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

If you touch training, export or the federated client, also run the smoke configs with the extras
installed (`akili train --config configs/smoke.yaml`, `akili simulate --config
configs/federated-smoke.yaml`).

## Rules specific to this repository

- **Synthetic data only.** Never commit real SMS, statements, payee names, till numbers or phone
  numbers, not even in a test. Extend `synthetic.py` instead; its payees are marked DEMO, SAMPLE,
  FAKE or TEST and its numbers start with 000 or 0700.
- **No weights or run outputs.** `artefacts/`, `mlruns/` and model files stay out of git.
- **Features are a contract with the app.** A change to `features.py` or `categories.yaml`
  changes what the Dart side must compute; bump the category set `version` and say so in the PR.
- **Privacy thresholds** in `tracks.yaml` may be tightened freely; loosening one needs a reason
  in the PR and a maintainer's approval.

## Commits and pull requests

Small, focused commits with short imperative messages. One topic per pull request, with tests for
new behaviour and green CI.
