# Contributing to akili

Thank you for helping. The code of conduct, security policy and general contribution terms are
the organisation defaults in [pesaclarity/.github](https://github.com/pesaclarity/.github);
please read them first. Report security or privacy issues privately as described there, not in
public issues.

## Setup

```sh
uv sync                            # core only, fast (no PyTorch, so no privacy accounting)
uv sync --extra torch              # PyTorch (CPU), Opacus, onnx and MLflow: train and export
                                   # the models, and the privacy accountant
uv sync --extra torch --extra fl   # and Flower, as tool/check runs it
```

## Checks run on your machine

There is no CI on GitHub; the checks run locally. Once per clone, enable the git hook:

```sh
tool/setup-hooks
```

The pre-push hook checks that the titles of the commits you push follow Conventional Commits
(see below), then runs `tool/check`: ruff format and lint, mypy, a check that TensorFlow is
not installed, all tests with the `[torch]` and `[fl]` extras (the ones that run rayo's C
library use rayo's sources at the pinned commit, fetched into `.cache/`), the smoke training,
DP-SGD, federated and quick-audit runs, then mypy and the tests on the core install alone (no
PyTorch or Opacus), and `akili accountants --check` with the extras. `tool/check --fast`
(format, lint, types and the tests your environment can run) is there for running by hand
while you work. In an emergency, `git push --no-verify` skips the
hook; run `tool/check` before the pull request is merged.

## Rules specific to this repository

- **Synthetic data only.** Never commit real SMS, statements, payee names, till numbers or phone
  numbers, not even in a test. Extend `synthetic.py` instead; its payees are marked DEMO, SAMPLE,
  FAKE or TEST and its numbers start with 000 or 0700.
- **No weights or run outputs.** `artefacts/`, `mlruns/` and model files stay out of git.
- **Features are a contract with the app.** A change to the output of `features.py` must bump
  `FEATURE_SCHEMA_VERSION`, update [docs/FEATURES.md](docs/FEATURES.md) and regenerate the
  vectors with `uv run akili vectors`; a change to `categories.yaml` must bump its `version`. Say
  so in the PR.
- **Privacy thresholds** in `tracks.yaml` may be tightened freely; loosening one needs a reason
  in the PR and a maintainer's approval. The membership-inference limits stay marked
  `provisional: true` until issue #3 signs them off.
- **No TensorFlow, Keras, TFLite or Google libraries** (decision 0023). Models train in
  PyTorch and ship as rayo's weights through ONNX (`export/onnx.py`). The old Keras code is at
  the `tflite-archive` tag; the frozen Ask writer is not retrained here.
- **Privacy code** (`privacy/`, `train/dp_sgd.py`, `train/torch_dp_sgd.py`): if you change an attack or the training
  procedure, run the full audit pair (see [docs/PRIVACY-AUDIT.md](docs/PRIVACY-AUDIT.md)) and put
  the before and after tables in the PR.

## Commits and pull requests

- Small, focused pull requests against `dev`: one topic each, with tests for new behaviour and
  a passing `tool/check`. `main` takes only `dev` and release pull requests (see
  [Branches and releases](README.md#branches-and-releases)).
- Every commit follows [Conventional Commits](https://www.conventionalcommits.org): one short
  line such as `feat: add a Sacco layout`, `fix: …`, `feat!: …` for a breaking change. Pull
  requests into `dev` are rebase-merged, so each commit lands on `dev` as is and the commits set
  the next version and the changelog. The pre-push hook checks them; it also
  accepts git's own `Revert "…"` titles and skips merge commits
  (`tool/commit-titles origin/dev..HEAD` checks by hand).
- Changes users can see are `feat` or `fix`. Review follow-ups they don't see use a hidden type
  (`refactor`, `test`, `style`, `chore`, `docs`, `ci`, `build`) so the changelog stays clean.
  Write them as new commits: pushed history is never rewritten, so no `fixup!`, `squash!` or
  `amend!` commits.
- One logical change per commit.
