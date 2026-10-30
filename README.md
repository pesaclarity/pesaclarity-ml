# pesaclarity-ml (akili)

Akili is the machine learning side of Pesa Clarity, the free, ad-free Android app (with an iPhone version in the works) that turns M-Pesa
and bank SMS into a daily plan. It holds the small transaction categoriser that runs on the phone
(TensorFlow Lite), a synthetic data generator, centralised and DP-SGD training, the client side of
federated learning, TFLite export with a size budget, MLflow tracking, and the privacy release gate
that every model must pass before it leaves the server. The federated aggregator lives in
[pesaclarity-backend](https://github.com/pesaclarity/pesaclarity-backend).

*Akili* is Swahili for intelligence or mind.

Status: **pre-alpha**. Everything here trains on synthetic data only.

## Quick start

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                   # core: data, configs, features, privacy gate
uv run pytest                             # core tests; TensorFlow tests skip
uv run akili synth --out /tmp/tx.jsonl    # write synthetic transactions
uv run akili vectors                      # regenerate the feature test vectors for the app
uv run akili epsilon --config configs/dp-sgd.yaml

uv sync --all-extras                      # add TensorFlow, MLflow and Flower
uv run akili train --config configs/smoke.yaml
uv run akili train --config configs/dp-sgd.yaml
uv run akili simulate --config configs/federated-smoke.yaml
uv run akili audit --quick --config configs/audit-no-dp.yaml --config configs/audit-dp-sgd.yaml
```

Runs write the TFLite model, `model-manifest.json`, `labels.json`, `privacy-report.json` and,
when audited, `audit-report.json` to `artefacts/<name>/`, and
log to a local MLflow store in `mlruns/` (`uv run mlflow ui --backend-store-uri
sqlite:///mlruns/mlflow.db` to browse). Both folders are git-ignored.

## The model

Input is one int32 vector per transaction: up to 8 hashed payee tokens plus amount bucket,
direction, hour bucket, weekday and month phase (start, middle, payday end). Output is a softmax
over the default Kenyan category set in
[`src/akili/resources/categories.yaml`](src/akili/resources/categories.yaml). About 70k
parameters; the quantised TFLite file is about 80 KB, against a 512 KB budget in the configs.
The app must reproduce [`features.py`](src/akili/features.py) exactly in Dart; it uses FNV-1a
hashing so no payee vocabulary ships with the model. [docs/FEATURES.md](docs/FEATURES.md) is the
language-neutral spec, and
[`feature_vectors.v1.json`](src/akili/resources/feature_vectors.v1.json) holds the conformance
cases the Dart code must pass. Each exported model carries a `model-manifest.json` with the
feature schema version, so the app can refuse a model it cannot feed.

## The message reader

A second, separate model reads the fields of bank and wallet messages that no parser rule
matched, for the app's review inbox: amount, payee, date, time, balance, reference, fee, and
whether it is money out, money in or not a transaction. It tags each token with a small 1-D
convolutional network (about 200 KB as TFLite) and trains on synthetic messages only, built from
invented institutions, people and amounts in seven sentence layouts. The person confirms every
suggestion. [docs/READER.md](docs/READER.md) is the spec the app implements, with conformance
vectors in [`reader_vectors.v1.json`](src/akili/resources/reader_vectors.v1.json).

```sh
uv run akili reader --out artefacts/reader      # train, score unseen layouts, export
uv run akili reader-vectors                     # regenerate the app's test vectors
uv run akili reader-examples                    # regenerate docs/READER-EXAMPLES.md
```

`reader-report.json` holds each layout's scores when that layout was left out of training: the
honest measure for wording the app has never seen. Training covers ten layouts, three of them
modelled on public M-PESA, Equity and KCB samples (receipt first, "Your transaction of ...", a
16-character code first). On synthetic messages in an unseen layout the mean F1 is about 0.85 for
the amount, 0.72 for the payee and 0.99 for the date; payee-first sentences ("JOHN DOE has sent
you ...") and the bank-to-wallet layouts are the weak spots. Layouts it was trained on score close
to 1. Real messages will score lower.
[docs/READER-EXAMPLES.md](docs/READER-EXAMPLES.md) shows sample messages for each layout with their
fields marked.

## Configs and tracks

| Config | What it runs |
| --- | --- |
| `configs/smoke.yaml` | Centralised training, no DP. Fast check that everything works. |
| `configs/dp-sgd.yaml` | Centralised DP-SGD; reports example-level epsilon and delta. |
| `configs/audit-no-dp.yaml` | Membership-inference audit baseline: varied payees, no DP. Fails. |
| `configs/audit-dp-sgd.yaml` | The same data and model with DP-SGD, audited with 16 shadow models. |
| `configs/federated-smoke.yaml` | Federated simulation without update noise; mechanics only. |
| `configs/federated-app.yaml` | DP-FedAvg simulation on the app track; user-level budget. |
| `configs/federated-commercial.yaml` | Commercial track: opted-in users only, cohort of 1,000+. |

The two training tracks from the spec, app (on by default) and commercial (separate opt-in), are
defined with their consent keys and release thresholds in
[`src/akili/resources/tracks.yaml`](src/akili/resources/tracks.yaml).

DP-SGD is implemented in [`train/dp_sgd.py`](src/akili/train/dp_sgd.py) rather than with
tensorflow-privacy, whose last release supports neither Python 3.12 nor Keras 3. Budgets are
computed with Google's `dp-accounting` (the accountant tensorflow-privacy uses).

## Release gate

No model leaves the server until `akili gate --evidence <file>` passes: DP budget within the
track limits, secure aggregation, minimum cohort, a real membership-inference audit within the
limits, a DPIA filed with the ODPC and, for the commercial track, legal review. See
[docs/RELEASE-CHECKLIST.md](docs/RELEASE-CHECKLIST.md).

`akili audit` trains the model, then attacks it with a loss-threshold attack and LiRA (online
and offline, with shadow models trained the same way) and reports attack AUC, advantage and the
true-positive rate at 0.1% and 1% false positives. The audit limits are **provisional** until
[#3](https://github.com/pesaclarity/pesaclarity-ml/issues/3) reviews them. How to run it and read
the numbers: [docs/PRIVACY-AUDIT.md](docs/PRIVACY-AUDIT.md).

## Project layout

```text
configs/                  experiment configs (YAML, validated by pydantic)
docs/FEATURES.md          feature pipeline spec for the app
docs/PRIVACY-AUDIT.md     membership-inference audit: running it, reading it
docs/RELEASE-CHECKLIST.md release conditions for any model
src/akili/
  resources/              categories.yaml, tracks.yaml, feature test vectors
  config.py               experiment config models
  features.py             feature extraction shared with the app
  synthetic.py            fake transactions for tests and smoke training
  vectors.py              feature test vectors (`akili vectors`)
  model.py                the Keras categoriser            [train]
  train/                  centralised and DP-SGD training   [train]
  export/                 TFLite export and size budget
  federated/              client, update maths, simulation, Flower adapter
  privacy/                privacy accounting, membership-inference audit (mia.py [train]),
                          release gate
  tracking.py             MLflow wrapper                    [train]
  cli.py                  the `akili` command
tests/                    pytest; test_tf_* need the [train] extra
```

## Testing

```sh
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest
```

CI runs this on the core install, then again with `--all-extras` plus the smoke training runs.

## Licence

Code: [AGPL-3.0](LICENSE). Trained model weights are not covered by the AGPL and will be released
under a separate model licence; none are committed here. See [MODEL-WEIGHTS.md](MODEL-WEIGHTS.md).
The spec, brand and design decisions live in [pesaclarity/docs](https://github.com/pesaclarity/docs).
