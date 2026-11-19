# pesaclarity-ml (akili)

Akili is the machine learning side of Pesa Clarity, the free, ad-free app for Android and iPhone that turns M-Pesa
and bank SMS into a daily plan. It holds the small models that run on the phone (trained in PyTorch and shipped as rayo's
weights; the frozen Ask writer is still Keras and TFLite): the
transaction categoriser, the message reader, and Ask's question reader and reply writer. Around
them sit a synthetic data generator, centralised and DP-SGD training, the client side of
federated learning, export with a size budget, MLflow tracking, and the privacy release gate
that every model must pass before it leaves the server. The federated aggregator lives in
[pesaclarity-backend](https://github.com/pesaclarity/pesaclarity-backend).

*Akili* is Swahili for intelligence or mind.

Status: **pre-alpha**. Everything here trains on synthetic data only.

## Quick start

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                   # core: data, configs, features, privacy gate
uv run pytest                             # core tests; TensorFlow and torch tests skip
uv run akili synth --out /tmp/tx.jsonl    # write synthetic transactions
uv run akili vectors                      # regenerate the feature test vectors for the app
uv run akili ledger                       # what every DP run so far has spent, per dataset

uv sync --extra torch                     # PyTorch (CPU), Opacus, onnx and MLflow
uv run akili epsilon --config configs/dp-sgd.yaml   # privacy accounting needs Opacus
uv run akili accountants --check          # dp-accounting's and Opacus's budgets side by side
uv run akili fine-tune-vectors            # regenerate the app's fine-tuning vectors
uv run akili train --config configs/smoke.yaml
uv run akili train --config configs/dp-sgd.yaml
uv run akili simulate --config configs/federated-smoke.yaml
uv run akili audit --quick --config configs/audit-no-dp.yaml --config configs/audit-dp-sgd.yaml

uv sync --extra train --extra fl          # TensorFlow, MLflow, Opacus and Flower: the Keras models
uv run akili train --keras --config configs/smoke.yaml   # the old Keras categoriser, until TensorFlow goes
```

Runs write rayo's int8 `categoriser.weights`, the ONNX file it came from,
`model-manifest.json`, `export-report.json`, `labels.json`, `privacy-report.json`,
`privacy-budget.json` (what every run on the same data has spent together) and, when audited,
`audit-report.json` to `artefacts/<name>/`. Every DP run, audit shadow models included, is added
to the budget ledger `artefacts/privacy-ledger.jsonl`. Runs
log to a local MLflow store in `mlruns/` (`uv run mlflow ui --backend-store-uri
sqlite:///mlruns/mlflow.db` to browse). Both folders are git-ignored.

## The model

The categoriser ([docs/CATEGORISER.md](docs/CATEGORISER.md)). Input is one int32 vector per transaction: up to 8 hashed payee tokens plus amount bucket,
direction, hour bucket, weekday and month phase (start, middle, payday end). Output is a softmax
over the default Kenyan category set in
[`src/akili/resources/categories.yaml`](src/akili/resources/categories.yaml) (version 3: income
splits into salary, business and freelance, money received, support from family, allowance and
bursary, and gifts; the app's default categories mirror it). 67,731 parameters, trained in
PyTorch and shipped as rayo's int8 `categoriser.weights` (75,776 bytes, against a 512 KB budget
in the configs; no TFLite file). The app fine-tunes its two dense layers on the person's own
choices by the step in [docs/FINE-TUNING.md](docs/FINE-TUNING.md). On the configs' held-out
synthetic people it scores level with or above the Keras model it replaced on every config
(DP-SGD per person: 0.932 against 0.923, the mean over ten noise seeds).
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
convolutional network, trained in PyTorch and shipped as rayo's int8 `reader.weights` (about
188 KB; no TFLite file), and trains on synthetic messages only, built from
invented institutions, people and amounts in eleven sentence layouts. The person confirms every
suggestion. [docs/READER.md](docs/READER.md) is the spec the app implements, with conformance
vectors in [`reader_vectors.v1.json`](src/akili/resources/reader_vectors.v1.json).

```sh
uv run akili reader --out artefacts/reader      # [torch] train, score unseen layouts, export
uv run akili reader-vectors                     # regenerate the app's test vectors
uv run akili reader-examples                    # regenerate docs/READER-EXAMPLES.md
```

`reader-report.json` holds each layout's scores when that layout was left out of training: the
honest measure for wording the app has never seen. Training covers eleven layouts: three
modelled on public M-PESA, Equity and KCB samples (receipt first, "Your transaction of ...", a
16-character code first), and one in the style of Sacco alerts (deposits and loan repayments,
"Dear Member, your deposit of ..."). On synthetic messages in an unseen layout the mean F1 is
about 0.85 for the amount, 0.72 for the payee and 1.00 for the date, scored on the int8 file
as rayo runs it (the Keras model it replaced: 0.79, 0.70 and 0.98); key-value messages
("Amount: ... Merchant: ..."), payee-first sentences ("JOHN DOE has sent you ..."), the
bank-to-wallet layouts and Sacco alerts (where the Sacco itself is the payee) are the weak
spots. Layouts it was trained on score close to 1. Real
messages will score lower.
[docs/READER-EXAMPLES.md](docs/READER-EXAMPLES.md) shows sample messages for each layout with their
fields marked.

## Ask

Ask answers questions about the person's own money in plain words, on the phone: "how much did
I spend on food last month?", "nimetumia pesa ngapi kwa fare wiki hii", "what sells best?". Two
small models and the app's own database lookups answer; nothing leaves the phone.

- The **ask reader** reads a question into one of 30 intents (spent, sent to, bills due,
  safe to spend, sales, low stock, follow-ups such as "and only Naivas?", greetings, out of
  scope, ...) and marks the words that name a category, payee, period, item, account or
  amount. Its input is the message reader's encoding unchanged, so the app's tokeniser serves
  both. A 1-D convolutional network like the message reader's: about 240k parameters,
  trained in PyTorch and shipped as rayo's int8 `ask-reader.weights` (about 253 KB, budget
  400 KB; no TFLite file), which the app runs on rayo, or in Dart
  ([docs/WEIGHTS.md](docs/WEIGHTS.md)).
- The app matches the marked words against its own records, parses the dates with rules and
  runs the lookup.
- The **ask writer**, a small causal transformer (about 530k parameters, about 650 KB
  as TFLite), writes a short, warm reply from fact tokens such as `kind=spent cmp=less {amount}
  {category} {period}`. Replies hold placeholders only, never a number; the app fills them,
  runs a checker and falls back to a fixed sentence when a reply fails it.

Both train on synthetic text only: questions from about 830 templates with Kenyan English,
Sheng and Swahili wording, typos and no punctuation, and thousands of invented payee and
account names; replies combined from about 1,000 hand-written fragments. [docs/ASK.md](docs/ASK.md) is the spec the app implements, with
conformance cases in [`ask_examples.v1.json`](src/akili/resources/ask_examples.v1.json).

```sh
uv run akili ask-reader --out artefacts/ask-reader --jobs 5  # [torch] train, export, score unseen templates
uv run akili ask-reader-probe --model path/to/ask-reader.weights  # score any exported reader
uv run akili ask-reader-weights --model path/to/ask-reader.tflite # a Keras reader's weights (--keras)
uv run akili ask-writer --out artefacts/ask-writer   # train, sample and check replies, export
uv run akili ask-examples                            # regenerate the app's conformance cases
```

`ask-reader-report.json` holds the scores ([docs/ASK.md](docs/ASK.md#training-and-scores)
has the table, against v1 and the Keras v3). Scores are of the int8 file as rayo runs it.
When a fifth of every intent's templates is left out of training, in five folds, intent
accuracy on the unseen version 1 phrasings is about 0.88 (v1: 0.83) and slot F1 about 0.86
(v1: 0.83; payee, category and item lower, period, account and amount higher). Many held-out templates are Sheng or Swahili phrasings with
words no other template uses, which no model can read from the rest. On a hard held-out set of
counts, sums for one payee, money in from someone, money in and account balances, with templates
and names training never sees, intent accuracy is 0.97 and the payee or account is marked
exactly in 0.94 of the questions naming one (v1: 0.64 and 0.11); on the questions in the app's
tests, 1.00 and 0.98 (v1: 0.82 and 0.35; the Keras v3: 1.00 and 0.96). It reads the intent of
67 of the 68 hand-written questions in the conformance file right, and 60 of 62 questions built on
the previous version's misses ("charges on KCB this month", "how much to john", "has Otieno paid me"). `ask-writer-report.json` samples 2,000 replies at temperature 0.8: about 99.4% pass the
checker (98.4% for `spent`, scored on its own; the rest fall back to the fixed sentence), and
the same facts give a different reply about seven times in ten. Real questions
will score lower; the app falls back to "Sorry, I did not get that" when the reader is unsure.

## Configs and tracks

| Config | What it runs |
| --- | --- |
| `configs/smoke.yaml` | Centralised training, no DP. Fast check that everything works. |
| `configs/dp-sgd.yaml` | Centralised DP-SGD per person; reports user-level epsilon and delta. |
| `configs/audit-no-dp.yaml` | Membership-inference audit baseline: varied payees, no DP. Fails. |
| `configs/audit-dp-sgd.yaml` | The same data and model with DP-SGD, audited with 16 shadow models. |
| `configs/federated-smoke.yaml` | Federated simulation without update noise; mechanics only. |
| `configs/federated-app.yaml` | DP-FedAvg simulation on the app track; user-level budget. |
| `configs/federated-commercial.yaml` | Commercial track: opted-in users only, cohort of 1,000+. |

The two training tracks from the spec, app (on by default) and commercial (separate opt-in), are
defined with their consent keys and release thresholds in
[`src/akili/resources/tracks.yaml`](src/akili/resources/tracks.yaml).

DP-SGD is implemented in [`train/torch_dp_sgd.py`](src/akili/train/torch_dp_sgd.py) with
`torch.func` per-person gradients (Opacus's own engine clips per example). Budgets are computed
with [Opacus](https://opacus.ai)'s accountants: RDP for the gate, PRV for the tighter figure
(decision 0023; [DP-SGD-REVIEW.md](docs/DP-SGD-REVIEW.md#2026-11-19-the-pytorch-loop-and-opacus)
compares them with Google's `dp-accounting`, which computed the figures before).

The privacy unit is one person, as in federated training: each step samples people, and each
sampled person's examples give one clipped gradient, so the budget covers everything they
contributed. Noise comes from OS entropy and is never logged; `dp.noise_seed` makes a synthetic
run reproducible, and the gate refuses any model trained with it. The ledger
([`privacy/ledger.py`](src/akili/privacy/ledger.py)) composes every DP run on the same data,
because each run spends the budget again.

## Release gate

No model leaves the server until `akili gate --evidence <file>` passes: a DP budget per person
(not per transaction) within the track limits, noise drawn from OS entropy rather than a fixed
seed, the data's total across every recorded run within the limits too, secure aggregation,
minimum cohort, a real membership-inference audit within the limits, a DPIA filed with the ODPC
and, for the commercial track, legal review. See
[docs/RELEASE-CHECKLIST.md](docs/RELEASE-CHECKLIST.md).

`akili audit` trains the model, then attacks it with a loss-threshold attack and LiRA (online
and offline, with shadow models trained the same way) and reports attack AUC, advantage and the
true-positive rate at 0.1% and 1% false positives, against limits signed off in
[#3](https://github.com/pesaclarity/pesaclarity-ml/issues/3). How to run it and read the
numbers: [docs/PRIVACY-AUDIT.md](docs/PRIVACY-AUDIT.md). Each privacy report also gives the
tighter PRV accountant's epsilon (`epsilon_pld`, the name kept from the PLD accountant it
replaced) next to the RDP one the gate checks.

## Project layout

```text
configs/                  experiment configs (YAML, validated by pydantic)
docs/ASK.md               Ask reader and writer spec for the app
docs/CATEGORISER.md       the categoriser: its file, scores and parity
docs/FEATURES.md          feature pipeline spec for the app
docs/FINE-TUNING.md       the categoriser's fine-tuning step, spec for the app
docs/PRIVACY-AUDIT.md     membership-inference audit: running it, reading it
docs/RELEASE-CHECKLIST.md release conditions for any model
docs/WEIGHTS.md           plain weights format the app's Dart models read
src/akili/
  resources/              categories.yaml, tracks.yaml, feature test vectors
  config.py               experiment config models
  features.py             feature extraction shared with the app
  synthetic.py            fake transactions for tests and smoke training
  vectors.py              feature test vectors (`akili vectors`)
  reader/                 the message reader: tokens, synthetic messages, the torch
                          model and training [torch] (keras_train.py: the old Keras
                          reader [train], until TensorFlow is removed)
  ask/                    Ask: questions, replies, reader and writer training, examples;
                          reader_model.py: the torch Ask reader [torch]
                          (keras_reader.py: the old Keras one [train])
  model.py                the categoriser in PyTorch (keras_model.py: the old Keras
                          one [train], until TensorFlow is removed)
  train/                  centralised and DP-SGD training: the PyTorch backend [torch]
                          (keras_pipeline.py, keras_dp_sgd.py: the Keras one [train])
  export/                 size budget, plain weights format, the fine-tuning step;
                          onnx.py: a torch model to rayo's weights via ONNX [torch];
                          tflite.py, trainable.py: the Keras exports [train]
  federated/              client, update maths, simulation, Flower adapter
  privacy/                privacy accounting (Opacus, in [torch] and [train]), budget
                          ledger, the accountants compared, membership-inference audit,
                          release gate
  tracking.py             MLflow wrapper                    [train]
  cli.py                  the `akili` command
tests/                    pytest; test_tf_* need the [train] extra, most test_torch_* [torch]
```

## Testing

```sh
tool/check          # everything: lint, types, all tests, smoke runs, the torch export without TensorFlow, then the core install
tool/check --fast   # format, lint, types and the tests that skip TensorFlow
tool/setup-hooks    # once: the pre-push hook runs tool/check
```

Checks run on your machine, not on GitHub; GitHub only runs releases. See
[CONTRIBUTING.md](CONTRIBUTING.md#checks-run-on-your-machine).

## Branches and releases

`main` is what model releases are trained and exported from, so it takes only tested, approved
changes.

1. Pull requests go to `dev` and are rebase-merged, so every commit lands on `dev` as is and
   each needs a [Conventional Commits](https://www.conventionalcommits.org) title (`feat: …`,
   `fix: …`, `feat!: …` for a breaking change; `docs`, `test`, `ci`, `chore`, `refactor`,
   `build`, `perf`, `style`, `revert` too). Use `feat` or `fix` for changes users see, and a
   hidden type (`refactor`, `test`, `style`, `chore`, `docs`, `ci`, `build`) for review
   follow-ups they don't, as new commits rather than `fixup!` ones. The pre-push hook checks
   your commit titles.
2. When `dev` is ready, open a pull request from `dev` to `main`. The `Release gate` check allows
   only `dev` (or the release pull request) into `main`, and its QA job waits for a reviewer in
   the `qa` environment. Merge it with a merge commit, not a squash, so each change keeps its own
   title.
3. On `main`, release-please keeps a release pull request open with the next version (SemVer from
   the commit titles: `feat` bumps the minor, `fix` the patch; before 1.0 a breaking change bumps the
   minor) in `pyproject.toml`, `uv.lock` and `akili.__version__`, and the `CHANGELOG.md` entry.
   Merging it tags `vX.Y.Z` and merges the release back into `dev`.

## Get involved

- **Star [the app](https://github.com/pesaclarity/pesaclarity-app)** so more people find a private money app that runs on their phone.
- Machine learning and privacy reviewers are welcome, especially a second look at the private training and its audits.
- **Pick a first issue:** [good first issues](https://github.com/pesaclarity/pesaclarity-ml/issues?q=is%3Aopen+label%3A%22good+first+issue%22) or [help wanted](https://github.com/pesaclarity/pesaclarity-ml/issues?q=is%3Aopen+label%3A%22help+wanted%22).
- **Read [CONTRIBUTING.md](CONTRIBUTING.md)** before your first pull request. Never include real messages, statements or phone numbers.

## Licence

Code: [AGPL-3.0](LICENSE). Trained model weights are not covered by the AGPL, and none are
committed here. App-track weights, the ones that ship in the app, are under
[CC BY-SA 4.0](LICENSE-WEIGHTS). Commercial-track weights are licensed only by written agreement on
the [commercial terms](COMMERCIAL-MODEL-TERMS.md). See [MODEL-WEIGHTS.md](MODEL-WEIGHTS.md).
The spec, brand and design decisions live in [pesaclarity/docs](https://github.com/pesaclarity/docs).
