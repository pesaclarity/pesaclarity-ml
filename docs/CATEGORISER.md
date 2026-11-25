# The categoriser

The categoriser suggests a category for each transaction, on the phone. Its features are
[FEATURES.md](FEATURES.md); its training, privacy and audits are in the
[README](../README.md#configs-and-tracks), [DP-SGD-REVIEW.md](DP-SGD-REVIEW.md) and
[PRIVACY-AUDIT.md](PRIVACY-AUDIT.md). This page is the model, its file, its scores and its
parity. Since decision 0023's stage 4 it trains in PyTorch and ships as rayo's int8
`categoriser.weights`. The Keras model it replaced, and its TFLite files, were removed with
TensorFlow in stage 5 (the code is at the `tflite-archive` tag, the files in its release).

## The model and its file

[`akili.model.Categoriser`](../src/akili/model.py), the Keras network (`akili.keras_model`, at
the `tflite-archive` tag) layer for layer under the same names:

| Layer | What | Default size |
| --- | --- | --- |
| `token_embedding` | the payee's hashed tokens, averaged over those that are not padding (id 0) | 4096 x 16 |
| `context_0` .. `context_4` | amount bucket, direction, hour bucket, weekday, month phase | 12, 2, 6, 7, 3 rows x 4 |
| `hidden` | dense, ReLU, on the mean and the five context embeddings (36); dropout 0.1 after it while training | 36 to 32 |
| `category` | dense, softmax over `labels.json` | 32 to 27 |

67,731 parameters with the default config, the Keras model's count. The input is the app's
row, `features` int32 `[13]` (token ids, then the five context columns); the exported graph
takes that row and slices it, so rayo gets what the app builds today.

The export ([`akili.train.torch_backend`](../src/akili/train/torch_backend.py)): `torch.onnx`
at opset 17, then rayo-convert with `Options(weights="int8")` (decision 0023), which quantises
`token_embedding.table` (one scale) and `hidden.kernel` (one scale per output) to int8 and keeps
the tensors of fewer than 1,024 values float32 (the context tables and `category`), as
TFLite's converter did. `categoriser.weights` is **75,776 bytes** against the 512,000-byte
budget of the configs (`export.max_size_bytes`, now on this file; the Keras TFLite file was
82,376). `run_dir` gets:

| File | |
| --- | --- |
| `categoriser.weights` | what the app runs (PCWT version 1 with rayo's graph, [WEIGHTS.md](WEIGHTS.md)) |
| `categoriser.onnx` | the float graph it was converted from, named in the weights' `source` |
| `model-manifest.json` | `weights_file`, `weights_sha256`, `weights_size_bytes`, `weights_format_version`, `source` (the ONNX file's name, SHA-256 and size), the feature fields as before (`feature_schema_version`, `feature_config`, `input`, `layout`), `rayo_inputs`, `outputs`, and `fine_tune` |
| `export-report.json` | size, SHA-256, parameters, the torch, onnx and rayo-convert versions, and the parity check |
| `labels.json`, `privacy-report.json`, `privacy-budget.json`, `audit-report.json` | as before |

There is no TFLite file and no `categoriser-train.tflite`: the manifest's `fine_tune` block
replaces its `trainable` block, and [FINE-TUNING.md](FINE-TUNING.md) specifies the dense
layers' SGD step the app implements, with vectors torch computes.

Training is seeded and deterministic (one thread, torch's deterministic algorithms): the same
code, torch version and machine give the same file. Two runs of `smoke`, `audit-no-dp` and
`audit-dp-sgd` gave the same SHA-256 (`7a683e82…`, `edd7d21a…`, `111fbaa5…`; torch
2.15.0+cpu, onnx 1.23.2, rayo-convert at 054c831).

## Training

The Keras recipe in torch, with one change each way, both chosen on these synthetic configs:

- **Without DP**: Adam (Keras' epsilon) on cross-entropy, the rows shuffled each epoch, the
  configs' learning rate, epochs and batch, **with the learning rate lowered to 0 along a
  cosine** over the run (`training.lr_schedule: cosine`; `constant` is the Keras recipe).
- **With DP-SGD** ([`akili.train.torch_dp_sgd`](../src/akili/train/torch_dp_sgd.py)): the same
  per-person clipping, noise, Poisson sampling, SGD with momentum 0.9 and constant learning
  rate, **releasing an exponential moving average of the weights over the steps**
  (`dp.weight_average: 0.99`, its decay ramping up over the first steps; 0 is the Keras recipe).
  The average is computed from the noisy steps alone, so it costs no privacy
  ([DP-SGD-REVIEW.md](DP-SGD-REVIEW.md#2026-11-19-the-pytorch-loop-and-opacus)).

Tried and dropped: label smoothing 0.1 (lower everywhere), the cosine with DP-SGD (DP needs
large steps to the end: 2 points lower), the average without DP (no gain).

| Recipe, PyTorch (trained model) | smoke | dp-sgd, 3 noise seeds | audit-no-dp | audit-dp-sgd |
| --- | --- | --- | --- | --- |
| Keras recipe as is | 0.921 | 0.926 | 0.921 | 0.786 |
| cosine | **0.928** | 0.905 | **0.928** | 0.761 |
| label smoothing 0.1 | 0.920 | 0.920 | 0.923 | 0.762 |
| weight average 0.99 | 0.926 | **0.932** | 0.926 | **0.810** |
| **cosine without DP, average with DP (shipped)** | **0.928** | **0.932** | **0.928** | **0.810** |

## Scores

`val.accuracy` on the configs' held-out people (whole people, never trained on), with the
configs' data and seeds, against the Keras pipeline at `origin/dev` (cb27820). The PyTorch
figures are of the trained model and, in brackets, of `categoriser.weights` on rayo's C library
(the Keras figures are of the Keras model, as the pipeline logged them).

| Config | Keras | PyTorch (int8 file on rayo) |
| --- | --- | --- |
| `smoke` | 0.9275 | 0.9275 (0.9275) |
| `dp-sgd`, mean of noise seeds 1 to 10 (range) | 0.9233 (0.913 to 0.930) | **0.9319** (0.9320; 0.926 to 0.936) |
| `dp-sgd` as configured, noise from OS entropy (one run) | | 0.9353 (0.9355) |
| `audit-no-dp` | 0.9238 | **0.9277** (0.9276) |
| `audit-dp-sgd` (noise seed 1001) | 0.7881 | **0.8102** (0.8095) |
| `federated-smoke` | 0.6413 | **0.6992** (0.7000) |
| `federated-app` | 0.6170 | **0.7009** (0.7013) |
| `federated-commercial` | 0.5537 | **0.5922** (0.5923) |
| `feature_vectors.v1.json` | exact | exact (features are unchanged; the 60 vectors are also parity rows) |

With the same noise seed, every one of the ten DP-SGD runs is above the Keras one (by 0.1 to
1.7 points). The privacy budgets are unchanged (`dp-sgd` and `audit-dp-sgd` ε 2.666 at δ 1e-6;
`federated-app` 5.616), and the DP-SGD model is audited within the app limits
([PRIVACY-AUDIT.md](PRIVACY-AUDIT.md#reading-the-numbers)). Scores are on synthetic data; real
transactions will score lower.

## Statement payees

*2026-11-25.* The app gives the categoriser the counterparty's name as its readers parse it:
`EXAMPLE GROCERS` from an M-PESA till SMS, `EXAMPLE SUPERMARKET` from a KCB statement's POS
line, and, where a statement's line is not split, its whole details column (`MERCHANT PAYMENT
TO ...`, `PAY BILL TO 000123 - ...`). The features are the same for SMS and statement rows. But
every payee the generator wrote ended in its till, paybill or phone (`DEMO GROCERS TILL
000012`, `TEST PERSON 12 0700***012`); only airtime, charges and Fuliza repayments had none,
and only people had `TEST`. The model learnt those suffixes and markers as the category: a name
without a number read as airtime, fees or Fuliza, `GROCERS` alone as airtime (1.00), and a word
it never saw (`EXAMPLE`, `NAIVAS`) landed on token rows the DP noise had moved, so the web
analyser's parity tests on the app's synthetic statements found grocers as family support and
fuel stations as Fuliza repayments.

`payee_format: app` (`SyntheticDataConfig`) writes what the app gives: the name alone 60% of the
time, a statement's details 25% (`MERCHANT PAYMENT TO`, `POS TXN`, `PAY BILL TO 000nnn -`,
`CUSTOMER TRANSFER TO 0700***0nn -`) and the SMS form 15%, with any of the four fake markers on
any category. The amounts, times and labels are the `sms` format's (the forms take their own
random stream). `dp-sgd` and both audit configs use it with `payee_style: varied`.

The harder data needed more signal per step: DP-SGD with the old settings scored 0.68 on it.
On scratch datasets with seeded noise, more people per step and more steps were what helped
(weight decay on the noisy steps, tried to hold unused token rows near zero, cost 25 to 65
points):

| `dp-sgd` settings | ε (δ 1e-6) | held-out people | statement payees |
| --- | --- | --- | --- |
| 100 people a step, σ 1.5, 8 epochs, lr 0.5 (before) | 2.67 | 0.68 to 0.73 | 0.39 to 0.66 |
| 400, σ 3.0, 16 epochs, lr 1.0 | 3.24 | 0.78 to 0.79 | 0.73 to 0.74 |
| **400, σ 3.0, 24 epochs, lr 1.0 (shipped then)**, four noise seeds | **4.01** | **0.85 to 0.86** | **0.76 to 0.87** |
| 400, σ 3.0, 32 epochs, lr 1.0 (shipped since [the section below](#transport-salary-and-money-from-people)) | 4.66 | 0.86 | 0.87 |

ε 4.01 is within the app track's signed-off 8.0 (`tracks.yaml`, unchanged); the run's data
(seed 2026, not the audits' seed 7) has spent 5.82 over the two runs on it. The audit of the same
settings passes (`audit-dp-sgd`, 16 shadow models, ε 4.01): loss AUC 0.505, LiRA online 0.498,
offline 0.503, the largest advantage 0.031, TPR at 1% FPR 1.55% at most and at 0.1% 0.30%.

The shipped run (`dp-sgd`, noise from OS entropy), against the model it replaces, on the
`dp-sgd` data's held-out people in each format and on `akili statement-eval`'s 246 invented
statement rows (markers `EXAMPLE` and `MFANO` and words the generator never writes; people count
as right in any person category):

| | before | after |
| --- | --- | --- |
| held-out people, `payee_format: app` | 0.341 | **0.849** |
| held-out people, `payee_format: sms` (varied) | 0.625 | **0.867** |
| statement payees | 0.122 | **0.813** |

Per category on the statement payees (precision / recall):

| | before | after |
| --- | --- | --- |
| groceries | 0.20 / 0.03 | 1.00 / 0.60 |
| fuel | 0.11 / 0.08 | 0.69 / 1.00 |
| eating_out | 0.17 / 0.04 | 0.96 / 0.92 |
| snacks | 0.00 / 0.00 | 0.95 / 1.00 |
| transport | 1.00 / 0.11 | 1.00 / 0.17 |
| utilities | 0.00 / 0.00 | 1.00 / 0.75 |
| school_fees | 0.07 / 0.08 | 1.00 / 0.83 |
| health | 0.00 / 0.00 | 1.00 / 0.96 |
| rent | 0.50 / 0.25 | 1.00 / 0.88 |
| shopping | 0.00 / 0.00 | 0.90 / 1.00 |
| lifestyle | 0.19 / 0.39 | 0.84 / 0.89 |
| giving | 0.00 / 0.00 | 0.35 / 1.00 |
| savings | 1.00 / 0.50 | 1.00 / 0.75 |
| loan_repayment | 0.00 / 0.00 | 0.57 / 1.00 |
| salary | 1.00 / 1.00 | 1.00 / 0.50 |

And per category on the held-out people (`app` format), every one of the 27 rose: groceries
0.33 / 0.24 to 0.81 / 0.78, fuel 0.10 / 0.21 to 0.83 / 0.90, family support 0.09 / 0.15 to 0.77 /
0.63; the weakest after are the four kinds of money from people (money received 0.47 / 0.49,
gifts 0.67 / 0.54), which only the amount and time tell apart. Real payees will score lower; the
app's rules name well-known shops and stations (Naivas, Quickmart, Carrefour, Shell, Rubis,
TotalEnergies and others) before the model runs.

## Transport, salary and money from people

*2026-11-25, later.* After the statement-payee model, `akili statement-eval` still found
transport at a recall of 0.17 and salary at 0.50, and on held-out people the four kinds of money
from people were the weakest. What the rows looked like and what they were taken for (the
shipped model, on the set below and on held-out `dp-sgd` people):

- **Transport** (`EXAMPLE TAXI RIDES`, `POS TXN MFANO TUMAINI SHUTTLE`, `EXAMPLE MATATU SACCO`):
  taken for giving (20 of 54), fuel (14) and groceries (6). The generator named transport with
  four words only (`SACCO`, `SHUTTLE`, `RIDES`, `COURIERS`; `SACCO` is savings too), so a taxi,
  boda boda, coach or airline was a word it never saw, and every fare was a matatu fare (KES 150).
- **Salary** (`EXAMPLE EMPLOYER LTD`, `PAY BILL FROM 000272 - EXAMPLE COMPANY PAYROLL`): taken for
  business income. Salary was the one category the varied names left alone: three fixed names,
  never a statement's wording.
- **Money from people**: on the statement set every one landed in a received category; on held-out
  people the errors are between the four (gifts as money received 130, money received as support
  77 and as gifts 75), which only the amount and the time tell apart.

`payee_format: app` now also writes (made-up words only): transport as a taxi, boda boda, matatu
SACCO, bus service, coach or airline, and 15% of trips long ones (a coach, a train, a flight) at
10 to 40 times the fare; salary with invented employers (`... LTD SALARY`, `... LIMITED PAYROLL`,
`... LTD NET PAY`, `WAGES`) and statement credits (`SALARY ...`, `PAYROLL CREDIT ...`, `PAY BILL
FROM 000nnn - ...`); and money in from people as a statement words it (`CUSTOMER TRANSFER FROM
0700***0nn - ...`, `FUNDS RECEIVED FROM ...`, `RECEIVED FROM ...`), not `TRANSFER TO`. The `sms`
format's data is unchanged.

`akili statement-eval` has 302 rows (from 246): taxi, shuttle, matatu, boda, cab and bus rows,
long trips (railway, airline, coach tickets at KES 4,500), salary credits (`LTD SALARY`, `LIMITED
NET PAY`, `PAYROLL`, from a paybill) and eight transfers from people, still with the markers
`EXAMPLE` and `MFANO` and words the generator never writes. Money to people counts as right in
either sent category and money from people in any received one; `people` reports both.

**Privacy budget.** The ledger composes every run on the same dataset (`synthetic_dataset_id`,
a hash of the data settings), and the gate checks that total against the track's limit, so the
cap is cumulative per dataset, not per run. Seed 2026 had spent 5.82 of the app track's 8.0;
one more run of these settings on it would compose past 8.0. Each generator change is a new set
of invented people, so `dp-sgd` moved to a new seed with a budget of its own: 2027 and 2028 were
spent on trials of the forms above (24 epochs: 0.76 and 0.66 on the statement set, with
groceries and health down), and 2029, with 32 epochs (ε **4.66** at δ 1e-6, PRV 4.33; one run,
so the data's total is 4.66), is the shipped run. `tracks.yaml` and `accountants.v1.json` are
unchanged.

The full audit of the same settings and data forms passes (`audit-dp-sgd`, 16 shadow models,
32 epochs, ε 4.66): loss AUC 0.514, LiRA online 0.497, offline 0.517, the largest advantage
0.041, TPR at 1% FPR 1.50% at most and at 0.1% 0.00%. The no-DP baseline still fails (LiRA
offline AUC 0.660), as it should. The quick (CI-only) audit now trains at most 24 epochs: with its 40 people, 32
epochs of noise moved the int8 file 0.60 from torch, past even its wider parity limits.

Before (the model from the section above) and after, on the same rows: the new statement set,
and held-out people of the new generator with seed 2026 (people neither model was trained on)
and seed 2029 (`dp-sgd`'s own held-out people). Precision / recall:

| | statement set | held-out, seed 2026 | held-out, seed 2029 |
| --- | --- | --- | --- |
| accuracy | 0.722 → **0.818** | 0.825 → **0.862** | 0.832 → **0.863** |
| airtime_data |  | 0.96 / 0.98 → 0.98 / 0.99 | 0.95 / 0.99 → 0.97 / 0.99 |
| allowance_bursary |  | 0.82 / 0.96 → 0.94 / 0.97 | 0.83 / 0.96 → 0.96 / 0.97 |
| betting |  | 0.89 / 0.92 → 0.91 / 0.97 | 0.90 / 0.94 → 0.93 / 0.96 |
| business_income |  | 0.82 / 0.88 → 0.88 / 0.89 | 0.81 / 0.89 → 0.90 / 0.91 |
| eating_out | 0.88 / 0.92 → 0.71 / 0.83 | 0.87 / 0.91 → 0.83 / 0.88 | 0.85 / 0.90 → 0.82 / 0.92 |
| family_support |  | 0.72 / 0.62 → 0.77 / 0.55 | 0.76 / 0.62 → 0.74 / 0.52 |
| fees_charges |  | 0.93 / 0.96 → 0.97 / 0.96 | 0.94 / 0.98 → 0.97 / 0.98 |
| friends_social |  | 0.65 / 0.61 → 0.71 / 0.58 | 0.69 / 0.67 → 0.70 / 0.64 |
| fuel | 0.49 / 1.00 → 0.88 / 0.88 | 0.84 / 0.91 → 0.82 / 0.90 | 0.86 / 0.89 → 0.85 / 0.92 |
| fuliza_repayment |  | 0.96 / 0.98 → 0.98 / 0.98 | 0.94 / 0.97 → 0.95 / 0.97 |
| gifts_received |  | 0.63 / 0.49 → 0.63 / 0.57 | 0.58 / 0.47 → 0.58 / 0.56 |
| giving | 0.27 / 1.00 → 0.35 / 1.00 | 0.83 / 0.85 → 0.93 / 0.93 | 0.84 / 0.85 → 0.96 / 0.91 |
| groceries | 0.75 / 0.60 → 0.95 / 0.70 | 0.81 / 0.75 → 0.89 / 0.89 | 0.81 / 0.75 → 0.87 / 0.86 |
| health | 0.89 / 0.96 → 0.96 / 0.96 | 0.82 / 0.90 → 0.87 / 0.93 | 0.87 / 0.93 → 0.89 / 0.93 |
| lifestyle | 0.84 / 0.89 → 1.00 / 1.00 | 0.76 / 0.81 → 0.87 / 0.89 | 0.77 / 0.82 → 0.88 / 0.93 |
| loan_repayment | 0.57 / 1.00 → 0.78 / 0.88 | 0.90 / 0.91 → 0.92 / 0.94 | 0.90 / 0.90 → 0.93 / 0.92 |
| money_received |  | 0.43 / 0.49 → 0.44 / 0.38 | 0.47 / 0.49 → 0.48 / 0.40 |
| rent | 1.00 / 0.88 → 0.89 / 1.00 | 0.93 / 0.94 → 0.93 / 0.97 | 0.94 / 0.95 → 0.94 / 0.97 |
| salary | 1.00 / 0.65 → 0.95 / 1.00 | 0.94 / 0.87 → 0.95 / 0.97 | 0.95 / 0.83 → 0.95 / 0.98 |
| savings | 1.00 / 0.75 → 0.88 / 0.88 | 0.92 / 0.93 → 0.92 / 0.96 | 0.92 / 0.90 → 0.92 / 0.95 |
| school_fees | 1.00 / 0.83 → 1.00 / 0.75 | 0.87 / 0.92 → 0.91 / 0.91 | 0.93 / 0.94 → 0.94 / 0.92 |
| shopping | 0.90 / 1.00 → 1.00 / 0.94 | 0.84 / 0.82 → 0.89 / 0.93 | 0.85 / 0.84 → 0.89 / 0.92 |
| snacks | 0.95 / 1.00 → 1.00 / 1.00 | 0.88 / 0.88 → 0.88 / 0.94 | 0.85 / 0.87 → 0.89 / 0.90 |
| support_received |  | 0.59 / 0.55 → 0.61 / 0.72 | 0.65 / 0.64 → 0.65 / 0.74 |
| tax |  | 0.94 / 0.97 → 0.95 / 0.96 | 0.95 / 0.98 → 0.96 / 0.97 |
| transport | 1.00 / 0.07 → 0.94 / 0.56 | 0.81 / 0.58 → 0.89 / 0.77 | 0.79 / 0.61 → 0.81 / 0.78 |
| utilities | 1.00 / 0.75 → 1.00 / 0.75 | 0.84 / 0.92 → 0.89 / 0.95 | 0.85 / 0.94 → 0.94 / 0.95 |
| money to people (statement, 4 rows) | 0.75 → 0.50 | | |
| money from people (statement, 8 rows) | 1.00 → 0.75 | | |

The four kinds of money from people trade among themselves: support received and gifts rose,
money received and family support lost recall while their precision held or rose. The app now
narrows the model to the received categories for money from a person (pesaclarity-app), and
names transport operators and salary wording before the model runs.

## Parity

The export fails unless `categoriser.weights` is close to the torch model's float
probabilities on 2,060 rows: 2,000 transactions generated with the next data seed and the 60
feature vectors. The limits (`export.parity_*`) are, in the style of the readers':

- at most **0.25** apart on any row, and **0.08** at the 99.9th percentile;
- the same label wherever torch's two best probabilities are more than **0.2** apart.

Decision 0023 proposed 1e-2. Without DP the categoriser is near it: `smoke` within 0.013
(99.9% of rows within 0.010), `audit-no-dp` within 0.025. With DP it is not, for a reason of
the int8 scheme rather than of the conversion: DP-SGD adds noise to every row of the token
table at every step, so rows no person's tokens hashed to drift (with the Keras recipe in torch,
`dp-sgd` with noise seed 1: the largest value 4.2, against 3.1 among rows in use), and the table's one int8 scale spans that drift. A row whose tokens
land on those rows moves most, and the feature vectors' invented payees are such rows. The
Keras TFLite file has the same scheme and moved as far (0.062 on the same generated rows for
`dp-sgd` with noise seed 1). Measured:

| Config | Largest | 99.9th percentile | Labels compared / same |
| --- | --- | --- | --- |
| `smoke` | 0.013 | 0.010 | 1,945 / 1,945 |
| `audit-no-dp` | 0.025 | 0.022 | 2,005 / 2,005 |
| `audit-dp-sgd` | 0.057 | 0.045 | 1,789 / 1,789 |
| `dp-sgd`, OS entropy | 0.066 | 0.030 | 1,983 / 1,983 |
| `dp-sgd`, noise seeds 1 to 10 | 0.045 to 0.165 | 0.030 to 0.057 | all the same |

Keeping the table float32 would give 0.053 at the most on that model, but a 272 KB file;
the int8 file's accuracy is the trained model's to about 0.1 point, so the table
stays int8. rayo's C library and rayo-convert's NumPy reference give the same outputs to 1e-5
(`tests/test_categoriser_torch.py`).

## Federated

The federated simulation trains the same module ([`akili.federated.run`](../src/akili/federated/run.py)
with the torch backend). The clients still send NumPy arrays in the order and layout
pesaclarity-backend's aggregator holds: Keras' `get_weights()` order (`token_embedding`,
`context_0` .. `context_4`, `hidden` kernel and bias, `category` kernel and bias) with dense
kernels `[in, out]` (`akili.model.keras_layout`; torch's `[out, in]` is transposed at the
boundary). The backend needs no change.

## Versions

- **PyTorch** (decision 0023, stage 4): the same network, layer names, features, labels, data,
  configs and seeds; the recipe above; shipped as rayo's int8 `categoriser.weights` with the
  ONNX source; fine-tuning by the written step instead of `categoriser-train.tflite`.
- **Keras**: `categoriser.tflite` (dynamic-range int8) and `categoriser-train.tflite`; removed
  with TensorFlow in stage 5 (a DP-SGD run of both is in the `tflite-archive` release).
