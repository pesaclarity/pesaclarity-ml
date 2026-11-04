# DP-SGD review (issue #3)

Review of the custom DP-SGD code against the privacy accountant's assumptions, the track
thresholds and the state of maintained DP libraries. Reviewed on 2026-11-04 against TensorFlow
2.21.0, Keras 3.15.1, dp-accounting 0.6.0 and Python 3.12.13.

Some terms used below:

* **DP-SGD**: training where each example's gradient is limited in size ("clipped") and random
  noise is added, so no single example can change the model much.
* **Accountant**: the code that turns the noise, sampling rate and number of steps into a
  privacy budget (epsilon, delta).
* **Epsilon (ε)**: how much one record can shift what the model reveals. Smaller is more private.
* **Delta (δ)**: the small chance that the epsilon promise fails. It must be well below 1/N,
  where N is the number of records.
* **Poisson sampling**: each step, every example joins the batch independently with probability
  q. Batch sizes vary.

## Summary

The core algorithm is right. The code does what the accountant assumes:

* Each example's whole gradient is clipped to the norm C, across all layers together.
* Gaussian noise with standard deviation (noise multiplier × C) is added once per step to the
  sum of clipped gradients, then divided by the *expected* batch size.
* Batches use true Poisson sampling, and the accountant is told Poisson sampling.
* The step count and sampling rate given to the accountant match what training does.

I checked each of these by reading the code and by measuring it (see the tests added to
`tests/test_tf_training.py`).

I fixed two small things:

1. The release gate now refuses a delta that is not below 1/population. Before, it only checked
   the fixed track limit, which is too loose once a dataset has more than about a million
   records.
2. Training now reports the steps it actually took and the sampling rate it actually used, and
   the pipeline checks both against the accountant. Before, the step check compared one formula
   with a copy of itself.

The bigger problems are not in the maths. They are about what the budget covers. Fix these
before any real data is used:

* **The privacy unit is one transaction, not one person.** A user with 60 transactions gets far
  weaker protection than the published epsilon suggests.
* **The noise is reproducible from a seed that is published.** Anyone with the seed can
  regenerate the noise. That is fine for synthetic data, but it voids the guarantee on real data.
* **Some uses of the data are outside the budget**: the audit's shadow models, the logged training
  loss and validation accuracy, repeated runs, and any tuning on private data.

No maintained library is a drop-in replacement yet. tensorflow-privacy has not released since
February 2024 and does not support Python 3.12. jax-privacy 2.0.0 (April 2026) has a Keras 3 API,
but only with the JAX backend. Opacus 1.6.0 (May 2026) is mature, but it is for PyTorch.

## Checks

Status key: **Passed** (correct as is), **Fixed** (changed in this review), **Open** (needs a
decision or larger work).

### 1. Per-example clipping: Passed

* `src/akili/train/dp_sgd.py:59-65`: each example gets its own gradient tape on a batch of one
  (`xi[None]`). `tf.clip_by_global_norm` then clips the list of every trainable variable's
  gradient to one joint L2 norm C. This is per example, not per layer and not per batch.
* `dp_sgd.py:77`: `tf.vectorized_map` runs this over the batch. For the `L2Loss` op inside the
  clip it falls back to a `while_loop` (TensorFlow logs a warning). That is slower, but still
  correct.
* The model has no layers that mix examples (`src/akili/model.py:27-46`): embeddings, a masked
  mean inside each example, dense layers and dropout. There is no batch norm.
* Measured: one step with q = 1 and no noise matches a hand-computed sum of per-example clipped
  gradients to about 1e-6, for C = 0.05 and C = 1.0. Test:
  `test_dp_sgd_clips_each_example_over_all_variables`. Rewriting the code to clip per layer or
  per batch makes it fail (checked by temporarily breaking the code).

### 2. Noise: Passed

* `dp_sgd.py:53`: std = `l2_norm_clip × noise_multiplier`.
* `dp_sgd.py:78-82`: one fresh Gaussian draw per coordinate per step, added to the *sum* of
  clipped gradients, then divided by `expected_batch = q × n` (`dp_sgd.py:51`), not by the
  realised batch size. This is what Poisson accounting assumes.
* `dp_sgd.py:86-91, 101-103`: an empty Poisson batch still applies a noise-only step, as it
  should.
* SGD momentum (`dp_sgd.py:55`) only processes the noisy gradients further, so it costs no
  privacy.
* Measured: over 25 seeds, the noise std was 0.99986 ± 0.0005 of σ·C. Noise in different steps
  was uncorrelated (largest correlation 0.005). Test: `test_dp_sgd_noise_std_is_multiplier_times_clip`,
  which uses q = 0.75 and a seed where the realised batch differs from the expected one, so
  dividing by the realised size fails it. So do the wrong std (σ instead of σ·C) and fixed-size
  batches.

### 3. Sampling: Passed

* `dp_sgd.py:99`: `rng.random(n) < q` picks each example independently. This is Poisson
  sampling. It is not shuffled fixed-size batches.
* `src/akili/privacy/accounting.py:72-76`: `RdpAccountant` with `PoissonSampledDpEvent(q,
  GaussianDpEvent(σ))`. This matches the code. The known gap of accounting fixed batches as
  Poisson is not present here.
* The accountant counts adding or removing one example, and the code's divide-by-expected-batch
  matches that. If you ever want "replace one example" instead, the sensitivity doubles. For
  `configs/dp-sgd.yaml` that would raise ε from 1.83 to about 11.7.
* RDP gives an upper bound. For `configs/dp-sgd.yaml` (n = 9,600, q = 0.0133, 375 steps,
  σ = 1.1, δ = 1e-6), RDP with the repo's orders gives ε = 1.830, RDP with the library's default
  orders gives 1.814, and the PLD accountant gives 1.522. The repo's figure is conservative and
  valid. Switching to `dp_accounting.pld.PLDAccountant` would report a tighter, still correct ε.

### 4. Steps, epochs and delta: Passed, two things Fixed

* Steps: `dp_sgd.py:95` and `accounting.py:91` both use `epochs × max(1, round(n / batch))`, and
  q = `min(1, batch / n)` in both (`dp_sgd.py:50`, `accounting.py:90`). The pipeline passes the
  same `n = len(y_train)` to both (`src/akili/train/pipeline.py:109-118`).
* **Fixed:** `train_dp_sgd` returned the planned step count, not the counted one. The check in
  `pipeline.py` therefore compared a formula with a copy of itself. Training now returns
  `steps=len(sizes)` (counted in the loop) and the `sampling_rate` it used (`dp_sgd.py:105-110`).
  The pipeline checks both against the report (`pipeline.py:120-123`). Test:
  `test_dp_sgd_matches_what_the_accountant_is_told`.
* Delta: the configs use δ = 1e-6 with n = 9,600, which is well below 1/N.
* **Fixed:** the gate only checked δ against the fixed track limit (1e-6 app, 1e-7 commercial).
  With more than 1 million (app) or 10 million (commercial) records, a δ that passes is no longer
  below 1/N. `src/akili/privacy/gate.py:59-65` now fails any report with δ × population ≥ 1. Test:
  the new case in `test_each_unmet_condition_fails`. `default_delta()` already picks
  `min(1e-5, 1/(10N))`, so the default path was fine.
* `akili epsilon` estimates n as `int(n × (1 − validation_fraction))`, which can be one lower
  than the real split. That gives a slightly larger, so safe, ε. No change needed.

### 5. Leaks outside DP: mostly Open

| Item | Status | Evidence and notes |
| --- | --- | --- |
| Batch norm or other layers that mix examples | Passed | None in `model.py:27-46`. |
| Vocabulary or feature hashing built from data | Passed | FNV-1a hashing into a fixed number of buckets (`features.py`). Amount buckets and categories are fixed constants or YAML. Nothing is learned from the data. |
| Class weights | Passed | None. |
| Validation split and early stopping | Passed | `split_indices` holds the validation set out (`pipeline.py:50`). There is no early stopping or checkpoint selection. |
| Validation accuracy and training loss | Open (low) | `val.accuracy` (`pipeline.py:135`) is a non-private number about held-out records. `train.final_loss` (`pipeline.py:124`, from `dp_sgd.py:84`) is a non-private loss on the last sampled training batch. Both go to MLflow. On real data, keep them internal or compute them with DP. |
| Noise and sampling seed | **Open (high, before real data)** | `dp_sgd.py:93-94` seed both the noise and the batch choice from `cfg.seed`. That is 42 in every config, and it is logged to MLflow (`pipeline.py:107`). Anyone who knows the seed and the code can regenerate every noise draw, so the DP guarantee no longer holds. The `dp-sgd` and `audit-dp-sgd` configs even share the same noise. The federated client already notes that real clients must use OS entropy (`federated/client.py:70`). Suggested fix: keep the seed for model initialisation, but draw DP noise and sampling from OS entropy (`secrets` / `np.random.SeedSequence()`) unless the config explicitly asks for a reproducible synthetic run. Not done here, because it changes how reproducible audits and CI are. |
| Privacy unit | **Open (high, owner decision)** | The budget is per transaction (`PrivacyUnit.EXAMPLE`, `accounting.py:95`). Users have many transactions (60 in the synthetic configs). By group privacy, a user with k transactions is only protected at about k × ε. For ε = 1.83, that is about 37 for k = 20 and about 110 for k = 60, which is effectively nothing. The gate compares example-level and user-level ε against the same `max_epsilon` (`gate.py:51-54`). Recommendation: release only user-level budgets, either from DP-FedAvg or from DP-SGD with per-user clipping. Or make the gate refuse `unit: example` reports. |
| Audit shadow models | Open (medium) | `privacy/mia.py:122-133` trains 16 more models on the same records, validation records included, and publishes the attack metrics. None of this is in ε. That is fine on synthetic data. On real data, run the audit on a separate proxy dataset, or count it. |
| Repeated runs and tuning | Open (medium) | Each run on the same data spends budget again. There is no ledger, and the reported ε covers one run only. Tuning learning rate, clip or σ on private data also leaks (Papernot & Steinke 2022). Tune on synthetic or public data only, and keep a per-dataset log of every DP run and its ε. |
| Gate requirements for centralised models | Open (note) | The gate always requires secure aggregation and a minimum cohort (`gate.py:67-72`). A centralised DP-SGD model can only pass if its evidence claims these, which would not be true. Decide whether centralised models may be released at all. If not, say so in the gate. |

### 6. Numerical issues: Passed, one note Open

* Microbatching: none. The code computes a true per-example gradient for every example, with
  no microbatches of more than one, so the sensitivity is exactly C.
* Gradient accumulation: none. Noise is added once per optimiser step.
* Float32 noise: TensorFlow draws noise as float32 from a fast pseudo-random generator (Philox).
  It is not a cryptographically secure generator. Floating-point and timing attacks on
  DP noise have been shown (Mironov 2012; Jin et al. 2022). The practical risk is low here, but
  Opacus offers a `secure_mode` for this reason. Open, low. Revisit together with the seed fix.

## Maintained DP-SGD libraries

Checked on 2026-11-04 from PyPI metadata, plus the jax-privacy 2.0.0 wheel's source. I could not
reach GitHub's API from this session, so I could not see unreleased work on tensorflow-privacy's
main branch.

| Library | Latest release | Python | Keras 3? | Notes |
| --- | --- | --- | --- | --- |
| [tensorflow-privacy](https://pypi.org/project/tensorflow-privacy/) | 0.9.0, 2024-02-14 | `>=3.9, <3.12` | No | No release in about 2.7 years. It pins `dp-accounting==0.4.3`, `numpy~=1.21` and `tensorflow-probability~=0.22`. The repo pinned TensorFlow ≤ 2.15 to avoid Keras 3 ([search index of tensorflow/privacy issues](https://issues.ecosyste.ms/hosts/GitHub/repositories/tensorflow%2Fprivacy/issues), PR #558). Not usable on this stack. |
| [jax-privacy](https://pypi.org/project/jax-privacy/) | 2.0.0, 2026-04-17 (1.0.0 was 2025-07-14) | `>=3.11` | Yes, JAX backend only | `jax_privacy.keras_api.make_private` wraps a Keras 3 model ([docs](https://jax-privacy.readthedocs.io/en/stable/overview.html), [source](https://jax-privacy.readthedocs.io/en/latest/_modules/jax_privacy/keras_api.html)). It raises an error unless `keras.config.backend() == "jax"`. Uses dp-accounting ≥ 0.6.0. **Caution:** `poisson_sampling_in_fit` defaults to `False`, which means ordinary `fit()` batches. Set it to `True`, or you hit the fixed-batch gap again. It needs JAX, jaxlib and optax, and TFLite export from the JAX backend would need checking. |
| [Opacus](https://pypi.org/project/opacus/) | 1.6.0, 2026-05-05 | 3.12 works (requires torch ≥ 2.6) | n/a (PyTorch) | Mature and maintained ([opacus.ai](https://opacus.ai)). Poisson sampling, `secure_mode`, and RDP/PRV accountants. Switching would mean moving the training stack to PyTorch and the export to LiteRT/TFLite via PyTorch conversion. |
| [dp-accounting](https://pypi.org/project/dp-accounting/) | 0.6.0, 2026-02-09 | `>=3.10` | n/a | Already in use. Has RDP and PLD accountants. |
| keras-dp, private-keras | none | | | Neither is on PyPI. |

Recommendation: keep the custom loop for now. It is small, correct, and now covered by tests
that check its key properties. Revisit jax-privacy if the project moves Keras to the JAX
backend. Then compare its clip and noise output with the tests here, and turn on
`poisson_sampling_in_fit=True`. Keep checking tensorflow-privacy releases. If a release supports
Keras 3 and Python 3.12, it would be the easiest switch.

## Recommended thresholds

These are recommendations for the owner. I have not changed `src/akili/resources/tracks.yaml`.

Numbers for the audit limits come from a simulation of a perfectly private model (400 trials,
with "worst of three attacks" approximated by three independent draws):

| Per side | AUC p95 (worst of 3) | Advantage p95 (worst of 3) | TPR@0.1% p95 (worst of 3) | TPR@1% p95 (worst of 3) |
| --- | --- | --- | --- | --- |
| 2,000 | 0.519 | 0.049 | 0.0050 | 0.018 |
| 5,000 | 0.513 | 0.030 | 0.0030 | 0.015 |

Reference points from `docs/PRIVACY-AUDIT.md`: the no-DP baseline reached AUC 0.575, advantage
0.122 and TPR@0.1% 2.15%. Under the current app limits, its AUC (0.575) and TPR@1% (3.75%) would
both pass. Only the advantage and TPR@0.1% limits catch it.

| Setting | App now | App recommended | Commercial now | Commercial recommended | Why |
| --- | --- | --- | --- | --- | --- |
| Privacy unit | example | **user** | example or user | **user** | The person is what we promise to protect. Example-level ε says little about a user with many transactions (see check 5). |
| max ε | 8 | 8 hard limit, aim for ≤ 3 | 3 | 3 hard limit, aim for ≤ 1 | 8 at user level is common in production federated systems, but it gives almost no bound on attacks: TPR ≤ e^ε × FPR + δ is vacuous at ε = 8. Licensed models deserve a meaningful bound: at ε = 1, TPR@1% ≤ 2.7%. |
| max δ | 1e-6 | 1e-6 **and** < 1/N (now enforced); aim for ≤ 1/(10N) | 1e-7 | 1e-7 **and** < 1/N; aim for ≤ 1/(10N) | δ ≥ 1/N allows mechanisms that leak a whole record. `default_delta()` already uses 1/(10N). |
| max attack AUC | 0.60 | **0.55** | 0.55 | **0.53** | 0.60 does not separate the no-DP baseline (0.575) from a private model. Both new values are well above the null p95 (0.519 at 2,000 per side). |
| max attack advantage | 0.10 | 0.10 | 0.05 | 0.05, with 5,000 per side | At 2,000 per side, 0.05 is at the null p95 for the worst of three attacks, so a clean model fails about 1 time in 20. At 5,000 per side the null p95 drops to 0.030. |
| max TPR at 1% FPR | 0.05 | **0.03** | 0.03 | **0.02** | The no-DP baseline reached 0.0375, which passes 0.05. The null p95 is 0.015 to 0.018. |
| max TPR at 0.1% FPR | 0.01 | 0.01 | 0.005 | 0.005, with 5,000 per side | At 2,000 per side, 0.005 equals the null p95 for the worst of three attacks. At 5,000 per side the null p95 is 0.003. |
| min audit examples (each side) | 1,000 | **2,000** | 2,000 | **5,000** | More examples shrink chance variation, roughly with the square root of n. At 1,000 non-members, TPR@0.1% rests on a single false positive. |
| `provisional` | true | false once signed off | true | false once signed off | Flip only when the owner accepts the numbers in #3. |

Two conditions apply to all of these:

* The thresholds only mean what they say once the seed issue and the audit-data issue in
  check 5 are fixed.
* Report ε from the PLD accountant alongside RDP. PLD is tighter, and for the current config it
  gives 1.52 rather than 1.83.

## Files changed in this review

* `src/akili/train/dp_sgd.py`: `DPTrainingStats` gains `sampling_rate`, and `steps` is now
  counted in the loop.
* `src/akili/train/pipeline.py`: `FitOutput.dp_sampling_rate`, plus a check that it matches the
  accountant.
* `src/akili/privacy/gate.py`: refuse δ ≥ 1/population.
* `tests/test_tf_training.py`: tests for clipping, noise scale and division by the expected
  batch, and a test that training matches the accountant.
* `tests/test_audit_and_gate.py`: a gate case for δ ≥ 1/population.
