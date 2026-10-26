# Model release checklist

A model may leave the server, to phones or to a licensee, only when every item below is ticked.
The items mirror "Release conditions" in the [product spec](https://github.com/pesaclarity/docs).
The automated items are checked by `akili gate --evidence <file>` against the track policy in
[`src/akili/resources/tracks.yaml`](../src/akili/resources/tracks.yaml); the gate exits non-zero
on any failure. The manual items are recorded in the evidence file by the person who did them.

Copy this list into the release issue and fill it in.

## Release

- Model name and version:
- Track: app / commercial
- Category set version (`labels.json`):
- Feature schema version (`model-manifest.json`):
- MLflow run id:
- TFLite SHA-256:

## Privacy (checked by the gate)

- [ ] Trained with DP-SGD (or DP-FedAvg); `privacy-report.json` attached.
- [ ] Privacy budget published: epsilon = ___, delta = ___, unit (example or user), accountant,
      noise multiplier, clip norm, sampling rate and steps.
- [ ] Epsilon and delta are within the track limits.
- [ ] Federated updates were combined with secure aggregation.
- [ ] Every round met the minimum cohort size for the track (app 100, commercial 1,000).
- [ ] Membership-inference audit (`akili audit`, see [PRIVACY-AUDIT.md](PRIVACY-AUDIT.md)) run
      on the release model with at least the track's `min_audit_examples`; for every attack
      (loss, LiRA online and offline) the AUC, advantage and TPR at 1% and 0.1% FPR are within
      the track limits. `audit-report.json` attached. The audit limits are provisional until
      [#3](https://github.com/pesaclarity/pesaclarity-ml/issues/3) is closed.

## Compliance (recorded in the evidence file)

- [ ] Data protection impact assessment for federated training filed with the ODPC.
- [ ] Commercial track only: legal review by a Kenyan data protection lawyer before the first
      licence.
- [ ] Commercial track only: trained only on users who opted in to the commercial track, and
      withdrawals received before training were honoured.
- [ ] The model is not a credit or risk scoring model (never offered).

## Shipping

- [ ] TFLite file within the size budget for 2 GB RAM phones (gate checks
      `model_size_bytes` against `max_model_size_bytes`).
- [ ] TFLite output matches the Keras model on a synthetic sample.
- [ ] Model file signed (Ed25519) with the release key held outside this repository; the app
      verifies the signature before use.
- [ ] Weights published under the model licence (see [MODEL-WEIGHTS.md](../MODEL-WEIGHTS.md)),
      never committed here.

## Evidence file

`akili gate` reads a JSON file shaped like `akili.privacy.gate.ReleaseEvidence`. `audit` is one
attack's result or a list of them; `akili gate --evidence <file> --audit audit-report.json`
fills it from the audit report instead:

```json
{
  "model_name": "categoriser-2026.11",
  "track": "app",
  "privacy": { "...": "contents of privacy-report.json" },
  "audit": [
    { "attack": "loss", "auc": 0.51, "advantage": 0.02, "tpr_at_1pct_fpr": 0.011,
      "tpr_at_0_1pct_fpr": 0.001, "n_members": 2000, "n_non_members": 2000 },
    { "attack": "lira-online", "auc": 0.52, "advantage": 0.03, "tpr_at_1pct_fpr": 0.012,
      "tpr_at_0_1pct_fpr": 0.0015, "n_members": 2000, "n_non_members": 2000 }
  ],
  "cohort_size": 2400,
  "secure_aggregation": true,
  "dpia_filed": true,
  "legal_review": false,
  "model_size_bytes": 81720,
  "max_model_size_bytes": 512000
}
```
