# Model weights

The code in this repository is licensed under AGPL-3.0 (see [LICENSE](LICENSE)). Trained model
weights are licensed separately, by track.

## App-track weights: CC BY-SA 4.0

The categoriser and message reader weights that ship inside the Pesa Clarity app, including
federated models built from people who left "Help improve the app" on, are licensed under the
[Creative Commons Attribution-ShareAlike 4.0 International licence](LICENSE-WEIGHTS) (CC BY-SA
4.0).

- You may copy, share and adapt them, for any purpose, including commercially.
- Give credit: "Pesa Clarity categoriser (or message reader), by the Pesa Clarity project,
  CC BY-SA 4.0", with a link to this repository and a note of any changes.
- Share adaptations, such as fine-tuned weights, under CC BY-SA 4.0 too.

The licence comes with no warranty. The weights only suggest categories and fields; the person
confirms every suggestion. We ask, as a request rather than a condition of the licence, that no
one uses them for credit or risk scoring, or to make decisions about individuals.

A free licence keeps the app's weights compatible with F-Droid and lets anyone check what ships
on their phone.

## Commercial-track weights: licensed only by agreement

Models trained only from people who opted in to the separate commercial track are never published.
They are licensed to organisations under a written agreement on the terms in
[COMMERCIAL-MODEL-TERMS.md](COMMERCIAL-MODEL-TERMS.md). A Kenyan data protection lawyer reviews the
agreement before the first licence, as the spec's release conditions require.

## For both tracks

- No weights are committed to this repository. `.gitignore` excludes model artefacts
  (`artefacts/`, `mlruns/`, `*.tflite`, `*.keras`, checkpoints and saved models).
- Every release, on either track, must first pass the
  [release checklist](docs/RELEASE-CHECKLIST.md): a user-level privacy budget within the track
  limits, secure aggregation, the minimum cohort and a passing membership-inference audit. Its
  `privacy-report.json` and `audit-report.json` are published with the weights.
- Models you train yourself with this code, on your own data, are yours. The AGPL still applies to
  the code you run and to any changes you make to it.

Why two licences: the spec keeps licensing federated models possible as the project's only
income, without ever licensing anything that describes individual people. See "Federated learning
and model licensing" in the [product spec](https://github.com/pesaclarity/docs).
