# Model weights

The code in this repository is licensed under AGPL-3.0 (see [LICENSE](LICENSE)). Trained model
weights are not.

- Weights trained by Pesa Clarity, including federated models built from users who opted in, will
  be released under a separate model licence. That licence has not been chosen yet.
- No weights are committed to this repository. `.gitignore` excludes model artefacts
  (`artefacts/`, `mlruns/`, `*.tflite`, `*.keras`, checkpoints and saved models).
- Models you train yourself with this code, on your own data, are yours. The AGPL still applies to
  the code you run and to any changes you make to it.

Why a separate licence: the spec keeps licensing federated models possible as the project's only
income, without ever licensing anything that describes individual people. See "Federated learning
and model licensing" in the [product spec](https://github.com/pesaclarity/docs).

Any weights release must first pass the [release checklist](docs/RELEASE-CHECKLIST.md).
