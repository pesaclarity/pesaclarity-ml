# Fine-tuning the categoriser on the phone, version 1

The app fine-tunes the categoriser on the person's own confirmed categories, on the phone; no
transaction, label or weight leaves it. This page is the spec of that step. The code that
defines it is [`akili.export.fine_tune`](../src/akili/export/fine_tune.py), and
[`fine_tune_vectors.v1.json`](../src/akili/resources/fine_tune_vectors.v1.json) holds the
conformance vectors an implementation must pass (`akili fine-tune-vectors` regenerates them;
torch's automatic gradients compute the reference).

Until decision 0023 the Keras export shipped a second TFLite file for this,
`categoriser-train.tflite`, with `train`, `infer`, `save` and `restore` signatures and
hand-written gradients (removed with TensorFlow in stage 5; at the `tflite-archive` tag).
The PyTorch categoriser ships no second file: the step below is small enough to write down,
and the app implements it itself (in Dart, or in rayo once it has a backward pass).

## What trains

Only the two dense layers at the top train; everything before them stays as shipped.

| Tensor | Shape (rayo's layout) | Start value |
| --- | --- | --- |
| `hidden.kernel` (W1) | `[hidden, in]`, `[32, 36]` by default | the shipped file's, dequantised (int8 x scale per output) |
| `hidden.bias` (b1) | `[hidden]` | the shipped file's |
| `category.kernel` (W2) | `[classes, hidden]` | the shipped file's (float32) |
| `category.bias` (b2) | `[classes]` | the shipped file's |

`f`, what `hidden` reads, is computed from `categoriser.weights` exactly as rayo runs it: the
mean of the token embeddings over the tokens that are not 0, then the five context
embeddings (the `div` and `context_*` nodes of its graph, concatenated). Tuned weights are kept
as float32 in the app's encrypted storage, tied to the SHA-256 of the weights file they were
learned on, and replace the four tensors when the model runs. A new weights file starts from
its own values again.

The manifest's `fine_tune` block says which model this applies to:

```json
"fine_tune": {
  "spec_version": 1,
  "trains": ["hidden.kernel", "hidden.bias", "category.kernel", "category.bias"],
  "input": "hidden",
  "weights": {"hidden.kernel": [32, 36], "hidden.bias": [32],
              "category.kernel": [27, 32], "category.bias": [27]},
  "settings": {"learning_rate": 0.05, "epochs": 3, "batch_size": 16,
               "max_examples": 500, "min_examples": 10, "min_hours_between_runs": 24}
}
```

The app refuses a `spec_version` it does not implement, and shapes that are not the weights
file's. `settings` are the old `trainable.fine_tune` values, unchanged
(`export.fine_tune` in the config).

## One step

On a batch of `n` rows `f` (`[n, in]`), their labels `y` (indices into `labels.json`) and the
learning rate `lr`, in float32:

```text
z1 = f W1^T + b1          h = relu(z1)
z2 = h W2^T + b2          p = softmax(z2)            (per row; subtract the row's max first)
loss = mean over the rows of -log p[y]               (reported; computed before the update)

g2  = (p - onehot(y)) / n
dW2 = g2^T h              db2 = sum of g2's rows
gh  = (g2 W2) * (z1 > 0)
dW1 = gh^T f              db1 = sum of gh's rows

W1 -= lr dW1   b1 -= lr db1   W2 -= lr dW2   b2 -= lr db2
```

Every gradient is taken from the weights before the step. There is no dropout, momentum or
weight decay on the phone. A hidden unit with `z1 <= 0` on every row of the batch gets no
gradient, so its kernel row and bias do not move (the `dead_unit` vector).

A run is `settings.epochs` passes over the most recent `settings.max_examples` confirmations,
in batches of `settings.batch_size`, one step per batch; the app chooses the order and waits
until it has `settings.min_examples` and `settings.min_hours_between_runs` have passed.

## Vectors

Each case of `fine_tune_vectors.v1.json` has `sizes` (in, hidden, classes), the start
`weights`, and `steps`: each with its rows `f`, labels `y`, `lr`, the `loss` before the update
and every tensor `after` it. Steps run in order, each from the weights the last one left. An
implementation passes when every loss and weight is within `tolerance` (1e-5) of the file.

| Case | What it checks |
| --- | --- |
| `one_row` | a batch of one |
| `batch` | five rows, two with the same label |
| `dead_unit` | a hidden unit off for every row keeps its weights |
| `three_steps` | three batches in a row |

Small sizes keep the file readable; the step does not depend on them.
