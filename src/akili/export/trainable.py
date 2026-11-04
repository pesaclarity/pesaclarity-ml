"""The trainable TFLite export for fine-tuning on the phone (needs the [train] extra).

The app fine-tunes the categoriser on the person's own confirmations without anything leaving the
phone. TensorFlow's automatic gradients need ops outside the TFLite builtin set, which would mean
shipping the Flex delegate (several MB). So the token and context embeddings stay frozen, and the
two dense layers on top (``hidden`` and ``category``) train with hand-written gradients that use
builtin ops only.

Signatures:

* ``train(x, y, lr) -> loss``: one SGD step on a batch; ``y`` holds label indices.
* ``infer(x) -> output``: class probabilities, the same as the Keras model.
* ``save() -> w1, b1, w2, b2``: the dense weights, for the app to keep in encrypted storage.
* ``restore(w1, b1, w2, b2) -> ok``: loads weights saved earlier.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.config import ExportConfig
from akili.export.size import ExportedModel, check_size

TRAINABLE_NAME = "categoriser-train.tflite"
SIGNATURES = ("train", "infer", "save", "restore")
WEIGHTS = ("w1", "b1", "w2", "b2")


def _trainable_module(model: Any) -> Any:
    tf = require("tensorflow", "train")
    keras = require("keras", "train")
    hidden = model.get_layer("hidden")
    category = model.get_layer("category")
    body = keras.Model(model.input, hidden.input)
    width = int(model.input.shape[1])
    n_classes = int(category.units)

    class Trainable(tf.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.body = body
            # Keras 3 variables are not tracked by tf.Module; track the tf.Variables beneath.
            self.frozen = [v.value for v in body.variables]
            self.w1 = tf.Variable(hidden.kernel.numpy(), name="w1")
            self.b1 = tf.Variable(hidden.bias.numpy(), name="b1")
            self.w2 = tf.Variable(category.kernel.numpy(), name="w2")
            self.b2 = tf.Variable(category.bias.numpy(), name="b2")

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            f = self.body(x, training=False)
            h = tf.nn.relu(tf.matmul(f, self.w1) + self.b1)
            return f, h, tf.nn.softmax(tf.matmul(h, self.w2) + self.b2)

        @tf.function(  # type: ignore[untyped-decorator]
            input_signature=[
                tf.TensorSpec([None, width], tf.int32, name="x"),
                tf.TensorSpec([None], tf.int32, name="y"),
                tf.TensorSpec([], tf.float32, name="lr"),
            ]
        )
        def train(self, x: Any, y: Any, lr: Any) -> dict[str, Any]:
            f, h, p = self.forward(x)
            onehot = tf.one_hot(y, n_classes)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            loss = -tf.reduce_sum(onehot * tf.math.log(p + 1e-7)) / n
            # Softmax with cross-entropy: d loss / d logits = p - onehot.
            dz2 = (p - onehot) / n
            dw2 = tf.matmul(h, dz2, transpose_a=True)
            db2 = tf.reduce_sum(dz2, axis=0)
            dh = tf.matmul(dz2, self.w2, transpose_b=True) * tf.cast(h > 0, tf.float32)
            dw1 = tf.matmul(f, dh, transpose_a=True)
            db1 = tf.reduce_sum(dh, axis=0)
            self.w2.assign_sub(lr * dw2)
            self.b2.assign_sub(lr * db2)
            self.w1.assign_sub(lr * dw1)
            self.b1.assign_sub(lr * db1)
            return {"loss": loss}

        @tf.function(input_signature=[tf.TensorSpec([None, width], tf.int32, name="x")])  # type: ignore[untyped-decorator]
        def infer(self, x: Any) -> dict[str, Any]:
            return {"output": self.forward(x)[2]}

        @tf.function(input_signature=[])  # type: ignore[untyped-decorator]
        def save(self) -> dict[str, Any]:
            return {name: getattr(self, name).read_value() for name in WEIGHTS}

        @tf.function(  # type: ignore[untyped-decorator]
            input_signature=[
                tf.TensorSpec(hidden.kernel.shape, tf.float32, name="w1"),
                tf.TensorSpec(hidden.bias.shape, tf.float32, name="b1"),
                tf.TensorSpec(category.kernel.shape, tf.float32, name="w2"),
                tf.TensorSpec(category.bias.shape, tf.float32, name="b2"),
            ]
        )
        def restore(self, w1: Any, b1: Any, w2: Any, b2: Any) -> dict[str, Any]:
            self.w1.assign(w1)
            self.b1.assign(b1)
            self.w2.assign(w2)
            self.b2.assign(b2)
            return {"ok": tf.constant(1, tf.int32)}

    return Trainable()


def weight_shapes(model: Any) -> dict[str, list[int]]:
    """The shape of each trainable weight, for the manifest."""
    hidden = model.get_layer("hidden")
    category = model.get_layer("category")
    return {
        "w1": [int(d) for d in hidden.kernel.shape],
        "b1": [int(d) for d in hidden.bias.shape],
        "w2": [int(d) for d in category.kernel.shape],
        "b2": [int(d) for d in category.bias.shape],
    }


def to_trainable_tflite_bytes(model: Any, workdir: str | Path) -> bytes:
    """Convert to a TFLite flatbuffer with the four signatures, builtin ops only."""
    tf = require("tensorflow", "train")
    module = _trainable_module(model)
    saved = Path(workdir) / "trainable-savedmodel"
    signatures = {name: getattr(module, name).get_concrete_function() for name in SIGNATURES}
    tf.saved_model.save(module, str(saved), signatures=signatures)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_resource_variables = True
    return bytes(converter.convert())


def export_trainable(model: Any, path: str | Path, cfg: ExportConfig) -> ExportedModel:
    """Write the trainable model to ``path`` and enforce the size budget."""
    import tempfile

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out.write_bytes(to_trainable_tflite_bytes(model, tmp))
    return check_size(out, cfg.max_size_bytes)


class TrainableRunner:
    """Drives the exported signatures on CPU, as the app does, for tests and parity checks."""

    def __init__(self, path: str | Path) -> None:
        tf = require("tensorflow", "train")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._interpreter = tf.lite.Interpreter(model_path=str(path))
        self.signatures = self._interpreter.get_signature_list()

    def _run(self, name: str, **inputs: Any) -> dict[str, Any]:
        return dict(self._interpreter.get_signature_runner(name)(**inputs))

    def train(
        self, x: npt.NDArray[np.int32], y: npt.NDArray[np.int32], learning_rate: float
    ) -> float:
        out = self._run(
            "train", x=x.astype(np.int32), y=y.astype(np.int32), lr=np.float32(learning_rate)
        )
        return float(out["loss"])

    def infer(self, x: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        return np.asarray(self._run("infer", x=x.astype(np.int32))["output"], dtype=np.float32)

    def save(self) -> dict[str, npt.NDArray[np.float32]]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in self._run("save").items()}

    def restore(self, weights: dict[str, npt.NDArray[np.float32]]) -> None:
        self._run("restore", **{k: weights[k].astype(np.float32) for k in WEIGHTS})
