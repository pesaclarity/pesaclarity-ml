"""TFLite export (needs the [train] extra)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from akili._optional import require
from akili.config import ExportConfig
from akili.export.size import ExportedModel, check_size


def to_tflite_bytes(model: Any, quantize: bool) -> bytes:
    """Convert a Keras model to a TFLite flatbuffer, optionally with dynamic-range quantisation."""
    tf = require("tensorflow", "train")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return bytes(converter.convert())


def export_tflite(model: Any, path: str | Path, cfg: ExportConfig) -> ExportedModel:
    """Write the TFLite model to ``path`` and enforce the size budget."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(to_tflite_bytes(model, cfg.quantize))
    return check_size(out, cfg.max_size_bytes)


def run_tflite(path: str | Path, features: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
    """Run the exported model on CPU, for parity checks against the Keras model."""
    tf = require("tensorflow", "train")
    with warnings.catch_warnings():
        # tf.lite.Interpreter is deprecated in favour of ai-edge-litert; fine for a CPU check.
        warnings.simplefilter("ignore", UserWarning)
        interpreter = tf.lite.Interpreter(model_path=str(path))
    inp = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(inp["index"], list(features.shape))
    interpreter.allocate_tensors()
    interpreter.set_tensor(inp["index"], features.astype(np.int32))
    interpreter.invoke()
    out = interpreter.get_output_details()[0]
    return np.asarray(interpreter.get_tensor(out["index"]), dtype=np.float32)
