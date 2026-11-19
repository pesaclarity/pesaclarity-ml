"""A PyTorch model to rayo's plain weights: ``torch.onnx`` at a pinned opset, then rayo-convert.

The path every model takes once it moves to PyTorch (pesaclarity/docs, decision 0023):

1. :func:`to_onnx` exports a ``torch.nn.Module`` with example inputs through PyTorch's
   TorchScript-based ONNX exporter (``dynamo=False``) at opset :data:`OPSET` (17; rayo-convert
   reads 13 or later), with the inputs and outputs named, constants folded and every
   initializer named after its layer: the parameters under their module names
   (``conv_1.weight``), and a linear layer's kernel the exporter folds transposed as
   ``<layer>.kernel`` (``fields.kernel``, Keras' ``[in, out]``) instead of the exporter's own
   ``onnx::MatMul_52``. The same module, inputs and torch version give the same bytes.
2. rayo-convert reads the ONNX graph (rayo's docs/CONVERTING.md), maps its ops to rayo's and,
   with ``weights="int8"``, quantises the weights itself: symmetric int8 kernels (one scale per
   output) and embedding tables (one scale), the activations quantised on the fly by rayo's
   kernels, as the TFLite dynamic-range files the app first ran were.
3. :func:`weights_of_module` gives :class:`akili.export.weights.Weights`, the same PCWT file as
   the old TFLite path gave: tensors named after the module's layers (``conv_1.kernel``,
   ``slot_hidden.bias``, ``token_embedding.table``, ``ln.gamma``), so a torch module whose
   layers keep the Keras names keeps the file's layout and the names the app reads.

:func:`parity` runs the torch model and the converted file side by side; :func:`deterministic`
fixes the seeds, algorithms and threads. Needs the ``[torch]`` extra (PyTorch on CPU and
``onnx``); no TensorFlow.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from rayo_convert import Options, Result, convert

from akili._optional import require
from akili.export.rayo import weights_from_result
from akili.export.weights import Weights, read_weights

if TYPE_CHECKING:
    import torch

OPSET = 17  # decision 0023: pinned; rayo-convert needs 13 or later

# rayo's weight roles for torch's parameter names, by module type.
_ROLES: dict[str, dict[str, str]] = {
    "Linear": {"weight": "kernel", "bias": "bias"},
    "Conv1d": {"weight": "kernel", "bias": "bias"},
    "Embedding": {"weight": "table"},
    "LayerNorm": {"weight": "gamma", "bias": "beta"},
}


def _torch() -> Any:
    return require("torch", "torch")


def deterministic(seed: int = 0, threads: int = 1) -> None:
    """Fixed seeds, deterministic algorithms and a fixed thread count, so the same code, torch
    version and machine give the same model and the same exported bytes."""
    torch = _torch()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)


def to_onnx(
    module: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    outputs: Sequence[str],
    opset: int = OPSET,
) -> bytes:
    """``module`` as an ONNX model: its inputs named and in the order of ``inputs`` (each an
    example tensor, batch of one), its outputs named ``outputs`` in the order ``forward``
    returns them. Exported in eval mode, without gradients."""
    torch = _torch()
    require("onnx", "torch")
    was_training = module.training
    module.eval()
    buf = io.BytesIO()
    try:
        with torch.no_grad(), warnings.catch_warnings():
            # The TorchScript-based exporter is marked deprecated in favour of the dynamo one,
            # which needs onnxscript and opset 18; this one keeps opset 17 and the parameters'
            # names. Moving is a change of this function alone (rayo-convert reads both).
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            torch.onnx.export(
                module,
                tuple(inputs.values()),
                buf,
                input_names=list(inputs),
                output_names=list(outputs),
                opset_version=opset,
                dynamo=False,
                do_constant_folding=True,
                export_params=True,
                keep_initializers_as_inputs=False,
            )
    finally:
        module.train(was_training)
    return _keep_layer_names(buf.getvalue(), module)


def _keep_layer_names(blob: bytes, module: torch.nn.Module) -> bytes:
    """``blob`` with every initializer named after its layer.

    The exporter folds a linear layer on a 3-D input into a ``MatMul`` by its kernel
    transposed, ``[in, out]``, under a name of its own (``onnx::MatMul_52``). That is Keras'
    kernel layout, so the initializer is renamed ``<layer>.kernel`` (``fields.kernel``), the
    layer taken from the node's scope (``/fields/MatMul``) and the values checked against the
    layer's weight. rayo-convert reads the same layer from that name as it did from the
    scope, so the converted tensors and graph are unchanged. (Parameters the exporter shares
    because their values are equal keep their names through ``Identity`` nodes.) A file with
    no such initializer comes back byte for byte."""
    import onnx
    from onnx import numpy_helper

    model = onnx.load_from_string(blob)
    graph = model.graph
    inits = {t.name: t for t in graph.initializer}
    params = dict(module.named_parameters())
    renames: dict[str, str] = {}
    for node in graph.node:
        layer = _scope(node.name)
        if node.op_type != "MatMul" or layer is None or len(node.input) < 2:
            continue
        name = node.input[1]
        weight = params.get(f"{layer}.weight")
        if name not in inits or "::" not in name or weight is None or weight.ndim != 2:
            continue
        kernel = f"{layer}.kernel"
        value = numpy_helper.to_array(inits[name])
        if kernel not in inits and np.array_equal(weight.detach().cpu().numpy().T, value):
            renames[name] = kernel
    if not renames:
        return blob
    for t in graph.initializer:
        t.name = renames.get(t.name, t.name)
    for node in graph.node:
        for k, name in enumerate(node.input):
            node.input[k] = renames.get(name, name)
    return bytes(model.SerializeToString())


def _scope(node_name: str) -> str | None:
    """The module path the exporter named a node by (``/fields/MatMul``: ``fields``;
    ``/attention_0/query/MatMul``: ``attention_0.query``)."""
    if not node_name.startswith("/"):
        return None
    parts = [p for p in node_name.split("/")[1:-1] if p]
    return ".".join(parts) or None


def expected_tensors(module: torch.nn.Module) -> list[str]:
    """The tensors rayo-convert must give for ``module``: each layer's under its name and
    rayo's role (``conv_1.kernel``, ``conv_1.bias``, ``token_embedding.table``, ``ln.gamma``).
    Layers of other types are not listed."""
    names = []
    for path, sub in module.named_modules():
        roles = _ROLES.get(type(sub).__name__)
        if not path or roles is None:
            continue
        for param, role in roles.items():
            if getattr(sub, param, None) is not None:
                names.append(f"{path}.{role}")
    return names


@dataclass(frozen=True, slots=True)
class Export:
    """An exported model: the ONNX file, rayo-convert's result, and the plain weights."""

    onnx: bytes
    result: Result
    weights: Weights

    @property
    def data(self) -> bytes:
        """The ``.weights`` file."""
        return self.weights.to_bytes()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def weights_of_module(
    module: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    outputs: Sequence[str],
    options: Options,
    *,
    layers: list[dict[str, Any]],
    required: list[str] | None = None,
    onnx_name: str | None = None,
) -> Export:
    """``module`` exported to ONNX and converted by rayo-convert with ``options``
    (``options.weights``: ``float32`` or ``int8``). Every tensor in ``required`` (by default
    :func:`expected_tensors`) must be among the converted ones, so a layer's name is never
    lost on the way. The weights' ``source`` is the ONNX file, named ``onnx_name`` (by default
    the model's name with ``.onnx``)."""
    blob = to_onnx(module, inputs, outputs)
    name = onnx_name or f"{options.name or 'model'}.onnx"
    result = convert(blob, name, options)
    want = expected_tensors(module) if required is None else required
    weights = weights_from_result(blob, name, result, options=options, layers=layers, required=want)
    return Export(blob, result, weights)


def export(
    module: torch.nn.Module,
    inputs: Mapping[str, torch.Tensor],
    outputs: Sequence[str],
    path: str | Path,
    options: Options,
    *,
    layers: list[dict[str, Any]],
    required: list[str] | None = None,
) -> Export:
    """:func:`weights_of_module`, written: the ``.weights`` file at ``path`` and the ONNX file
    beside it (``path`` with ``.onnx``), which the weights' ``source`` names."""
    out = Path(path)
    onnx_path = out.with_suffix(".onnx")
    done = weights_of_module(
        module, inputs, outputs, options, layers=layers, required=required, onnx_name=onnx_path.name
    )
    onnx_path.write_bytes(done.onnx)
    out.write_bytes(done.data)
    return done


@dataclass(frozen=True, slots=True)
class Parity:
    """The torch model against the converted file over a set of inputs: the largest absolute
    difference per output, and for the outputs that are labels (each row's arg-max: a
    question's intent, a token's slot), how many rows were compared and how many agree. A row
    whose two best torch scores are closer than the ``margin`` it was compared with is not
    counted: a difference within the tolerance may rightly reorder them."""

    max_abs: dict[str, float] = field(default_factory=dict)
    rows: int = 0
    same: int = 0
    close: int = 0  # rows left out: torch's top two within the margin

    def within(self, tolerance: float, outputs: Sequence[str] | None = None) -> bool:
        return all(v <= tolerance for k, v in self.max_abs.items() if not outputs or k in outputs)


def run_torch(module: torch.nn.Module, feeds: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """The torch model's outputs for one input (batch of one), as float32 without the batch."""
    torch = _torch()
    module.eval()
    with torch.no_grad():
        out = module(*(torch.from_numpy(np.asarray(v)[None]) for v in feeds.values()))
    outs = out if isinstance(out, tuple | list) else (out,)
    return [o.detach().numpy().astype(np.float32)[0] for o in outs]


def parity(
    module: torch.nn.Module,
    converted: Any,
    feeds: Sequence[Mapping[str, np.ndarray]],
    outputs: Sequence[str],
    *,
    labels: Sequence[str] = (),
    bind: Mapping[str, int] | None = None,
    margin: float = 0.0,
) -> Parity:
    """``module`` (torch, float) against ``converted`` (anything with ``run(feeds, bind)``
    giving the outputs in order: :class:`RayoRunner`, :class:`ReferenceRunner`) on each
    input in ``feeds``."""
    worst = dict.fromkeys(outputs, 0.0)
    rows = same = close = 0
    for f in feeds:
        want = run_torch(module, f)
        got = converted.run(f, bind)
        for name, w, out in zip(outputs, want, got, strict=True):
            g = np.asarray(out).reshape(w.shape)
            worst[name] = max(worst[name], float(np.abs(g - w).max()))
            if name not in labels:
                continue
            w2, g2 = w.reshape(-1, w.shape[-1]), g.reshape(-1, w.shape[-1])
            top = np.sort(w2, axis=-1)
            clear = top[:, -1] - top[:, -2] > margin
            close += int((~clear).sum())
            rows += int(clear.sum())
            same += int((w2.argmax(-1) == g2.argmax(-1))[clear].sum())
    return Parity(worst, rows, same, close)


class RayoRunner:
    """A converted file on rayo's C library (``rayo_convert.runtime``), built from a rayo
    checkout (``rayo_dir``: its ``src/`` and ``include/``)."""

    def __init__(self, data: bytes, rayo_dir: str | Path) -> None:
        from rayo_convert.runtime import Library, Model, Session

        graph = read_weights(data).graph or {}
        self.count = len(graph.get("outputs", []))
        self.session = Session(Model(Library.build(Path(rayo_dir)), data))

    def run(
        self, feeds: Mapping[str, np.ndarray], bind: Mapping[str, int] | None = None
    ) -> list[np.ndarray]:
        for symbol, value in (bind or {}).items():
            self.session.bind(symbol, value)
        self.session.run([_rayo_input(v) for v in feeds.values()])
        return [self.session.output(i) for i in range(self.count)]


class ReferenceRunner:
    """The source graph on rayo-convert's NumPy reference (``rayo_convert.ref``), with the
    int8 meaning the converter gives it: what rayo's C library is checked against, bit for
    bit in its weights, in rayo's own tests. Needs no rayo checkout."""

    def __init__(self, onnx: bytes, options: Options) -> None:
        from rayo_convert import onnx_frontend, passes, quantise

        g = onnx_frontend.read(onnx)
        passes.fold(g)
        if options.weights == "int8":
            quantise.apply(g, quantise.Int8(tuple(options.int8_tensors), options.int8_min_size))
        self.graph = g

    def run(
        self, feeds: Mapping[str, np.ndarray], bind: Mapping[str, int] | None = None
    ) -> list[np.ndarray]:
        from rayo_convert import ref

        g = self.graph
        values = ref.run(
            g,
            {
                i: np.asarray(v).reshape(g.tensors[i].shape)
                for i, v in zip(g.inputs, feeds.values(), strict=True)
            },
        )
        return [values[o] for o in g.outputs]


def _rayo_input(v: np.ndarray) -> np.ndarray:
    a = np.asarray(v)
    return a.astype(np.int32) if a.dtype.kind in "iu" else a.astype(np.float32)


def int8(options: Options) -> Options:
    """``options`` with rayo-convert's int8 weights from the float graph."""
    return replace(options, weights="int8")
