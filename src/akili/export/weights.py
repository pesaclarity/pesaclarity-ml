"""Plain weights: a model's tensors in a small file any language can read without TFLite.

The app runs small models in its own code instead of a native runtime, and reads their
weights from this file. The format is specified in [docs/WEIGHTS.md](../../../docs/WEIGHTS.md):

    "PCWT" | uint32 format version | uint32 header length | UTF-8 JSON header | tensor data

all little-endian. The header names the source model, lists the layers in the order they run
and gives each tensor's name, dtype, shape, offset and, for an int8 tensor, its quantisation.
This module writes and reads the file and needs only NumPy.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = b"PCWT"
FORMAT_VERSION = 1
_DTYPES: dict[str, np.dtype[Any]] = {"int8": np.dtype("<i1"), "float32": np.dtype("<f4")}


@dataclass(frozen=True, slots=True)
class Tensor:
    """One tensor. An int8 tensor's real values are ``scales[c] * (value - zero_point)``,
    ``c`` its index along ``axis`` (or one scale for the whole tensor when ``axis`` is None)."""

    value: np.ndarray
    scales: np.ndarray | None = None
    zero_point: int = 0
    axis: int | None = None

    def __post_init__(self) -> None:
        if self.value.dtype.name not in _DTYPES:
            raise ValueError(f"dtype {self.value.dtype.name} is not int8 or float32")
        if (self.value.dtype.name == "int8") != (self.scales is not None):
            raise ValueError("int8 tensors need scales, and only they have them")
        if self.scales is not None:
            want = 1 if self.axis is None else self.value.shape[self.axis]
            if self.scales.shape != (want,):
                raise ValueError(f"{self.scales.shape[0]} scales for {want} channels")

    def real(self) -> np.ndarray:
        """The values as float32, dequantised when int8."""
        if self.scales is None:
            return self.value.astype(np.float32)
        shape = [1] * self.value.ndim
        if self.axis is not None:
            shape[self.axis] = -1
        q = self.value.astype(np.float32) - np.float32(self.zero_point)
        return (q * self.scales.astype(np.float32).reshape(shape)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class Weights:
    """A model's layers and tensors, and the model file they were read from."""

    model: str
    source_file: str
    source_sha256: str
    source_size_bytes: int
    layers: list[dict[str, Any]]
    tensors: dict[str, Tensor] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        entries: list[dict[str, Any]] = []
        blobs: list[bytes] = []
        offset = 0

        def put(array: np.ndarray) -> tuple[int, int]:
            nonlocal offset
            data = array.astype(_DTYPES[array.dtype.name]).tobytes()
            at = offset
            blobs.append(data + b"\0" * (-len(data) % 4))
            offset += len(blobs[-1])
            return at, len(data)

        for name, t in self.tensors.items():
            at, size = put(t.value)
            entry: dict[str, Any] = {
                "name": name,
                "dtype": t.value.dtype.name,
                "shape": list(t.value.shape),
                "offset": at,
                "bytes": size,
            }
            if t.scales is not None:
                s_at, s_size = put(t.scales.astype(np.float32))
                entry["quantization"] = {
                    "axis": t.axis,
                    "zero_point": t.zero_point,
                    "scales_offset": s_at,
                    "scales_bytes": s_size,
                }
            entries.append(entry)
        header = json.dumps(
            {
                "format_version": FORMAT_VERSION,
                "model": self.model,
                "source": {
                    "file": self.source_file,
                    "sha256": self.source_sha256,
                    "size_bytes": self.source_size_bytes,
                },
                "layers": self.layers,
                "tensors": entries,
            },
            separators=(",", ":"),
        ).encode()
        # Pad with spaces so the tensor data starts on a 4-byte boundary.
        header += b" " * (-(12 + len(header)) % 4)
        return MAGIC + struct.pack("<II", FORMAT_VERSION, len(header)) + header + b"".join(blobs)

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.to_bytes())
        return out


def read_weights(blob: bytes) -> Weights:
    """Parse a weights file; raises ValueError on anything that does not fit the spec."""
    if len(blob) < 12 or blob[:4] != MAGIC:
        raise ValueError("not a plain weights file")
    version, header_len = struct.unpack_from("<II", blob, 4)
    if version != FORMAT_VERSION:
        raise ValueError(f"format version {version}, this reads {FORMAT_VERSION}")
    start = 12 + header_len
    if start % 4 or start > len(blob):
        raise ValueError("bad header length")
    header = json.loads(blob[12:start].decode("utf-8"))
    if header.get("format_version") != version:
        raise ValueError("header and file disagree on the format version")
    data = blob[start:]
    end = 0

    def array(offset: int, size: int, dtype: np.dtype, shape: list[int]) -> np.ndarray:
        if offset % 4 or offset < 0 or offset + size > len(data):
            raise ValueError("tensor outside the data")
        if size != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
            raise ValueError("tensor size does not match its shape")
        nonlocal end
        end = max(end, offset + size + (-size % 4))
        return np.frombuffer(data, dtype, count=size // dtype.itemsize, offset=offset).reshape(
            shape
        )

    tensors: dict[str, Tensor] = {}
    for e in header["tensors"]:
        if e["dtype"] not in _DTYPES:
            raise ValueError(f"dtype {e['dtype']}")
        value = array(e["offset"], e["bytes"], _DTYPES[e["dtype"]], e["shape"])
        q = e.get("quantization")
        if q is None:
            tensors[e["name"]] = Tensor(value.astype(value.dtype.newbyteorder("=")))
            continue
        n = 1 if q["axis"] is None else e["shape"][q["axis"]]
        scales = array(q["scales_offset"], q["scales_bytes"], _DTYPES["float32"], [n])
        tensors[e["name"]] = Tensor(
            value.astype(np.int8),
            scales.astype(np.float32),
            zero_point=int(q["zero_point"]),
            axis=q["axis"],
        )
    if end != len(data):
        raise ValueError("data after the last tensor")
    source = header["source"]
    return Weights(
        model=header["model"],
        source_file=source["file"],
        source_sha256=source["sha256"],
        source_size_bytes=source["size_bytes"],
        layers=header["layers"],
        tensors=tensors,
    )
