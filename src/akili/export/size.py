"""Size budget for models that ship to 2 GB RAM phones."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


class ModelTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class ExportedModel:
    path: Path
    size_bytes: int
    sha256: str


def describe(path: str | Path) -> ExportedModel:
    p = Path(path)
    data = p.read_bytes()
    return ExportedModel(path=p, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def check_size(path: str | Path, max_size_bytes: int) -> ExportedModel:
    """Describe the exported file and raise if it is over budget."""
    exported = describe(path)
    if exported.size_bytes > max_size_bytes:
        raise ModelTooLargeError(
            f"{exported.path.name} is {exported.size_bytes} bytes; the budget is {max_size_bytes}"
        )
    return exported
