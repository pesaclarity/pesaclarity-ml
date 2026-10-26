"""Helpers for the optional [train] and [fl] extras."""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require(module: str, extra: str) -> ModuleType:
    """Import ``module`` or explain which extra installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is not installed. Install the [{extra}] extra: uv sync --extra {extra}"
        ) from exc
