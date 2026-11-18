"""akili.export.onnx in the core install: it imports without PyTorch, and asks for the
[torch] extra when a model is exported."""

from __future__ import annotations

import pytest

from akili._optional import has_module


def test_imports_without_torch() -> None:
    from akili.export import onnx

    assert onnx.OPSET == 17


@pytest.mark.skipif(has_module("torch"), reason="the [torch] extra is installed")
def test_export_names_the_extra() -> None:
    from akili.export import onnx

    with pytest.raises(ImportError, match=r"Install the \[torch\] extra"):
        onnx.deterministic()
