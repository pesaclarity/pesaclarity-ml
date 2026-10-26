"""Flower adapter (needs the [fl] extra).

Wraps :class:`CategoriserClient` as a Flower ``NumPyClient`` and ``ClientApp``. The matching
``ServerApp`` and strategy (FedAvg with secure aggregation and cohort minimums) live in
pesaclarity-backend; the phone runs the same logic in Dart/TFLite. This adapter exists for
simulation and for testing the protocol against the real aggregator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from akili._optional import require
from akili.federated.client import CategoriserClient


def to_numpy_client(client: CategoriserClient) -> Any:
    """Return a ``flwr.client.NumPyClient`` that delegates to ``client``."""
    flwr_client = require("flwr.client", "fl")

    class _Adapter(flwr_client.NumPyClient):  # type: ignore[misc,name-defined]
        def get_properties(self, config: Any) -> Any:
            return client.get_properties(config)

        def get_parameters(self, config: Any) -> Any:
            return client.get_parameters(config)

        def fit(self, parameters: Any, config: Any) -> Any:
            return client.fit(parameters, config)

        def evaluate(self, parameters: Any, config: Any) -> Any:
            return client.evaluate(parameters, config)

    return _Adapter()


def client_app(make_client: Callable[[Any], CategoriserClient]) -> Any:
    """Build a ``flwr.clientapp.ClientApp``; ``make_client`` receives the Flower ``Context``."""
    clientapp = require("flwr.clientapp", "fl")

    def client_fn(context: Any) -> Any:
        return to_numpy_client(make_client(context)).to_client()

    return clientapp.ClientApp(client_fn=client_fn)
