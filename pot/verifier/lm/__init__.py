"""Language model interfaces for the behavioral verifier."""

from .base import Model, EchoModel
from .api_client import DummyAPIModel

try:  # Optional dependency
    from .hf_local import HFLocalModel
except Exception:  # pragma: no cover - transformers may be missing
    HFLocalModel = None

__all__ = [
    "Model",
    "EchoModel",
    "DummyAPIModel",
    "HFLocalModel",
]
