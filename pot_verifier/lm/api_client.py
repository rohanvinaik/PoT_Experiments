from __future__ import annotations
from typing import Protocol, runtime_checkable, Dict, Any


@runtime_checkable
class APIGenerator(Protocol):
    """
    Implement this protocol for your API (OpenAI/Azure/your vendor).
    Must be deterministic given fixed params.
    """

    def generate(self, prompt: str) -> str: ...
    def name(self) -> str: ...


class DummyAPIModel:
    """Replace with a real API client. Deterministic echo for now."""

    def __init__(self, model_id: str = "api-demo", system: str | None = None):
        self._model_id = model_id
        self._system = system

    def generate(self, prompt: str) -> str:
        return f"[api:{self._model_id}] {prompt}"

    def name(self) -> str:
        return f"api:{self._model_id}"