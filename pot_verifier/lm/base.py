from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    """Abstract model interface that returns deterministic text for a prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class EchoModel(Model):
    """Tiny stub: echoes prompt tail. Replace in tests or fall back when no HF/API."""

    def __init__(self, tag: str = "echo"):
        self._tag = tag

    def generate(self, prompt: str) -> str:
        return f"[{self._tag}] {prompt}"

    def name(self) -> str:
        return f"echo:{self._tag}"