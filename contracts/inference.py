# contracts/inference.py
from typing import Protocol

class InferenceEngineProtocol(Protocol):
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        ...
