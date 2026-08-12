"""
Contrato del motor de inferencia de Mía Isabella.

El resto del sistema no debe conocer:
- Qwen
- Phi
- Llama
- llama.cpp
- Transformers

Solo conoce este contrato.
"""

from typing import Protocol


class InferenceEngineProtocol(Protocol):

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        """
        Genera texto a partir de un prompt.
        """
        ...
