"""
Motor de inferencia de respaldo de Mía Isabella.

Este módulo NO depende de ningún LLM.

Su objetivo es mantener viva la arquitectura cuando el backend
de lenguaje no esté disponible.

El fallback no pretende sustituir al modelo lingüístico.
Proporciona respuestas mínimas y seguras para que el sistema
pueda continuar funcionando.
"""

import logging
from typing import Optional


logger = logging.getLogger(__name__)


class FallbackInferenceEngine:
    """
    Motor de inferencia mínimo sin dependencia de modelos.

    Características:

    - No requiere Qwen.
    - No requiere Phi.
    - No requiere llama.cpp.
    - No requiere embeddings.
    - No modifica memoria.
    - No ejecuta herramientas.
    - No ejecuta acciones.
    """

    def __init__(
        self,
        assistant_name: str = "Mía Isabella",
    ):
        self.assistant_name = assistant_name

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Genera una respuesta básica sin utilizar un modelo.
        """

        if not prompt or not prompt.strip():
            return ""

        user_message = self._extract_user_message(prompt)

        if not user_message:
            return (
                f"{self.assistant_name} está disponible, "
                "pero no recibió ningún mensaje."
            )

        normalized = user_message.lower().strip()

        # ---------------------------------------------------------
        # SALUDOS
        # ---------------------------------------------------------

        if normalized in {
            "hola",
            "hola.",
            "buenas",
            "buenas.",
            "hey",
            "hello",
        }:
            return (
                f"Hola. Soy {self.assistant_name}. "
                "Estoy funcionando en modo local."
            )

        # ---------------------------------------------------------
        # IDENTIDAD
        # ---------------------------------------------------------

        if normalized in {
            "quien eres",
            "quién eres",
            "quien eres?",
            "quién eres?",
        }:
            return (
                f"Soy {self.assistant_name}, "
                "una asistente personal modular y local."
            )

        # ---------------------------------------------------------
        # ESTADO
        # ---------------------------------------------------------

        if (
            "cómo estás" in normalized
            or "como estas" in normalized
        ):
            return (
                "Estoy funcionando correctamente en modo local."
            )

        # ---------------------------------------------------------
        # AGRADECIMIENTO
        # ---------------------------------------------------------

        if normalized in {
            "gracias",
            "gracias.",
            "muchas gracias",
        }:
            return "De nada."

        # ---------------------------------------------------------
        # FALLBACK GENERAL
        # ---------------------------------------------------------

        return (
            "He recibido tu mensaje, pero el motor lingüístico "
            "principal no está disponible en este momento.\n\n"
            "Mi núcleo, memoria y sistema cognitivo continúan "
            "funcionando."
        )

    # =============================================================
    # EXTRACCIÓN DEL MENSAJE
    # =============================================================

    @staticmethod
    def _extract_user_message(prompt: str) -> str:
        """
        Extrae el último mensaje marcado como <|user|>.

        Si el prompt no utiliza ese formato, devuelve el texto
        completo.
        """

        marker = "<|user|>"

        if marker not in prompt:
            return prompt.strip()

        content = prompt.rsplit(marker, 1)[1]

        if "<|assistant|>" in content:
            content = content.split(
                "<|assistant|>",
                1,
            )[0]

        return content.strip()
