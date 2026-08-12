"""
Proveedor de inferencia local para Mía Isabella.

Responsabilidad única:
- Ejecutar un modelo GGUF local mediante llama-cli.
- No conoce Kernel.
- No conoce Orchestrator.
- No depende de Qwen, Phi ni ningún modelo concreto.
- El modelo se selecciona mediante configuración.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class LocalLlamaProvider:
    """Backend local de inferencia basado en llama-cli."""

    def __init__(
        self,
        model_path: str | Path,
        llama_cli: Optional[str] = None,
        context_size: int = 2048,
        threads: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.7,
        timeout: int = 180,
    ):
        self.model_path = Path(model_path).expanduser()
        self.context_size = int(context_size)
        self.threads = int(threads)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.timeout = int(timeout)

        self.llama_cli = (
            llama_cli
            or shutil.which("llama-cli")
        )

        if not self.llama_cli:
            raise RuntimeError(
                "No se encontró llama-cli. "
                "Mía Isabella necesita un backend local "
                "de inferencia."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo local: {self.model_path}"
            )

        logger.info(
            "LocalLlamaProvider listo: model=%s",
            self.model_path,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Genera una respuesta usando exclusivamente el modelo local."""

        if not prompt or not prompt.strip():
            return ""

        tokens = (
            self.max_tokens
            if max_tokens is None
            else max(1, int(max_tokens))
        )

        temp = (
            self.temperature
            if temperature is None
            else max(0.0, float(temperature))
        )

        command = [
            self.llama_cli,
            "-m",
            str(self.model_path),
            "-c",
            str(self.context_size),
            "-n",
            str(tokens),
            "-t",
            str(self.threads),
            "--temp",
            str(temp),
            "-p",
            prompt,
        ]

        logger.debug(
            "Ejecutando backend local: %s",
            self.model_path.name,
        )

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
            )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "El modelo local agotó el tiempo de inferencia."
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"No se pudo ejecutar llama-cli: {exc}"
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr.strip()

            raise RuntimeError(
                "El backend local terminó con error."
                + (f"\n{stderr}" if stderr else "")
            )

        output = result.stdout.strip()

        if not output:
            raise RuntimeError(
                "El modelo local no produjo texto."
            )

        return self._clean_output(output)

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Limpia únicamente artefactos típicos de llama-cli.

        No intenta interpretar ni modificar la respuesta.
        """

        if "<|assistant|>" in text:
            text = text.split(
                "<|assistant|>",
                1,
            )[1]

        if "<|user|>" in text:
            text = text.split(
                "<|user|>",
                1,
            )[0]

        if "<|system|>" in text:
            text = text.replace(
                "<|system|>",
                "",
            )

        return text.strip()
