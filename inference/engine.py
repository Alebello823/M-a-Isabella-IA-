# inference/engine.py

"""
Motor de inferencia de Mía Isabella.

Backend:
    llama.cpp mediante llama-cli.

Diseñado para Termux / Android.

No utiliza llama-cpp-python.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from contracts.inference import InferenceEngineProtocol

logger = logging.getLogger(__name__)


class LlamaCppEngine(InferenceEngineProtocol):
    """
    Motor de inferencia basado en llama-cli.

    Compatible con llamadas del kernel que proporcionen:

        model_path
        n_ctx
        n_threads
        max_tokens
        temperature
    """

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_threads: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ):
        self.model_path = Path(model_path)

        self.n_ctx = max(512, int(n_ctx))
        self.n_threads = max(1, int(n_threads))
        self.max_tokens = max(1, int(max_tokens))
        self.temperature = max(0.0, float(temperature))

        # ---------------------------------------------------------
        # Buscar llama-cli
        # ---------------------------------------------------------

        self.llama_cli = shutil.which("llama-cli")

        if not self.llama_cli:
            raise RuntimeError(
                "No se encontró llama-cli en PATH.\n"
                "Instálalo con:\n"
                "pkg install llama-cpp"
            )

        # ---------------------------------------------------------
        # Verificar modelo
        # ---------------------------------------------------------

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {self.model_path}"
            )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"La ruta del modelo no es un archivo: "
                f"{self.model_path}"
            )

        logger.info(
            "Inicializando motor llama.cpp"
        )

        logger.info(
            "llama-cli: %s",
            self.llama_cli
        )

        logger.info(
            "Modelo: %s",
            self.model_path
        )

        logger.info(
            "Contexto: %d",
            self.n_ctx
        )

        logger.info(
            "Threads: %d",
            self.n_threads
        )

        logger.info(
            "Max tokens: %d",
            self.max_tokens
        )

        logger.info(
            "Temperature: %.2f",
            self.temperature
        )

        # ---------------------------------------------------------
        # Comprobar backend
        # ---------------------------------------------------------

        self._verify_backend()

        logger.info(
            "Motor de inferencia inicializado correctamente"
        )

    # =============================================================
    # BACKEND
    # =============================================================

    def _verify_backend(self) -> None:
        """
        Verifica que llama-cli pueda ejecutarse.
        """

        try:

            result = subprocess.run(
                [
                    self.llama_cli,
                    "--version",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )

        except subprocess.TimeoutExpired as exc:

            raise RuntimeError(
                "llama-cli tardó demasiado en responder."
            ) from exc

        except OSError as exc:

            raise RuntimeError(
                f"No se pudo ejecutar llama-cli: {exc}"
            ) from exc

        if result.returncode != 0:

            raise RuntimeError(
                "llama-cli está instalado pero "
                "no pudo ejecutarse.\n"
                f"{result.stderr.strip()}"
            )

        version = result.stdout.strip()

        if version:

            logger.info(
                "llama.cpp disponible: %s",
                version.splitlines()[0]
            )

    # =============================================================
    # GENERACIÓN
    # =============================================================

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Genera texto mediante llama-cli.

        Si max_tokens o temperature no se proporcionan,
        utiliza los valores configurados en el constructor.
        """

        if not prompt or not prompt.strip():
            return ""

        # Valores por defecto del motor.

        if max_tokens is None:
            max_tokens = self.max_tokens

        if temperature is None:
            temperature = self.temperature

        max_tokens = max(
            1,
            int(max_tokens)
        )

        temperature = max(
            0.0,
            float(temperature)
        )

        # ---------------------------------------------------------
        # Comando llama-cli
        # ---------------------------------------------------------

        command = [
            self.llama_cli,

            "-m",
            str(self.model_path),

            "-c",
            str(self.n_ctx),

            "-n",
            str(max_tokens),

            "-t",
            str(self.n_threads),

            "--temp",
            str(temperature),

            "-p",
            prompt,
        ]

        logger.debug(
            "Ejecutando llama-cli: %s",
            self.model_path.name
        )

        # ---------------------------------------------------------
        # Entorno
        # ---------------------------------------------------------

        env = os.environ.copy()

        # ---------------------------------------------------------
        # Ejecutar
        # ---------------------------------------------------------

        try:

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=300,
            )

        except subprocess.TimeoutExpired as exc:

            logger.error(
                "Timeout durante la inferencia."
            )

            raise RuntimeError(
                "El modelo tardó demasiado en responder."
            ) from exc

        except OSError as exc:

            logger.exception(
                "Error ejecutando llama-cli."
            )

            raise RuntimeError(
                f"No se pudo ejecutar llama-cli: {exc}"
            ) from exc

        # ---------------------------------------------------------
        # Error del proceso
        # ---------------------------------------------------------

        if process.returncode != 0:

            stderr = process.stderr.strip()

            logger.error(
                "llama-cli terminó con código %s",
                process.returncode,
            )

            if stderr:

                logger.error(
                    "llama-cli stderr: %s",
                    stderr,
                )

            raise RuntimeError(
                "llama-cli terminó con un error. "
                f"Código: {process.returncode}"
            )

        # ---------------------------------------------------------
        # Resultado
        # ---------------------------------------------------------

        response = process.stdout.strip()

        if not response:

            logger.warning(
                "llama-cli no produjo texto."
            )

            return ""

        return self._clean_output(response)

    # =============================================================
    # LIMPIEZA
    # =============================================================

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Elimina delimitadores de conversación que puedan
        aparecer en la respuesta.
        """

        text = text.strip()

        stop_markers = [
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "\nUser:",
            "\nAssistant:",
        ]

        for marker in stop_markers:

            if marker in text:

                text = text.split(
                    marker,
                    1
                )[0]

        return text.strip()

    # =============================================================
    # INFORMACIÓN DEL MOTOR
    # =============================================================

    def info(self) -> Dict[str, Any]:
        """
        Información del motor actualmente activo.
        """

        return {
            "backend": "llama.cpp",
            "executable": self.llama_cli,
            "model": str(self.model_path),
            "model_name": self.model_path.name,
            "context": self.n_ctx,
            "threads": self.n_threads,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
