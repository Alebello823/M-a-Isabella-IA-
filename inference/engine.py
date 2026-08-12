"""
Motor de inferencia de Mía Isabella.

Responsabilidad:
- Ejecutar llama.cpp mediante llama-cli.
- Realizar generaciones ONE-SHOT.
- No mantener una sesión interactiva.
- No depender de Qwen, Phi ni de ningún modelo concreto.
- Exponer una interfaz estable para el Kernel y el Orchestrator.

Arquitectura:

    Orchestrator
          |
          v
    InferenceEngineProtocol
          |
          v
    LlamaCppEngine
          |
          v
       llama-cli
          |
          v
       GGUF model

El modelo es intercambiable.
Mía Isabella no depende de la identidad de ningún LLM.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class LlamaCppEngine:
    """
    Motor de inferencia basado en llama.cpp/llama-cli.

    Diseñado para ejecutar una generación independiente por llamada.

    IMPORTANTE:
    llama-cli puede entrar automáticamente en modo conversación cuando
    detecta un chat template. Para Mía esto es incorrecto porque el
    proceso debe terminar después de generar una respuesta.

    Por eso utilizamos --single-turn.
    """

    def __init__(
        self,
        model_path,
        n_ctx: int = 2048,
        n_threads: int = 4,
        max_tokens: int = 256,
        temperature: float = 0.7,
        timeout: int = 180,
        llama_cli: Optional[str] = None,
        **kwargs,
    ):
        self.model_path = Path(model_path)

        self.n_ctx = max(128, int(n_ctx))
        self.n_threads = max(1, int(n_threads))
        self.max_tokens = max(1, int(max_tokens))
        self.temperature = max(0.0, float(temperature))
        self.timeout = max(10, int(timeout))

        self.llama_cli = (
            llama_cli
            or shutil.which("llama-cli")
        )

        if not self.llama_cli:
            raise RuntimeError(
                "No se encontró llama-cli en PATH. "
                "Instala llama.cpp antes de iniciar Mía Isabella."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo GGUF: {self.model_path}"
            )

        logger.info(
            "Inicializando motor llama.cpp"
        )

        logger.info(
            "llama-cli: %s",
            self.llama_cli,
        )

        logger.info(
            "Modelo: %s",
            self.model_path,
        )

        logger.info(
            "Contexto: %d",
            self.n_ctx,
        )

        logger.info(
            "Threads: %d",
            self.n_threads,
        )

        logger.info(
            "Max tokens: %d",
            self.max_tokens,
        )

        logger.info(
            "Temperature: %.2f",
            self.temperature,
        )

        self._verify_llama_cli()

    # ============================================================
    # VERIFICACIÓN
    # ============================================================

    def _verify_llama_cli(self) -> None:
        """
        Comprueba que llama-cli puede ejecutarse.

        No carga el modelo aquí.
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
                check=False,
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
            # Algunas builds pueden no implementar --version
            # de forma tradicional. No abortamos solamente por eso.
            logger.warning(
                "llama-cli devolvió código %s durante la "
                "verificación.",
                result.returncode,
            )

            if result.stderr.strip():
                logger.warning(
                    "llama-cli stderr: %s",
                    result.stderr.strip(),
                )

            return

        version = result.stdout.strip()

        if version:
            logger.info(
                "llama.cpp disponible: %s",
                version.splitlines()[0],
            )

    # ============================================================
    # GENERACIÓN
    # ============================================================

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Genera una respuesta independiente mediante llama-cli.

        IMPORTANTE:

        Esta función NO abre una conversación interactiva.

        Cada llamada:

            prompt
              |
              v
          llama-cli
              |
              v
          respuesta
              |
              v
          proceso termina

        Esto permite que Orchestrator continúe inmediatamente.
        """

        if not prompt or not prompt.strip():
            return ""

        if max_tokens is None:
            max_tokens = self.max_tokens

        if temperature is None:
            temperature = self.temperature

        max_tokens = max(
            1,
            int(max_tokens),
        )

        temperature = max(
            0.0,
            float(temperature),
        )

        # --------------------------------------------------------
        # Comando ONE-SHOT
        # --------------------------------------------------------

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

            # IMPORTANTE:
            # fuerza una sola interacción y salida.
            "--single-turn",

            # No necesitamos que llama-cli imprima el prompt.
            "--no-display-prompt",

            # Evita los timings de CLI mezclándose con la respuesta.
            "--no-show-timings",

            "-p",
            prompt,
        ]

        logger.debug(
            "Ejecutando llama-cli en modo one-shot."
        )

        logger.debug(
            "Modelo: %s",
            self.model_path,
        )

        logger.debug(
            "Contexto: %d",
            self.n_ctx,
        )

        logger.debug(
            "Max tokens: %d",
            max_tokens,
        )

        logger.debug(
            "Threads: %d",
            self.n_threads,
        )

        logger.debug(
            "Timeout: %ds",
            self.timeout,
        )

        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
                stdin=subprocess.DEVNULL,
                env=os.environ.copy(),
            )

        except subprocess.TimeoutExpired as exc:
            logger.error(
                "Timeout ejecutando llama-cli después de %ds.",
                self.timeout,
            )

            raise RuntimeError(
                "El motor de inferencia superó el tiempo máximo "
                f"de {self.timeout} segundos."
            ) from exc

        except OSError as exc:
            logger.error(
                "No se pudo ejecutar llama-cli: %s",
                exc,
            )

            raise RuntimeError(
                f"No se pudo ejecutar llama-cli: {exc}"
            ) from exc

        # --------------------------------------------------------
        # ERROR DEL PROCESO
        # --------------------------------------------------------

        if process.returncode != 0:
            stderr = (
                process.stderr.strip()
                if process.stderr
                else ""
            )

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
                f"Código: {process.returncode}. "
                f"{stderr}"
            )

        # --------------------------------------------------------
        # LIMPIAR RESPUESTA
        # --------------------------------------------------------

        stdout = (
            process.stdout.strip()
            if process.stdout
            else ""
        )

        stderr = (
            process.stderr.strip()
            if process.stderr
            else ""
        )

        # Algunos builds escriben información técnica en stderr.
        if stderr:
            logger.debug(
                "llama-cli stderr: %s",
                stderr,
            )

        if not stdout:
            logger.warning(
                "llama-cli terminó correctamente pero no produjo texto."
            )

            return ""

        # --------------------------------------------------------
        # FILTRADO DEFENSIVO
        # --------------------------------------------------------
        #
        # Aunque --single-turn debería impedir el modo interactivo,
        # eliminamos elementos del CLI que jamás deben llegar al
        # usuario de Mía.

        response = self._clean_output(stdout)

        if not response:
            logger.warning(
                "La salida de llama-cli quedó vacía después "
                "de la limpieza."
            )

            return ""

        logger.debug(
            "Inferencia completada: %d caracteres.",
            len(response),
        )

        return response

    # ============================================================
    # LIMPIEZA
    # ============================================================

    def _clean_output(self, output: str) -> str:
        """
        Limpia salida de llama-cli.

        No intenta interpretar la respuesta del modelo.

        Solo elimina elementos propios de la interfaz CLI.
        """

        if not output:
            return ""

        lines = output.splitlines()

        cleaned = []

        cli_markers = (
            "Loading model...",
            "available commands:",
            "Exiting...",
        )

        skip_commands = (
            "/exit",
            "/regen",
            "/clear",
            "/read ",
            "/glob ",
        )

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Conservamos espacios internos únicamente si ya
                # existe contenido.
                if cleaned:
                    cleaned.append("")
                continue

            # Ignorar comandos interactivos accidentales.
            if stripped.startswith(skip_commands):
                continue

            # Ignorar mensajes conocidos del CLI.
            if stripped in cli_markers:
                continue

            # Ignorar prompt vacío del CLI.
            if stripped == ">":
                continue

            cleaned.append(stripped)

        return "\n".join(cleaned).strip()

    # ============================================================
    # INFORMACIÓN
    # ============================================================

    def health_check(self) -> bool:
        """
        Verifica que el ejecutable y el modelo estén disponibles.
        """

        if not self.llama_cli:
            return False

        if not Path(self.llama_cli).exists():
            return False

        if not self.model_path.exists():
            return False

        return True

    def get_info(self):
        """
        Devuelve información básica del motor.
        """

        return {
            "backend": "llama.cpp",
            "executable": self.llama_cli,
            "model": str(self.model_path),
            "context": self.n_ctx,
            "threads": self.n_threads,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "mode": "single-turn",
        }
