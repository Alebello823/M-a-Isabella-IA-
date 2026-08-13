"""
Motor de inferencia de Mía Isabella.

Responsabilidades
-----------------
- Ejecutar modelos GGUF mediante llama-cli.
- Mantener una interfaz estable con Kernel/Orchestrator.
- Ejecutar una generación por petición.
- Controlar llama-cli correctamente en Termux.
- Utilizar el mecanismo de salida comprobado en Termux:
      printf '/exit\\n' | llama-cli ...
- Extraer únicamente la respuesta generada.
- No depender de Qwen, Phi ni de ningún modelo concreto.

Arquitectura

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
       GGUF

IMPORTANTE
----------
La build de llama-cli utilizada en Termux entra en una interfaz
interactiva aunque reciba -p.

Las pruebas reales demostraron que:

    printf '/exit\\n' | llama-cli ...

sí permite:

    1. cargar el modelo
    2. generar
    3. procesar /exit
    4. terminar correctamente

Por eso Python reproduce exactamente ese flujo.

NO se utilizan:
- selectors
- hilos de lectura
- polling de stdout
- detección artificial de final de generación
- /exit enviado después de leer stdout
- --single-turn

El timeout se utiliza únicamente como límite de seguridad.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class LlamaCppEngine:
    """
    Adaptador de inferencia de Mía Isabella para llama.cpp.

    El modelo es completamente intercambiable.

    Ejemplo:

        engine = LlamaCppEngine(
            model_path="models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            n_ctx=512,
            n_threads=4,
            max_tokens=32,
            temperature=0.7,
            timeout=45,
        )

        response = engine.generate(
            "Responde solamente: HOLA"
        )
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

        self.n_ctx = max(
            128,
            int(n_ctx),
        )

        self.n_threads = max(
            1,
            int(n_threads),
        )

        self.max_tokens = max(
            1,
            int(max_tokens),
        )

        self.temperature = max(
            0.0,
            float(temperature),
        )

        self.timeout = max(
            5,
            int(timeout),
        )

        self.llama_cli = (
            llama_cli
            or shutil.which("llama-cli")
        )

        if not self.llama_cli:
            raise RuntimeError(
                "No se encontró llama-cli en PATH. "
                "Instala llama.cpp antes de iniciar "
                "Mía Isabella."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No existe el modelo GGUF: "
                f"{self.model_path}"
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

        logger.info(
            "Timeout: %ds",
            self.timeout,
        )

        self._verify_llama_cli()

    # ============================================================
    # VERIFICACIÓN
    # ============================================================

    def _verify_llama_cli(self) -> None:
        """
        Comprueba que llama-cli puede ejecutarse.

        No carga el modelo.
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
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "llama-cli tardó demasiado "
                "durante la verificación."
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                f"No se pudo ejecutar llama-cli: {exc}"
            ) from exc

        output = self._merge_streams(
            result.stdout,
            result.stderr,
        )

        if output:
            first_line = (
                output.splitlines()[0].strip()
            )

            logger.info(
                "llama.cpp disponible: %s",
                first_line,
            )

        if result.returncode != 0:
            logger.warning(
                "llama-cli devolvió código %s "
                "durante la verificación.",
                result.returncode,
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
        Genera una respuesta mediante llama-cli.

        Flujo real:

            Python
              |
              v
            Popen
              |
              v
        llama-cli + GGUF
              |
              v
        generación
              |
              v
        /exit por stdin
              |
              v
        llama-cli termina
              |
              v
        parser
              |
              v
        respuesta limpia
        """

        if not prompt or not prompt.strip():
            return ""

        if max_tokens is None:
            max_tokens = self.max_tokens
        else:
            max_tokens = max(
                1,
                int(max_tokens),
            )

        if temperature is None:
            temperature = self.temperature
        else:
            temperature = max(
                0.0,
                float(temperature),
            )

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

            "-no-cnv",

            "--simple-io",

            "--no-display-prompt",

            "--no-show-timings",

            "-p",
            prompt,
        ]

        logger.debug(
            "Ejecutando llama-cli one-shot."
        )

        logger.debug(
            "Comando: %s",
            command,
        )

        process: Optional[
            subprocess.Popen
        ] = None

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=os.environ.copy(),
                start_new_session=True,
            )

            logger.debug(
                "llama-cli iniciado. PID=%s",
                process.pid,
            )

            stdout, stderr = process.communicate(
                input="/exit\n",
                timeout=self.timeout,
            )

        except subprocess.TimeoutExpired as exc:
            logger.error(
                "llama-cli superó el timeout de %ds.",
                self.timeout,
            )

            self._stop_process(
                process
            )

            raise RuntimeError(
                "El motor de inferencia superó "
                f"el tiempo máximo de "
                f"{self.timeout} segundos."
            ) from exc

        except KeyboardInterrupt:
            logger.warning(
                "Interrupción durante la inferencia."
            )

            self._stop_process(
                process
            )

            raise

        except OSError as exc:
            self._stop_process(
                process
            )

            logger.error(
                "No se pudo ejecutar llama-cli: %s",
                exc,
            )

            raise RuntimeError(
                f"No se pudo ejecutar llama-cli: {exc}"
            ) from exc

        finally:
            if process is not None:
                self._close_pipes(
                    process
                )

        if process is None:
            raise RuntimeError(
                "No se pudo crear el proceso llama-cli."
            )

        logger.debug(
            "llama-cli terminó con código %s.",
            process.returncode,
        )

        if stderr:
            self._log_stderr(
                stderr
            )

        if process.returncode not in (
            0,
            None,
        ):
            raise RuntimeError(
                "llama-cli terminó con código "
                f"{process.returncode}."
            )

        response = self._extract_response(
            stdout=stdout or "",
            prompt=prompt,
        )

        if not response:
            logger.warning(
                "llama-cli terminó correctamente "
                "pero no se pudo extraer una respuesta."
            )

            logger.debug(
                "stdout bruto:\n%s",
                stdout,
            )

            return ""

        logger.debug(
            "Respuesta generada correctamente: "
            "%d caracteres.",
            len(response),
        )

        return response

    # ============================================================
    # PARSER DE RESPUESTA
    # ============================================================

    def _extract_response(
        self,
        stdout: str,
        prompt: str,
    ) -> str:
        """
        Extrae únicamente el texto generado.

        Ejemplo real observado:

            Loading model...

            banner

            > Responde solamente: HOLA
            Hola!

            [ Prompt: 13.9 t/s | Generation: 11.0 t/s ]

            > /exit

            Exiting...

        Resultado:

            Hola!
        """

        if not stdout:
            return ""

        text = stdout.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        text = self._remove_ansi(
            text
        )

        lines = text.splitlines()

        prompt_clean = prompt.strip()

        start = self._find_prompt_start(
            lines,
            prompt_clean,
        )

        if start is None:
            logger.debug(
                "No se encontró el prompt "
                "dentro de stdout."
            )

            return ""

        response_lines = []

        for line in lines[start:]:
            stripped = line.strip()

            if not stripped:
                continue

            if self._is_generation_end(
                stripped
            ):
                break

            response_lines.append(
                line
            )

        response = "\n".join(
            response_lines
        ).strip()

        return self._clean_response(
            response
        )

    # ============================================================
    # LOCALIZAR PROMPT
    # ============================================================

    @staticmethod
    def _find_prompt_start(
        lines: list[str],
        prompt: str,
    ) -> Optional[int]:
        """
        Localiza la línea inmediatamente posterior
        al prompt introducido en llama-cli.
        """

        if not prompt:
            return None

        for index, line in enumerate(lines):
            stripped = line.strip()

            if stripped == prompt:
                return index + 1

            if stripped == f"> {prompt}":
                return index + 1

        escaped = re.escape(
            prompt
        )

        pattern = re.compile(
            rf"^\s*>\s*{escaped}\s*$"
        )

        for index, line in enumerate(lines):
            if pattern.match(line):
                return index + 1

        return None

    # ============================================================
    # FIN DE GENERACIÓN
    # ============================================================

    @staticmethod
    def _is_generation_end(
        line: str,
    ) -> bool:
        """
        Determina si una línea pertenece al control
        de llama-cli y no a la respuesta.
        """

        if not line:
            return False

        if line in (
            ">",
            ">>>",
            "/exit",
            "> /exit",
            "Exiting...",
        ):
            return True

        if re.match(
            r"^\[\s*Prompt:",
            line,
            flags=re.IGNORECASE,
        ):
            return True

        if re.match(
            r"^\[\s*Generation:",
            line,
            flags=re.IGNORECASE,
        ):
            return True

        return False

    # ============================================================
    # LIMPIEZA
    # ============================================================

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Limpia restos de la interfaz de llama-cli.
        """

        if not response:
            return ""

        response = re.sub(
            r"\[\s*Prompt:\s*[\d.]+\s*t/s"
            r".*?\]",
            "",
            response,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        response = re.sub(
            r"\[\s*Generation:\s*[\d.]+\s*t/s"
            r".*?\]",
            "",
            response,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        response = re.sub(
            r"^\s*Exiting\.\.\.\s*$",
            "",
            response,
            flags=re.MULTILINE,
        )

        response = re.sub(
            r"^\s*> /exit\s*$",
            "",
            response,
            flags=re.MULTILINE,
        )

        response = re.sub(
            r"^\s*>\s*$",
            "",
            response,
            flags=re.MULTILINE,
        )

        return response.strip()

    # ============================================================
    # ANSI
    # ============================================================

    @staticmethod
    def _remove_ansi(
        text: str,
    ) -> str:
        """
        Elimina secuencias ANSI de terminal.
        """

        return re.sub(
            r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])",
            "",
            text,
        )

    # ============================================================
    # CONTROL DE PROCESO
    # ============================================================

    def _stop_process(
        self,
        process: Optional[subprocess.Popen],
    ) -> None:
        """
        Detiene llama-cli solamente cuando realmente
        existe un timeout, interrupción o error.

        Primero intenta terminar el grupo de procesos.
        Después fuerza kill si es necesario.
        """

        if process is None:
            return

        if process.poll() is not None:
            return

        logger.warning(
            "Finalizando proceso llama-cli "
            "PID=%s.",
            process.pid,
        )

        try:
            pgid = os.getpgid(
                process.pid
            )

        except OSError:
            pgid = None

        try:
            if pgid is not None:
                os.killpg(
                    pgid,
                    signal.SIGTERM,
                )
            else:
                process.terminate()

        except OSError:
            try:
                process.terminate()
            except OSError:
                pass

        try:
            process.wait(
                timeout=3
            )

            return

        except subprocess.TimeoutExpired:
            logger.warning(
                "llama-cli no terminó con "
                "terminate(); forzando cierre."
            )

        try:
            if pgid is not None:
                os.killpg(
                    pgid,
                    signal.SIGKILL,
                )
            else:
                process.kill()

        except OSError:
            try:
                process.kill()
            except OSError:
                pass

        try:
            process.wait(
                timeout=3
            )

        except subprocess.TimeoutExpired:
            logger.error(
                "No fue posible confirmar el "
                "cierre de llama-cli."
            )

    # ============================================================
    # CERRAR PIPES
    # ============================================================

    @staticmethod
    def _close_pipes(
        process: Optional[subprocess.Popen],
    ) -> None:
        """
        Cierra pipes restantes de forma segura.
        """

        if process is None:
            return

        for name in (
            "stdin",
            "stdout",
            "stderr",
        ):
            stream = getattr(
                process,
                name,
                None,
            )

            if stream is None:
                continue

            try:
                stream.close()

            except (
                OSError,
                ValueError,
            ):
                pass

    # ============================================================
    # STDERR
    # ============================================================

    def _log_stderr(
        self,
        stderr: str,
    ) -> None:
        """
        Registra stderr.

        ggml_opencl: platform IDs not available
        no se considera un error fatal porque
        las pruebas reales demostraron que el
        modelo continúa funcionando.
        """

        if not stderr:
            return

        for line in stderr.splitlines():
            line = line.strip()

            if not line:
                continue

            if "ggml_opencl" in line.lower():
                logger.debug(
                    "llama.cpp: %s",
                    line,
                )
                continue

            logger.debug(
                "llama-cli: %s",
                line,
            )

    # ============================================================
    # UTILIDADES
    # ============================================================

    @staticmethod
    def _merge_streams(
        stdout: Optional[str],
        stderr: Optional[str],
    ) -> str:
        """
        Combina stdout y stderr para diagnósticos.
        """

        parts = []

        if stdout:
            parts.append(
                stdout.strip()
            )

        if stderr:
            parts.append(
                stderr.strip()
            )

        return "\n".join(
            parts
        )

    # ============================================================
    # INFORMACIÓN
    # ============================================================

    def get_info(self) -> dict:
        """
        Devuelve información serializable del motor.
        """

        return {
            "backend": "llama.cpp",
            "executable": str(
                self.llama_cli
            ),
            "model": str(
                self.model_path
            ),
            "context": self.n_ctx,
            "threads": self.n_threads,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }

    @property
    def info(self) -> dict:
        """
        Compatibilidad con:

            engine.info

        NO utilizar:

            engine.info()
        """

        return self.get_info()

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self) -> bool:
        """
        Comprueba disponibilidad del ejecutable
        y del modelo sin cargarlo.
        """

        if not self.model_path.exists():
            return False

        if not self.model_path.is_file():
            return False

        executable = Path(
            self.llama_cli
        )

        if executable.exists():
            return True

        resolved = shutil.which(
            self.llama_cli
        )

        return bool(
            resolved
        )

    # ============================================================
    # DIAGNÓSTICO
    # ============================================================

    def diagnostics(self) -> dict:
        """
        Devuelve un diagnóstico del estado actual
        del motor sin ejecutar una inferencia.

        Útil para telemetry y debugging.
        """

        executable_exists = False

        if self.llama_cli:
            executable_exists = (
                Path(
                    self.llama_cli
                ).exists()
            )

            if not executable_exists:
                executable_exists = bool(
                    shutil.which(
                        self.llama_cli
                    )
                )

        return {
            "backend": "llama.cpp",
            "llama_cli": self.llama_cli,
            "executable_available": (
                executable_exists
            ),
            "model_path": str(
                self.model_path
            ),
            "model_exists": (
                self.model_path.exists()
            ),
            "model_is_file": (
                self.model_path.is_file()
            ),
            "context": self.n_ctx,
            "threads": self.n_threads,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }


__all__ = [
    "LlamaCppEngine",
]
