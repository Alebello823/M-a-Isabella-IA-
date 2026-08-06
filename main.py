import sys
import logging
from pathlib import Path
from uuid import uuid4

from internal.constants import LOGS_DIR  # Importamos LOGS_DIR
from core.kernel import Kernel
from telemetry.logger_forwarder import setup_root_logger

def setup_logging(boot_id: str):
    """Configura el logging raíz con archivo y consola, e inyecta boot_id."""
    # Creamos el directorio de logs si no existe
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Manejador de archivo
    file_handler = logging.FileHandler(LOGS_DIR / "mia_isabella.log", encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    
    # Manejador de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Limpiar manejadores anteriores por si acaso
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Retornar un adaptador que incluya boot_id en los mensajes
    adapter = logging.LoggerAdapter(root_logger, {"boot_id": boot_id})
    return adapter

def main():
    boot_id = uuid4().hex[:8]
    logger = setup_logging(boot_id)
    logger.info(f"Iniciando Mía Isabella (boot={boot_id})")

    kernel = Kernel(boot_id=boot_id)
    if not kernel.boot():
        logger.error("El sistema no pudo arrancar. Revisa los logs.")
        sys.exit(1)

    # Iniciamos la interfaz CLI
    from interfaces.cli.terminal import TerminalChat
    orchestrator = kernel.container.resolve("Orchestrator")
    chat = TerminalChat(orchestrator)
    chat.run()

if __name__ == "__main__":
    main()
