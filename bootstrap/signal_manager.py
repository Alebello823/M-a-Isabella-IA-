# bootstrap/signal_manager.py
import signal
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class SignalManager:
    """
    Maneja señales del sistema operativo para un apagado ordenado.
    """
    def __init__(self, shutdown_callback: Callable[[], None]):
        self.shutdown_callback = shutdown_callback
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame) -> None:
        logger.info(f"Señal {signum} recibida. Iniciando apagado...")
        if self.shutdown_callback:
            self.shutdown_callback()
        import sys
        sys.exit(0)
