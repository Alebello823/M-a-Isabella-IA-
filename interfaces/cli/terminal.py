# interfaces/cli/terminal.py
import logging
from datetime import datetime

from internal.schemas import Message, MessageRole
from core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

class TerminalChat:
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator

    def run(self):
        print("Mía Isabella - Chat CLI (escribe 'salir' para terminar)\n")
        while True:
            try:
                user_input = input("Tú: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("salir", "exit", "quit"):
                    print("¡Hasta luego!")
                    break

                # Crear mensaje de usuario
                user_msg = Message(
                    role=MessageRole.USER,
                    content=user_input,
                    timestamp=datetime.now()
                )

                # Procesar con orquestador
                print("Mía está pensando...", end="\r")
                assistant_msg = self.orchestrator.process_message(user_msg)
                print(f"Mía: {assistant_msg.content}\n")

            except KeyboardInterrupt:
                print("\nSaliendo...")
                break
            except Exception as e:
                logger.error(f"Error en CLI: {e}", exc_info=True)
                print(f"Error: {e}")
