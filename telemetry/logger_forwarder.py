# telemetry/logger_forwarder.py
import logging
from uuid import uuid4

class ContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = self.extra.copy()
        extra.update(kwargs.get('extra', {}))
        kwargs['extra'] = extra
        # Formateamos el mensaje con los campos del extra
        boot_id = extra.get("boot_id", "no-boot")
        conv_id = extra.get("conversation_id", "")
        tag = f"[boot={boot_id}]"
        if conv_id:
            tag += f"[conv={conv_id}]"
        return f"{tag} {msg}", kwargs

def setup_root_logger(boot_id: str):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(handler)
    # Retornamos un adaptador global para el módulo principal
    return ContextAdapter(logger, {"boot_id": boot_id})
