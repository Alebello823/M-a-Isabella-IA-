# bootstrap/configuration_loader.py
import yaml
import logging
from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from config.models import Settings
from internal.constants import DEFAULT_SETTINGS_PATH

logger = logging.getLogger(__name__)

class ConfigurationLoader:
    """
    Carga la configuración desde YAML y la valida con Pydantic.
    NO es singleton; se instancia en el Kernel.
    """
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DEFAULT_SETTINGS_PATH
        self._settings: Optional[Settings] = None

    def load(self) -> Settings:
        """Carga y valida la configuración. Retorna un objeto Settings."""
        if self._settings:
            return self._settings

        if not self.config_path.exists():
            logger.warning(f"No se encontró {self.config_path}, usando configuración por defecto.")
            self._settings = Settings()
            return self._settings

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self._settings = Settings(**data)
            logger.info(f"Configuración cargada desde {self.config_path}")
        except (yaml.YAMLError, ValidationError) as e:
            logger.error(f"Error al cargar configuración: {e}")
            self._settings = Settings()  # fallback seguro
        return self._settings

    @property
    def settings(self) -> Settings:
        if not self._settings:
            self.load()
        return self._settings
