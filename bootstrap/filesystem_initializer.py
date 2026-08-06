# bootstrap/filesystem_initializer.py
import os
import logging
from pathlib import Path
from typing import List, Tuple

from internal.constants import PROJECT_ROOT, DATA_DIR, LOGS_DIR, MODELS_DIR
from config.models import Settings

logger = logging.getLogger(__name__)

class FileSystemInitializer:
    """
    Crea los directorios necesarios y verifica permisos.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.required_dirs = [
            DATA_DIR,
            LOGS_DIR,
            MODELS_DIR,
            Path(self.settings.storage.vector_store_path).parent,
        ]
        # añadir directorio de modelos
        models_path = PROJECT_ROOT / "models"
        self.required_dirs.append(models_path)

    def initialize(self) -> bool:
        """
        Crea directorios, verifica permisos de escritura y espacio en disco.
        Retorna True si todo es correcto, False en caso de error crítico.
        """
        logger.info("Inicializando sistema de archivos...")

        for dir_path in self.required_dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                # Verificar permisos de escritura
                test_file = dir_path / ".write_test"
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                logger.error(f"No se pudo crear o escribir en {dir_path}: {e}")
                return False

        # Verificar espacio en disco (mínimo 500 MB)
        try:
            statvfs = os.statvfs(PROJECT_ROOT)
            free_space = statvfs.f_frsize * statvfs.f_bavail
            if free_space < 500 * 1024 * 1024:
                logger.warning(f"Espacio libre bajo: {free_space // (1024*1024)} MB, mínimo recomendado 500 MB")
        except Exception as e:
            logger.warning(f"No se pudo verificar espacio en disco: {e}")

        logger.info("Sistema de archivos inicializado correctamente.")
        return True
