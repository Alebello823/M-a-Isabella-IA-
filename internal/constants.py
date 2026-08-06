# internal/constants.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Directorios (se crearán en bootstrap, no durante import)
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Archivos de configuración
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.yaml"
DEFAULT_DB_PATH = DATA_DIR / "mia_isabella.db"
