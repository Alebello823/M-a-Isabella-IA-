# bootstrap/boot_validator.py
import sys
import sqlite3
import shutil
from dataclasses import dataclass, field
from internal.constants import DATA_DIR

@dataclass
class ValidationResult:
    success: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

class BootValidator:
    MIN_PYTHON = (3, 10)

    def validate(self) -> ValidationResult:
        result = ValidationResult(success=True)
        self._check_python(result)
        self._check_sqlite(result)
        self._check_disk(result)
        # Autocompletar success
        result.success = len(result.errors) == 0
        return result

    def _check_python(self, result: ValidationResult):
        major, minor = sys.version_info[:2]
        result.diagnostics["python_version"] = f"{major}.{minor}"
        if (major, minor) < self.MIN_PYTHON:
            result.errors.append(f"Python {major}.{minor} < 3.10")
        else:
            result.warnings.append(f"Python {major}.{minor} (OK)")

    def _check_sqlite(self, result: ValidationResult):
        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("SELECT 1")
            conn.close()
            result.diagnostics["sqlite"] = "OK"
        except Exception as e:
            result.errors.append(f"SQLite no disponible: {e}")

    def _check_disk(self, result: ValidationResult):
        free_gb = shutil.disk_usage(DATA_DIR).free / (1024**3)
        result.diagnostics["disk_free_gb"] = round(free_gb, 2)
        if free_gb < 1.0:
            result.errors.append(f"Espacio en disco insuficiente ({free_gb:.1f} GB)")
        elif free_gb < 5.0:
            result.warnings.append(f"Espacio bajo: {free_gb:.1f} GB")
