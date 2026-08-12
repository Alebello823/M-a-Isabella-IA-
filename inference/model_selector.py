"""
Selector de modelo local de Mía Isabella.

No contiene nombres de modelos obligatorios.
Selecciona un GGUF disponible según configuración
y existencia física.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class ModelSelector:
    """Selecciona un modelo GGUF local disponible."""

    def __init__(
        self,
        model_directory: str | Path,
    ):
        self.model_directory = Path(
            model_directory
        ).expanduser()

    def discover(self) -> list[Path]:
        """Devuelve todos los modelos GGUF encontrados."""

        if not self.model_directory.exists():
            return []

        return sorted(
            path
            for path in self.model_directory.glob("*.gguf")
            if path.is_file()
            and path.stat().st_size > 0
        )

    def select(
        self,
        preferred: str | None = None,
    ) -> Path:
        """
        Selecciona un modelo.

        Si preferred existe, tiene prioridad.
        Si no, utiliza el primer GGUF válido encontrado.
        """

        models = self.discover()

        if not models:
            raise FileNotFoundError(
                f"No se encontraron modelos GGUF en "
                f"{self.model_directory}"
            )

        if preferred:
            preferred_path = (
                self.model_directory / preferred
            )

            if (
                preferred_path.exists()
                and preferred_path.stat().st_size > 0
            ):
                return preferred_path

        return models[0]
