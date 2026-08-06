"""
Carga perfiles YAML y selecciona el más adecuado según el hardware real.
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from .hardware_sensor import HardwareInfo

logger = logging.getLogger(__name__)

class CapabilityMapper:
    def __init__(self, profiles_dir: Path):
        self.profiles = self._load_profiles(profiles_dir)

    def _load_profiles(self, profiles_dir: Path) -> list:
        profiles = []
        for filepath in sorted(profiles_dir.glob("tier_*.yaml")):
            with open(filepath) as fh:
                profile = yaml.safe_load(fh)
                profiles.append(profile)
        # Ordenar por RAM mínima ascendente
        return sorted(profiles, key=lambda p: p.get("min_ram_gb", 0))

    def select_profile(
        self, hw: HardwareInfo, manual_tier: Optional[str] = None
    ) -> Dict[str, Any]:
        if manual_tier:
            for p in self.profiles:
                if p["tier"] == manual_tier:
                    logger.info("Perfil forzado manualmente: %s", manual_tier)
                    return p
            raise ValueError(f"Perfil manual '{manual_tier}' no encontrado")

        selected = None
        for p in self.profiles:
            if p["min_ram_gb"] <= hw.available_ram_gb:
                selected = p
        if selected is None:
            selected = self.profiles[0]  # el más ligero
            logger.warning("RAM insuficiente para todos los perfiles, usando el más bajo")
        logger.info("Perfil automático seleccionado: %s", selected["tier"])
        return selected
