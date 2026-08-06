"""
Módulo de detección y adaptación al hardware.
"""
from .hardware_sensor import detect_hardware, HardwareInfo
from .capability_mapper import CapabilityMapper

__all__ = ["detect_hardware", "HardwareInfo", "CapabilityMapper"]
