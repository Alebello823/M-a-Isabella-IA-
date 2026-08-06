"""
Sensor de hardware compatible con Android/Termux.
Usa comandos nativos del sistema en lugar de psutil.
"""
import os
import re
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional

@dataclass
class HardwareInfo:
    total_ram_gb: float
    available_ram_gb: float
    cpu_name: str
    cpu_cores: int
    has_avx2: bool
    gpu_name: str = "none"
    cpu_temp_c: float = 0.0
    battery_pct: float = 0.0


def _get_ram_info():
    """Obtiene información de RAM desde /proc/meminfo"""
    try:
        with open("/proc/meminfo") as f:
            meminfo = f.read()
        
        total = re.search(r"MemTotal:\s+(\d+)", meminfo)
        available = re.search(r"MemAvailable:\s+(\d+)", meminfo)
        
        total_kb = int(total.group(1)) if total else 0
        avail_kb = int(available.group(1)) if available else 0
        
        return total_kb / 1024, avail_kb / 1024  # Convertir KB a MB
    except:
        return 0, 0


def _get_cpu_info():
    """Obtiene información de CPU desde /proc/cpuinfo"""
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        
        # Nombre del procesador
        name_match = re.search(r"Hardware\s*:\s*(.+)", cpuinfo)
        if not name_match:
            name_match = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        cpu_name = name_match.group(1).strip() if name_match else platform.processor() or "ARM"
        
        # Contar núcleos
        cores = len(re.findall(r"processor\s*:", cpuinfo))
        
        # Detectar AVX2 (raro en ARM, pero lo verificamos)
        has_avx2 = "avx2" in cpuinfo.lower()
        
        return cpu_name, max(cores, 1), has_avx2
    except:
        return platform.processor() or "Unknown", 4, False


def _get_temperature():
    """Obtiene temperatura del CPU (Android/Termux)"""
    # Posibles rutas de temperatura en Android
    temp_paths = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
        "/sys/devices/virtual/thermal/thermal_zone0/temp",
    ]
    
    for path in temp_paths:
        try:
            with open(path) as f:
                temp = int(f.read().strip()) / 1000.0
                if 10 < temp < 100:  # Rango razonable
                    return round(temp, 1)
        except:
            continue
    
    return 0.0


def _get_battery():
    """Obtiene nivel de batería en Android"""
    try:
        # Usar dumpsys (requiere acceso)
        result = subprocess.run(
            ["dumpsys", "battery"],
            capture_output=True, text=True, timeout=5
        )
        match = re.search(r"level:\s*(\d+)", result.stdout)
        if match:
            return float(match.group(1))
    except:
        pass
    
    # Alternativa: leer archivo de batería
    try:
        for f in os.listdir("/sys/class/power_supply/"):
            if "battery" in f.lower() or "BAT" in f:
                with open(f"/sys/class/power_supply/{f}/capacity") as cap:
                    return float(cap.read().strip())
    except:
        pass
    
    return 0.0


def detect_hardware() -> HardwareInfo:
    """Detecta hardware real del dispositivo"""
    
    # RAM
    total_ram_mb, avail_ram_mb = _get_ram_info()
    total_ram_gb = round(total_ram_mb / 1024, 2)
    available_ram_gb = round(avail_ram_mb / 1024, 2)
    
    # CPU
    cpu_name, cpu_cores, has_avx2 = _get_cpu_info()
    
    # Temperatura
    cpu_temp = _get_temperature()
    
    # Batería
    battery = _get_battery()
    
    return HardwareInfo(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        has_avx2=has_avx2,
        cpu_temp_c=cpu_temp,
        battery_pct=battery,
    )


if __name__ == "__main__":
    info = detect_hardware()
    print(f"RAM: {info.available_ram_gb}GB / {info.total_ram_gb}GB")
    print(f"CPU: {info.cpu_name} ({info.cores} cores)")
    print(f"AVX2: {info.has_avx2}")
    print(f"Temp: {info.cpu_temp_c}°C")
    print(f"Batería: {info.battery_pct}%")
