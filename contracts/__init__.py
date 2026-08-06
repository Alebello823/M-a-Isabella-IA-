# contracts/memory.py
from typing import Protocol

class MemoryProtocol(Protocol):
    """Contrato para el sistema de memoria."""
    pass# 

# contracts/inference.py
from typing import Protocol

class InferenceProtocol(Protocol):
    """Contrato para el motor de inferencia."""
    pass

# contracts/storage.py
from typing import Protocol

class StorageProtocol(Protocol):
    """Contrato para el almacenamiento físico."""
    pass
