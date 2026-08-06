# internal/exceptions.py
class DependencyNotRegisteredError(KeyError):
    """Se lanza cuando una dependencia no está registrada en el contenedor."""
    pass

class CircularDependencyError(Exception):
    """Se lanza si se detecta dependencia circular."""
    pass
