from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar('T')

class DependencyContainer:
    def __init__(self):
        self._instances: Dict[Any, Any] = {}
        self._factories: Dict[Any, Callable[[], Any]] = {}

    def register_instance(self, key: Any, instance: Any) -> None:
        """Registra una instancia ya creada bajo una clave (string o tipo)."""
        self._instances[key] = instance

    def register_factory(self, key: Any, factory: Callable[[], Any]) -> None:
        """Registra una fábrica para crear la dependencia bajo demanda."""
        self._factories[key] = factory

    def resolve(self, key: Any) -> Any:
        """Resuelve una dependencia por clave. Si no existe, lanza KeyError."""
        if key in self._instances:
            return self._instances[key]
        if key in self._factories:
            instance = self._factories[key]()
            self._instances[key] = instance
            return instance
        raise KeyError(f"Dependencia no registrada: {key}")
