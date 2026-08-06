# contracts/memory.py
"""
Interfaz (Protocol) para el sistema de memoria.
Cualquier componente que quiera almacenar/recuperar información debe implementar este contrato.
"""

from typing import Protocol, List, Optional, Dict, Any, Union
from internal.schemas import (
    Episode, Fact, Goal, Message, ConversationContext,
    MemoryType, EmotionTone, ConfidenceLevel
)

class MemoryProtocol(Protocol):
    """
    Contrato que debe cumplir el módulo de memoria.
    No importa si es SQLite, vector DB o una combinación.
    """

    # ---------- Operaciones episódicas ----------
    def save_episode(self, episode: Episode) -> str:
        """Guarda un episodio completo (conversación o evento). Retorna el ID."""
        ...

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Recupera un episodio por su ID."""
        ...

    def get_recent_episodes(self, limit: int = 10, since: Optional[str] = None) -> List[Episode]:
        """Obtiene los episodios más recientes, opcionalmente desde una fecha."""
        ...

    def search_episodes_by_content(self, query: str, limit: int = 5) -> List[Episode]:
        """Búsqueda semántica o por palabras clave en el contenido de los episodios."""
        ...

    # ---------- Operaciones semánticas (hechos) ----------
    def save_fact(self, fact: Fact) -> str:
        """Guarda un hecho (conocimiento atómico). Retorna el ID."""
        ...

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Recupera un hecho por ID."""
        ...

    def get_facts_about(self, subject: str, predicate: Optional[str] = None) -> List[Fact]:
        """Obtiene todos los hechos que involucran a un sujeto, opcionalmente filtrados por predicado."""
        ...

    def search_facts(self, query: str, limit: int = 10) -> List[Fact]:
        """Búsqueda semántica de hechos."""
        ...

    # ---------- Operaciones de grafo (relaciones) ----------
    def add_relation(self, subject: str, relation: str, object: str, metadata: Optional[Dict] = None) -> None:
        """Añade una relación entre dos entidades en el grafo de conocimiento."""
        ...

    def get_related(self, entity: str, relation: Optional[str] = None, depth: int = 1) -> List[Dict]:
        """Obtiene entidades relacionadas con una entidad dada, opcionalmente por relación y profundidad."""
        ...

    # ---------- Operaciones de contexto y memoria de trabajo ----------
    def save_conversation_context(self, context: ConversationContext) -> None:
        """Guarda el contexto de una conversación activa (memoria de trabajo)."""
        ...

    def get_conversation_context(self, thread_id: str) -> Optional[ConversationContext]:
        """Recupera el contexto de un hilo de conversación."""
        ...

    def clear_conversation_context(self, thread_id: str) -> None:
        """Limpia la memoria de trabajo de un hilo (cuando termina)."""
        ...

    # ---------- Curación y olvido ----------
    def mark_for_review(self, item_id: str, item_type: MemoryType) -> None:
        """Marca un ítem para que el curador lo revise (posible olvido o resumen)."""
        ...

    def get_items_for_curation(self, item_type: Optional[MemoryType] = None, limit: int = 20) -> List[Dict]:
        """Obtiene ítems pendientes de curación."""
        ...

    # ---------- Sincronización (para multiplataforma) ----------
    def get_changes_since(self, timestamp: str, device_id: Optional[str] = None) -> Dict[str, List]:
        """Obtiene todos los cambios (episodios, hechos, etc.) desde una fecha dada, para sincronizar."""
        ...

    def apply_sync_delta(self, delta: Dict[str, List], device_id: str) -> None:
        """Aplica un delta de sincronización proveniente de otro dispositivo."""
        ...

    # ---------- Utilidades ----------
    def get_stats(self) -> Dict[str, Any]:
        """Devuelve estadísticas de uso (número de episodios, hechos, etc.)."""
        ...
