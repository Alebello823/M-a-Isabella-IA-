# internal/schemas.py
"""
Modelos de datos compartidos (DTOs) para todo el sistema cognitivo.
Estas estructuras son la "moneda común" entre módulos.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Union
from uuid import UUID, uuid4

# ------------------ Enums básicos ------------------
class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class MemoryType(Enum):
    EPISODIC = "episodic"       # eventos, conversaciones
    SEMANTIC = "semantic"       # hechos, conocimientos
    PROCEDURAL = "procedural"   # habilidades, procedimientos
    EMOTIONAL = "emotional"     # estados afectivos
    GRAPH = "graph"             # relaciones entre entidades

class EmotionTone(Enum):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ANXIOUS = "anxious"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"

class ConfidenceLevel(Enum):
    HIGH = "high"       # > 80%
    MEDIUM = "medium"   # 50-80%
    LOW = "low"         # < 50%
    UNKNOWN = "unknown"

# ------------------ Mensajes ------------------
@dataclass
class Message:
    """Un mensaje individual en una conversación."""
    role: MessageRole                          # Obligatorio
    content: str                               # Obligatorio
    id: UUID = field(default_factory=uuid4)    # Opcional (con default)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

# ------------------ Episodios (Memoria episódica) ------------------
@dataclass
class Episode:
    """Un episodio es una unidad de experiencia: una conversación o un evento significativo."""
    id: UUID = field(default_factory=uuid4)
    title: Optional[str] = None                # resumen corto
    messages: List[Message] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    summary: Optional[str] = None              # resumen generado por el curador
    importance_score: float = 0.5              # 0-1, asignado por el curador
    tags: List[str] = field(default_factory=list)
    device_id: Optional[str] = None            # dónde ocurrió
    emotion_tone: EmotionTone = EmotionTone.NEUTRAL
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

# ------------------ Hechos (Memoria semántica) ------------------
@dataclass
class Fact:
    subject: str
    predicate: str
    object: Union[str, int, float, bool]

    id: UUID = field(default_factory=uuid4)
    confidence: float = 1.0
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    last_reviewed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    relation_type: Optional[str] = None

# ------------------ Objetivos ------------------
class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    BLOCKED = "blocked"

@dataclass
class Goal:
    description: str

    id: UUID = field(default_factory=uuid4)
    status: GoalStatus = GoalStatus.ACTIVE
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    sub_goals: List['Goal'] = field(default_factory=list)
    progress: float = 0.0
    notes: List[str] = field(default_factory=list)

# ------------------ Contexto ------------------
@dataclass
class ConversationContext:
    """El estado completo de una conversación en un momento dado."""
    thread_id: UUID
    messages: List[Message] = field(default_factory=list)
    current_topic: Optional[str] = None
    active_goal: Optional[Goal] = None
    user_intent: Optional[str] = None
    emotion_tone: EmotionTone = EmotionTone.NEUTRAL
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    last_interaction: datetime = field(default_factory=datetime.now)

# ------------------ Comandos y Eventos (para buses) ------------------
@dataclass
class Command:
    action: str

    id: UUID = field(default_factory=uuid4)
    parameters: Dict[str, Any] = field(default_factory=dict)
    issued_by: str = "user"
    priority: int = 1

@dataclass
class Event:
    name: str

    id: UUID = field(default_factory=uuid4)
    payload: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.now)
    source: str = "unknown"

# ------------------ Identity ------------------
@dataclass
class UserProfile:
    """Perfil del usuario."""
    id: UUID = field(default_factory=uuid4)
    name: Optional[str] = None
    preferred_language: str = "es"
    relationship_to_ai: str = "user"
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None

@dataclass
class AIPersonality:
    """Personalidad de la IA (no depende del LLM)."""
    name: str = "Mía Isabella"
    avatar_id: Optional[str] = None
    voice_profile: Dict[str, Any] = field(default_factory=dict)
    style: str = "cariñosa pero profesional"
    values: List[str] = field(default_factory=lambda: ["honestidad", "empatía", "privacidad"])
    communication_rules: Dict[str, Any] = field(default_factory=dict)
