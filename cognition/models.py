"""
Modelos internos del Cognitive Core de Mía Isabella.

Este módulo NO depende de ningún LLM.
Representa problemas, evidencias, hipótesis, planes,
riesgos y decisiones de forma estructurada.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class EvidenceType(Enum):
    FACT = "fact"
    OBSERVATION = "observation"
    INFERENCE = "inference"
    ASSUMPTION = "assumption"
    EXTERNAL = "external"


class HypothesisStatus(Enum):
    OPEN = "open"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REJECTED = "rejected"
    CONFIRMED = "confirmed"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionStatus(Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_CONFIRMATION = "needs_confirmation"


@dataclass
class Evidence:
    content: str
    evidence_type: EvidenceType = EvidenceType.OBSERVATION
    confidence: float = 0.5
    source: Optional[str] = None
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class Problem:
    description: str
    objective: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    known_facts: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)


@dataclass
class Hypothesis:
    statement: str
    prior_probability: float = 0.5
    supporting_evidence: List[Evidence] = field(default_factory=list)
    opposing_evidence: List[Evidence] = field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.OPEN
    confidence: float = 0.5

    # Señales cognitivas adicionales.
    evidence_strength: float = 0.0
    causal_support: float = 0.0
    contradiction_penalty: float = 0.0
    verification_required: bool = True

    id: UUID = field(default_factory=uuid4)

    def __post_init__(self):
        self.prior_probability = max(
            0.0,
            min(1.0, float(self.prior_probability))
        )

        self.confidence = max(
            0.0,
            min(1.0, float(self.confidence))
        )

        self.evidence_strength = max(
            0.0,
            min(1.0, float(self.evidence_strength))
        )

        self.causal_support = max(
            0.0,
            min(1.0, float(self.causal_support))
        )

        self.contradiction_penalty = max(
            0.0,
            min(1.0, float(self.contradiction_penalty))
        )


@dataclass
class PlanStep:
    action: str
    expected_result: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    reversible: bool = True
    completed: bool = False
    id: UUID = field(default_factory=uuid4)


@dataclass
class Plan:
    objective: str
    steps: List[PlanStep] = field(default_factory=list)
    confidence: float = 0.5
    estimated_risk: float = 0.0
    id: UUID = field(default_factory=uuid4)


@dataclass
class RiskAssessment:
    score: float
    level: RiskLevel
    probability: float
    impact: float
    reversibility: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class Decision:
    action: str
    rationale: List[str] = field(default_factory=list)
    confidence: float = 0.5
    risk: Optional[RiskAssessment] = None
    status: DecisionStatus = DecisionStatus.PROPOSED
    alternatives: List[str] = field(default_factory=list)
    id: UUID = field(default_factory=uuid4)


@dataclass
class Reflection:
    success: bool
    observations: List[str] = field(default_factory=list)
    mistakes: List[str] = field(default_factory=list)
    lessons: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class CognitiveResult:
    problem: Problem
    hypotheses: List[Hypothesis] = field(default_factory=list)
    plan: Optional[Plan] = None
    decision: Optional[Decision] = None
    reflection: Optional[Reflection] = None
    confidence: float = 0.0
    uncertainty: float = 1.0
    needs_more_information: bool = False
    questions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
