"""
Motor de razonamiento determinista.

No utiliza ningún modelo de lenguaje.

Responsabilidades:
- analizar evidencia
- detectar contradicciones
- aplicar reglas simples
- comparar hipótesis
- producir inferencias estructuradas
"""

import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from cognition.models import (
    Evidence,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
    Problem,
)

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Motor de razonamiento simbólico ligero.

    Diseñado para funcionar en dispositivos con pocos recursos.
    """

    def evaluate_hypothesis(
        self,
        hypothesis: Hypothesis,
    ) -> Hypothesis:
        """
        Evalúa una hipótesis de forma conservadora.

        Diferencia entre:
        - existencia de evidencia
        - fuerza de evidencia
        - soporte causal
        - contradicciones
        """

        support = self._evidence_weight(
            hypothesis.supporting_evidence
        )

        opposition = self._evidence_weight(
            hypothesis.opposing_evidence
        )

        total = support + opposition

        if total <= 0:
            hypothesis.confidence = (
                hypothesis.prior_probability * 0.60
            )
            hypothesis.evidence_strength = 0.0
            hypothesis.causal_support = 0.0
            hypothesis.verification_required = True
            hypothesis.status = HypothesisStatus.OPEN
            return hypothesis

        evidence_balance = support / total

        evidence_strength = min(
            1.0,
            total / 2.0,
        )

        # La coincidencia textual proporciona
        # solamente soporte causal débil.
        causal_support = (
            evidence_balance
            * evidence_strength
            * 0.35
        )

        confidence = (
            hypothesis.prior_probability * 0.30
            + evidence_balance * 0.30
            + evidence_strength * 0.20
            + causal_support * 0.20
        )

        confidence -= (
            hypothesis.contradiction_penalty * 0.25
        )

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        hypothesis.confidence = confidence
        hypothesis.evidence_strength = evidence_strength
        hypothesis.causal_support = causal_support

        hypothesis.verification_required = (
            causal_support < 0.70
            or evidence_strength < 0.70
        )

        if (
            confidence >= 0.90
            and causal_support >= 0.80
            and not hypothesis.verification_required
        ):
            hypothesis.status = HypothesisStatus.CONFIRMED

        elif confidence >= 0.65:
            hypothesis.status = HypothesisStatus.SUPPORTED

        elif confidence <= 0.20:
            hypothesis.status = HypothesisStatus.REJECTED

        elif confidence <= 0.40:
            hypothesis.status = HypothesisStatus.WEAKENED

        else:
            hypothesis.status = HypothesisStatus.OPEN

        return hypothesis

    def infer_from_rules(
        self,
        facts: Iterable[str],
        rules: Iterable[Tuple[str, str]],
    ) -> List[str]:
        """
        Aplica reglas simples:

            ("A", "B")

        significa:

            Si A entonces B.
        """

        known = {
            self._normalize(fact)
            for fact in facts
            if fact and fact.strip()
        }

        inferred: List[str] = []

        changed = True

        while changed:
            changed = False

            for premise, conclusion in rules:
                premise_n = self._normalize(premise)
                conclusion_n = self._normalize(conclusion)

                if premise_n in known and conclusion_n not in known:
                    known.add(conclusion_n)
                    inferred.append(conclusion)
                    changed = True

        return inferred

    def detect_contradictions(
        self,
        facts: Iterable[str],
    ) -> List[Tuple[str, str]]:
        """
        Busca contradicciones explícitas del tipo:

            "X es verdadero"
            "X no es verdadero"

        También detecta pares booleanos simples.
        """

        normalized = [
            self._normalize(f)
            for f in facts
            if f and f.strip()
        ]

        contradictions = []

        for fact in normalized:
            if fact.startswith("no "):
                opposite = fact[3:].strip()

                if opposite in normalized:
                    contradictions.append(
                        (fact, opposite)
                    )
            else:
                opposite = "no " + fact

                if opposite in normalized:
                    contradictions.append(
                        (fact, opposite)
                    )

        return contradictions

    def compare_hypotheses(
        self,
        hypotheses: List[Hypothesis],
    ) -> List[Hypothesis]:
        """Evalúa y ordena hipótesis por confianza."""

        evaluated = [
            self.evaluate_hypothesis(h)
            for h in hypotheses
        ]

        return sorted(
            evaluated,
            key=lambda h: h.confidence,
            reverse=True,
        )

    def analyze_problem(
        self,
        problem: Problem,
    ) -> Dict[str, object]:
        """
        Produce un diagnóstico estructurado del problema.
        """

        contradictions = self.detect_contradictions(
            problem.known_facts
        )

        missing_information = list(problem.unknowns)

        confidence = self._problem_confidence(problem)

        return {
            "problem_id": str(problem.id),
            "objective": problem.objective,
            "known_facts": list(problem.known_facts),
            "unknowns": missing_information,
            "constraints": list(problem.constraints),
            "contradictions": contradictions,
            "confidence": confidence,
            "needs_more_information": bool(
                missing_information or contradictions
            ),
        }

    @staticmethod
    def _evidence_weight(
        evidence: Iterable[Evidence],
    ) -> float:
        weight = 0.0

        for item in evidence:
            factor = 1.0

            if item.evidence_type == EvidenceType.FACT:
                factor = 1.20
            elif item.evidence_type == EvidenceType.OBSERVATION:
                factor = 1.00
            elif item.evidence_type == EvidenceType.EXTERNAL:
                factor = 1.10
            elif item.evidence_type == EvidenceType.INFERENCE:
                factor = 0.80
            elif item.evidence_type == EvidenceType.ASSUMPTION:
                factor = 0.35

            weight += item.confidence * factor

        return weight

    @staticmethod
    def _problem_confidence(problem: Problem) -> float:
        known = len(problem.known_facts)
        unknown = len(problem.unknowns)
        assumptions = len(problem.assumptions)

        denominator = known + unknown + assumptions

        if denominator == 0:
            return 0.25

        score = known / denominator

        score -= assumptions * 0.05

        return max(0.0, min(1.0, score))

    @staticmethod
    def _normalize(value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"\s+", " ", value)
        return value
