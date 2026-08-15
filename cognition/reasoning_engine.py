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
        Detecta contradicciones explícitas y cuantitativas.

        Tipos soportados:

        1. Contradicción explícita:

            "X es verdadero"
            "no X es verdadero"

        2. Contradicción cuantitativa:

            "El teléfono tiene 2 GB libres."
            "El teléfono tiene 20 GB libres."

        La detección cuantitativa es conservadora:
        solamente considera contradicción cuando dos
        afirmaciones parecen describir la misma propiedad
        y contienen valores numéricos diferentes.

        No intenta realizar razonamiento semántico general.
        """

        normalized = [
            self._normalize(f)
            for f in facts
            if f and f.strip()
        ]

        contradictions: List[Tuple[str, str]] = []

        # --------------------------------------------------
        # 1. Contradicciones explícitas
        # --------------------------------------------------

        for fact in normalized:
            if fact.startswith("no "):
                opposite = fact[3:].strip()

                if opposite in normalized:
                    pair = (fact, opposite)

                    if pair not in contradictions:
                        contradictions.append(pair)

            else:
                opposite = "no " + fact

                if opposite in normalized:
                    pair = (fact, opposite)

                    if pair not in contradictions:
                        contradictions.append(pair)

        # --------------------------------------------------
        # 2. Contradicciones cuantitativas
        # --------------------------------------------------

        for index, first in enumerate(normalized):
            for second in normalized[index + 1:]:
                if self._quantitative_contradiction(
                    first,
                    second,
                ):
                    contradictions.append(
                        (first, second)
                    )

        return contradictions

    @staticmethod
    def _quantitative_contradiction(
        first: str,
        second: str,
    ) -> bool:
        """
        Detecta dos afirmaciones cuantitativas incompatibles
        sobre una misma propiedad.

        Ejemplo:

            "el teléfono tiene 2 gb libres"
            "el teléfono tiene 20 gb libres"

        Devuelve True.

        No intenta decidir causalidad.
        """

        pattern = re.compile(
            r"""
            (?P<value>\d+(?:[.,]\d+)?)
            \s*
            (?P<unit>
                gb|gib|mb|mib|
                kb|kib|
                tb|tib|
                %|por\s*ciento
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        first_match = pattern.search(first)
        second_match = pattern.search(second)

        if not first_match or not second_match:
            return False

        first_unit = (
            first_match.group("unit")
            .lower()
            .replace(" ", "")
        )

        second_unit = (
            second_match.group("unit")
            .lower()
            .replace(" ", "")
        )

        # Unidades incompatibles no permiten
        # afirmar contradicción automáticamente.
        if first_unit != second_unit:
            return False

        first_value = float(
            first_match.group("value").replace(",", ".")
        )

        second_value = float(
            second_match.group("value").replace(",", ".")
        )

        # Mismo valor → no hay contradicción.
        if first_value == second_value:
            return False

        # Eliminamos los valores para comparar
        # la estructura restante de las afirmaciones.
        first_structure = pattern.sub(
            "<VALUE>",
            first,
            count=1,
        )

        second_structure = pattern.sub(
            "<VALUE>",
            second,
            count=1,
        )

        # Normalizamos espacios.
        first_structure = re.sub(
            r"\s+",
            " ",
            first_structure,
        ).strip()

        second_structure = re.sub(
            r"\s+",
            " ",
            second_structure,
        ).strip()

        # Si las estructuras son idénticas,
        # representan la misma afirmación cuantitativa.
        if first_structure == second_structure:
            return True

        # --------------------------------------------------
        # Comparación estructural ligera
        # --------------------------------------------------

        first_words = {
            word
            for word in first_structure.split()
            if len(word) >= 4
        }

        second_words = {
            word
            for word in second_structure.split()
            if len(word) >= 4
        }

        if not first_words or not second_words:
            return False

        overlap = first_words & second_words

        similarity = (
            len(overlap)
            / max(
                len(first_words),
                len(second_words),
            )
        )

        return similarity >= 0.70

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
