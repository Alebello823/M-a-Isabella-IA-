"""
Motor de incertidumbre del Cognitive Core.

Distingue entre:

- información conocida
- información desconocida
- supuestos
- calidad de evidencia
- contradicciones

No utiliza LLM.
"""

from typing import Iterable, List

from cognition.models import (
    Evidence,
    EvidenceType,
    Problem,
)


class UncertaintyEngine:

    def calculate(
        self,
        problem: Problem,
        evidence: Iterable[Evidence] = (),
    ) -> float:

        evidence = list(evidence)

        # --------------------------------------------------
        # Caso extremo: prácticamente no sabemos nada.
        # --------------------------------------------------

        if not problem.known_facts and not evidence:

            return 1.0

        known = len(
            problem.known_facts
        )

        unknown = len(
            problem.unknowns
        )

        assumptions = len(
            problem.assumptions
        )

        # --------------------------------------------------
        # Calidad de evidencia.
        # --------------------------------------------------

        evidence_quality = (
            self._evidence_quality(
                evidence
            )
        )

        # --------------------------------------------------
        # Incertidumbre estructural.
        # --------------------------------------------------

        total = (
            known
            + unknown
            + assumptions
        )

        if total > 0:

            structural = (
                unknown
                + assumptions
            ) / total

        else:

            structural = 1.0

        # --------------------------------------------------
        # Contradicciones.
        # --------------------------------------------------

        contradictions = self._count_contradictions(
            problem.known_facts
        )

        contradiction_penalty = min(
            0.25,
            contradictions * 0.10,
        )

        # --------------------------------------------------
        # Resultado.
        # --------------------------------------------------

        uncertainty = (
            structural * 0.55
            + (1.0 - evidence_quality) * 0.30
            + contradiction_penalty
        )

        return self._clamp(
            uncertainty
        )

    def identify_unknowns(
        self,
        problem: Problem,
    ) -> List[str]:

        return list(
            problem.unknowns
        )

    def needs_clarification(
        self,
        problem: Problem,
        threshold: float = 0.65,
    ) -> bool:

        uncertainty = self.calculate(
            problem
        )

        return uncertainty >= threshold

    @staticmethod
    def _evidence_quality(
        evidence: Iterable[Evidence],
    ) -> float:

        evidence = list(
            evidence
        )

        if not evidence:
            return 0.0

        factors = {
            EvidenceType.FACT: 1.00,
            EvidenceType.EXTERNAL: 0.90,
            EvidenceType.OBSERVATION: 0.80,
            EvidenceType.INFERENCE: 0.55,
            EvidenceType.ASSUMPTION: 0.25,
        }

        total = 0.0
        weight = 0.0

        for item in evidence:

            factor = factors.get(
                item.evidence_type,
                0.50,
            )

            total += (
                item.confidence
                * factor
            )

            weight += factor

        if weight <= 0:
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                total / weight,
            ),
        )

    @staticmethod
    def _count_contradictions(
        facts: Iterable[str],
    ) -> int:

        normalized = {
            str(fact).strip().lower()
            for fact in facts
            if fact and fact.strip()
        }

        count = 0

        for fact in normalized:

            if fact.startswith("no "):

                opposite = fact[3:].strip()

                if opposite in normalized:
                    count += 1

            else:

                opposite = (
                    "no " + fact
                )

                if opposite in normalized:
                    count += 1

        return count // 2

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )
