"""
Motor de incertidumbre.

Su función principal es impedir que Isabella confunda:
- hecho
- inferencia
- suposición
- desconocimiento
"""

from typing import Iterable, List

from cognition.models import Evidence, EvidenceType, Problem


class UncertaintyEngine:

    def calculate(
        self,
        problem: Problem,
        evidence: Iterable[Evidence] = (),
    ) -> float:
        evidence = list(evidence)

        if not problem.known_facts and not evidence:
            return 1.0

        known = len(problem.known_facts)
        unknown = len(problem.unknowns)
        assumptions = len(problem.assumptions)

        evidence_quality = 0.0

        if evidence:
            total = sum(
                item.confidence
                for item in evidence
            )

            evidence_quality = total / len(evidence)

        information_total = (
            known
            + unknown
            + assumptions
        )

        if information_total:
            structural_uncertainty = (
                unknown + assumptions
            ) / information_total
        else:
            structural_uncertainty = 1.0

        uncertainty = (
            structural_uncertainty * 0.65
            + (1.0 - evidence_quality) * 0.35
        )

        return max(
            0.0,
            min(1.0, uncertainty)
        )

    def identify_unknowns(
        self,
        problem: Problem,
    ) -> List[str]:
        return list(problem.unknowns)

    def needs_clarification(
        self,
        problem: Problem,
        threshold: float = 0.65,
    ) -> bool:
        uncertainty = self.calculate(problem)

        return uncertainty >= threshold
