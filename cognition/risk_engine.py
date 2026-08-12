"""
Motor de evaluación de riesgos.

No ejecuta acciones.
Solamente estima riesgo para que DecisionMaker
pueda decidir posteriormente.
"""

from typing import Iterable, List

from cognition.models import (
    Plan,
    PlanStep,
    RiskAssessment,
    RiskLevel,
)


class RiskEngine:

    def assess_step(
        self,
        step: PlanStep,
    ) -> RiskAssessment:

        probability = 0.20
        impact = 0.30
        reversibility = 0.90
        reasons: List[str] = []

        if step.risks:
            probability += min(
                0.35,
                len(step.risks) * 0.10
            )

            impact += min(
                0.30,
                len(step.risks) * 0.10
            )

            reasons.extend(step.risks)

        if not step.reversible:
            reversibility = 0.20
            impact += 0.20
            reasons.append(
                "La acción no es fácilmente reversible."
            )

        probability = min(1.0, probability)
        impact = min(1.0, impact)

        score = (
            probability
            * impact
            * (1.0 + (1.0 - reversibility))
        )

        score = min(1.0, score)

        level = self._level(score)

        return RiskAssessment(
            score=score,
            level=level,
            probability=probability,
            impact=impact,
            reversibility=reversibility,
            reasons=reasons,
        )

    def assess_plan(
        self,
        plan: Plan,
    ) -> RiskAssessment:

        assessments = [
            self.assess_step(step)
            for step in plan.steps
        ]

        if not assessments:
            return RiskAssessment(
                score=0.0,
                level=RiskLevel.LOW,
                probability=0.0,
                impact=0.0,
                reversibility=1.0,
            )

        worst = max(
            assessments,
            key=lambda item: item.score
        )

        plan.estimated_risk = worst.score

        return worst

    @staticmethod
    def _level(score: float) -> RiskLevel:
        if score >= 0.75:
            return RiskLevel.CRITICAL

        if score >= 0.50:
            return RiskLevel.HIGH

        if score >= 0.25:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW
