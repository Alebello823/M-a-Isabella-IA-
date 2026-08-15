"""
Motor de evaluación de riesgos del Cognitive Core.

Principios:

- determinista
- ligero
- explicable
- conservador
- sin LLM
- no ejecuta acciones

El motor diferencia entre:

1. acciones informativas
2. acciones reversibles
3. acciones modificadoras
4. acciones potencialmente destructivas

El objetivo es proporcionar una estimación operacional
para DecisionMaker.
"""

from typing import List

from cognition.models import (
    Plan,
    PlanStep,
    RiskAssessment,
    RiskLevel,
)


class RiskEngine:

    INFORMATION_KEYWORDS = (
        "medir",
        "observar",
        "identificar",
        "analizar",
        "recopilar",
        "diagnosticar",
        "verificar",
        "comprobar",
        "comparar",
        "inspeccionar",
    )

    MODIFICATION_KEYWORDS = (
        "cambiar",
        "modificar",
        "configurar",
        "instalar",
        "desinstalar",
        "actualizar",
        "reiniciar",
        "eliminar",
        "borrar",
        "crear",
        "mover",
    )

    DESTRUCTIVE_KEYWORDS = (
        "eliminar",
        "borrar",
        "formatear",
        "destruir",
        "sobrescribir",
        "resetear",
    )

    def assess_step(
        self,
        step: PlanStep,
    ) -> RiskAssessment:

        text = step.action.lower()

        reasons: List[str] = []

        # --------------------------------------------------
        # Riesgo base
        # --------------------------------------------------

        probability = 0.15
        impact = 0.20
        reversibility = 0.90

        # --------------------------------------------------
        # Acción informativa
        # --------------------------------------------------

        information_action = self._contains(
            text,
            self.INFORMATION_KEYWORDS,
        )

        if information_action:
            probability -= 0.05
            impact -= 0.05

            reasons.append(
                "La acción es principalmente informativa."
            )

        # --------------------------------------------------
        # Acción modificadora
        # --------------------------------------------------

        modification_action = self._contains(
            text,
            self.MODIFICATION_KEYWORDS,
        )

        if modification_action:
            probability += 0.15
            impact += 0.20

            reasons.append(
                "La acción puede modificar el sistema."
            )

        # --------------------------------------------------
        # Acción destructiva
        # --------------------------------------------------

        destructive_action = self._contains(
            text,
            self.DESTRUCTIVE_KEYWORDS,
        )

        if destructive_action:
            probability += 0.25
            impact += 0.30
            reversibility = min(
                reversibility,
                0.20,
            )

            reasons.append(
                "La acción puede producir consecuencias "
                "destructivas o difíciles de revertir."
            )

        # --------------------------------------------------
        # Riesgos explícitos del paso
        # --------------------------------------------------

        if step.risks:
            probability += min(
                0.30,
                len(step.risks) * 0.08,
            )

            impact += min(
                0.30,
                len(step.risks) * 0.08,
            )

            reasons.extend(step.risks)

        # --------------------------------------------------
        # Reversibilidad declarada
        # --------------------------------------------------

        if not step.reversible:
            reversibility = min(
                reversibility,
                0.20,
            )

            probability += 0.15
            impact += 0.20

            reasons.append(
                "La acción no está marcada como reversible."
            )

        # --------------------------------------------------
        # Límites
        # --------------------------------------------------

        probability = self._clamp(
            probability
        )

        impact = self._clamp(
            impact
        )

        reversibility = self._clamp(
            reversibility
        )

        # --------------------------------------------------
        # Riesgo compuesto
        # --------------------------------------------------

        score = (
            probability
            * impact
            * (
                1.0
                + (1.0 - reversibility)
            )
        )

        score = self._clamp(score)

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

        if not plan.steps:
            return RiskAssessment(
                score=0.0,
                level=RiskLevel.LOW,
                probability=0.0,
                impact=0.0,
                reversibility=1.0,
                reasons=[
                    "El plan no contiene acciones."
                ],
            )

        assessments = [
            self.assess_step(step)
            for step in plan.steps
        ]

        # El peor paso determina el riesgo operacional.
        worst = max(
            assessments,
            key=lambda item: item.score,
        )

        plan.estimated_risk = worst.score

        return worst

    @classmethod
    def _level(
        cls,
        score: float,
    ) -> RiskLevel:

        if score >= 0.75:
            return RiskLevel.CRITICAL

        if score >= 0.50:
            return RiskLevel.HIGH

        if score >= 0.25:
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    @staticmethod
    def _contains(
        text: str,
        keywords,
    ) -> bool:

        return any(
            keyword in text
            for keyword in keywords
        )

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(1.0, float(value)),
        )
