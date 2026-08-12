"""
Motor de decisión ligero.

Selecciona acciones utilizando:
- confianza
- riesgo
- incertidumbre
- reversibilidad
- valor informativo

No ejecuta ninguna acción real.
"""

from typing import Iterable, List

from cognition.models import (
    Decision,
    DecisionStatus,
    Hypothesis,
    Plan,
    RiskAssessment,
    RiskLevel,
)


class DecisionMaker:

    def choose(
        self,
        plan: Plan,
        risk: RiskAssessment,
        hypotheses: Iterable[Hypothesis] = (),
        uncertainty: float = 0.5,
    ) -> Decision:

        hypotheses = list(hypotheses)

        rationale: List[str] = []

        rationale.append(
            f"Confianza del plan: {plan.confidence:.2f}"
        )

        rationale.append(
            f"Riesgo estimado: {risk.score:.2f}"
        )

        rationale.append(
            f"Incertidumbre: {uncertainty:.2f}"
        )

        if hypotheses:
            best = max(
                hypotheses,
                key=lambda h: h.confidence,
            )

            rationale.append(
                "Hipótesis dominante: "
                f"{best.statement}"
            )

        selected_step = self._select_action(
            plan,
            uncertainty,
        )

        if selected_step is None:
            action = "Solicitar más información."
        else:
            action = selected_step.action

        if risk.level == RiskLevel.CRITICAL:
            status = DecisionStatus.REJECTED

        elif risk.level == RiskLevel.HIGH:
            status = DecisionStatus.NEEDS_CONFIRMATION

        elif uncertainty >= 0.65:
            status = DecisionStatus.NEEDS_CONFIRMATION

        elif plan.confidence < 0.35:
            status = DecisionStatus.NEEDS_CONFIRMATION

        else:
            status = DecisionStatus.ACCEPTED

        confidence = self._decision_confidence(
            plan.confidence,
            risk.score,
            uncertainty,
            selected_step.reversible
            if selected_step is not None
            else True,
        )

        if uncertainty >= 0.65:
            rationale.append(
                "La incertidumbre es suficientemente alta "
                "para requerir confirmación."
            )

        if selected_step is not None:
            if selected_step.reversible:
                rationale.append(
                    "La acción seleccionada es reversible."
                )

            if self._is_information_action(
                selected_step.action
            ):
                rationale.append(
                    "La acción seleccionada puede reducir "
                    "la incertidumbre antes de modificar el sistema."
                )

        return Decision(
            action=action,
            rationale=rationale,
            confidence=confidence,
            risk=risk,
            status=status,
            alternatives=[
                step.action
                for step in plan.steps
                if selected_step is None
                or step.id != selected_step.id
            ],
        )

    @classmethod
    def _select_action(
        cls,
        plan: Plan,
        uncertainty: float,
    ):
        if not plan.steps:
            return None

        scored = []

        for index, step in enumerate(plan.steps):
            score = 0.0
            text = step.action.lower()

            # Reducir incertidumbre tiene prioridad.
            if cls._is_information_action(text):
                score += 0.40

            # Verificación también es valiosa.
            if any(
                word in text
                for word in (
                    "verificar",
                    "medir",
                    "comprobar",
                    "identificar",
                    "analizar",
                )
            ):
                score += 0.20

            # Acciones reversibles son preferibles.
            if step.reversible:
                score += 0.15

            # Si la incertidumbre es alta, priorizamos diagnóstico.
            if uncertainty >= 0.50 and cls._is_information_action(
                text
            ):
                score += 0.20

            # Penalización leve por posición para evitar
            # que el primer paso gane siempre.
            score -= index * 0.01

            scored.append((score, step))

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return scored[0][1]

    @staticmethod
    def _is_information_action(
        text: str,
    ) -> bool:
        text = text.lower()

        keywords = (
            "información",
            "informacion",
            "identificar",
            "analizar",
            "medir",
            "diagnosticar",
            "verificar",
            "comprobar",
            "evidencia",
        )

        return any(
            keyword in text
            for keyword in keywords
        )

    @staticmethod
    def _decision_confidence(
        plan_confidence: float,
        risk: float,
        uncertainty: float,
        reversible: bool,
    ) -> float:

        value = (
            plan_confidence * 0.40
            + (1.0 - risk) * 0.25
            + (1.0 - uncertainty) * 0.25
            + (0.10 if reversible else 0.0)
        )

        return max(
            0.0,
            min(1.0, value),
        )
