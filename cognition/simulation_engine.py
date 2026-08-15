"""
Motor de simulación ligera del Cognitive Core.

La simulación NO ejecuta acciones.

Distingue entre:

- resultado observado
- acción no ejecutada
- acción bloqueada

Por diseño, una acción no ejecutada NO se considera
un fracaso real.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from cognition.models import Plan


@dataclass
class SimulationResult:

    success_probability: float

    expected_outcome: str

    failed_steps: List[str] = field(
        default_factory=list
    )

    assumptions: List[str] = field(
        default_factory=list
    )

    state: Dict[str, Any] = field(
        default_factory=dict
    )


class SimulationEngine:

    def simulate(
        self,
        plan: Plan,
        initial_state: Dict[str, Any] = None,
    ) -> SimulationResult:

        state = dict(
            initial_state or {}
        )

        failed_steps: List[str] = []
        assumptions: List[str] = []

        if not plan.steps:

            return SimulationResult(
                success_probability=0.0,
                expected_outcome=(
                    "No existe un plan que simular."
                ),
                state=state,
            )

        observed_complete = 0
        observed_failed = 0
        pending = 0

        for step in plan.steps:

            if step.preconditions:
                assumptions.extend(
                    step.preconditions
                )

            # ----------------------------------------------
            # Paso observado como completado
            # ----------------------------------------------

            if step.completed:

                observed_complete += 1

                state[
                    f"simulated:{step.action}"
                ] = "OBSERVED_COMPLETE"

                continue

            # ----------------------------------------------
            # Paso todavía no ejecutado
            # ----------------------------------------------

            pending += 1

            state[
                f"simulated:{step.action}"
            ] = "NOT_EXECUTED"

        # --------------------------------------------------
        # Solamente los resultados observados influyen
        # en la evidencia de éxito.
        # --------------------------------------------------

        observed_total = (
            observed_complete
            + observed_failed
        )

        if observed_total > 0:

            observed_success = (
                observed_complete
                / observed_total
            )

            probability = (
                observed_success
                * plan.confidence
            )

        else:

            # No existe evidencia de ejecución.
            # La simulación no inventa éxito ni fracaso.
            probability = (
                plan.confidence
                * 0.50
            )

        probability = max(
            0.0,
            min(
                1.0,
                probability,
            ),
        )

        # --------------------------------------------------
        # Resultado esperado
        # --------------------------------------------------

        if observed_failed:

            expected_outcome = (
                "Existen pasos observados con resultado "
                "negativo. Se requiere análisis antes "
                "de continuar."
            )

        elif pending:

            expected_outcome = (
                "El plan es estructuralmente viable, "
                "pero todavía requiere observaciones reales "
                "para validar sus resultados."
            )

        else:

            expected_outcome = (
                "Todos los pasos disponibles presentan "
                "resultados observados."
            )

        return SimulationResult(
            success_probability=probability,
            expected_outcome=expected_outcome,
            failed_steps=failed_steps,
            assumptions=assumptions,
            state=state,
        )
