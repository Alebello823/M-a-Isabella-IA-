"""
Motor de simulación ligera.

Permite evaluar consecuencias hipotéticas sin ejecutar acciones reales.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from cognition.models import Plan


@dataclass
class SimulationResult:
    success_probability: float
    expected_outcome: str
    failed_steps: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)


class SimulationEngine:

    def simulate(
        self,
        plan: Plan,
        initial_state: Dict[str, Any] = None,
    ) -> SimulationResult:

        state = dict(initial_state or {})
        failed_steps = []
        assumptions = []

        if not plan.steps:
            return SimulationResult(
                success_probability=0.0,
                expected_outcome="No existe un plan.",
                state=state,
            )

        observable_steps = 0
        blocked_steps = 0

        for step in plan.steps:

            if step.preconditions:
                assumptions.extend(
                    step.preconditions
                )

            # Una simulación no puede afirmar que
            # una acción real ocurrió.
            if not step.completed:
                blocked_steps += 1

                failed_steps.append(
                    step.action
                )

                state[
                    f"simulated:{step.action}"
                ] = "REQUIRES_OBSERVATION"

                continue

            observable_steps += 1

            state[
                f"simulated:{step.action}"
            ] = "OBSERVED_COMPLETE"

        probability = (
            observable_steps / len(plan.steps)
        )

        probability *= plan.confidence

        if blocked_steps:
            expected_outcome = (
                "El plan es estructuralmente viable, "
                "pero requiere observaciones reales antes "
                "de confirmar sus resultados."
            )
        else:
            expected_outcome = (
                "Los pasos observables del plan "
                "son compatibles con las condiciones conocidas."
            )

        return SimulationResult(
            success_probability=max(
                0.0,
                min(1.0, probability),
            ),
            expected_outcome=expected_outcome,
            failed_steps=failed_steps,
            assumptions=assumptions,
            state=state,
        )
