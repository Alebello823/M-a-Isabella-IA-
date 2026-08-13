"""
Motor de planificación determinista.

Construye planes pequeños, verificables y reversibles.

No ejecuta acciones.
No utiliza LLM.
"""

import re
from typing import List, Optional

from cognition.models import (
    Hypothesis,
    Plan,
    PlanStep,
    Problem,
)


class PlanningEngine:

    def create_plan(
        self,
        problem: Problem,
        hypotheses: Optional[List[Hypothesis]] = None,
    ) -> Plan:

        objective = (
            problem.objective
            or problem.description
        )

        text = self._problem_text(problem)

        # --------------------------------------------------
        # Plan especializado: rendimiento
        # --------------------------------------------------

        if self._is_performance_problem(text):
            return self._performance_plan(
                problem,
                objective,
            )

        # --------------------------------------------------
        # Plan especializado: información insuficiente
        # --------------------------------------------------

        if problem.unknowns:
            return self._investigation_plan(
                problem,
                objective,
            )

        # --------------------------------------------------
        # Fallback general
        # --------------------------------------------------

        return self._generic_plan(
            problem,
            objective,
        )

    def prioritize_steps(
        self,
        plan: Plan,
    ) -> Plan:
        """
        Prioriza acciones que:

        1. reducen incertidumbre
        2. son reversibles
        3. permiten verificación
        """

        def priority(step: PlanStep) -> tuple:
            text = step.action.lower()

            information = any(
                word in text
                for word in (
                    "medir",
                    "identificar",
                    "observar",
                    "recopilar",
                    "información",
                    "diagnosticar",
                )
            )

            verification = any(
                word in text
                for word in (
                    "verificar",
                    "comparar",
                    "comprobar",
                )
            )

            return (
                0 if information else 1,
                0 if step.reversible else 1,
                0 if verification else 1,
            )

        plan.steps.sort(key=priority)

        return plan

    def _performance_plan(
        self,
        problem: Problem,
        objective: str,
    ) -> Plan:

        steps = [
            PlanStep(
                action="Medir el consumo actual de CPU y RAM.",
                expected_result=(
                    "Identificación del recurso que presenta "
                    "mayor presión durante la inferencia."
                ),
                reversible=True,
            ),
            PlanStep(
                action=(
                    "Medir el efecto del tamaño de contexto "
                    "sobre el rendimiento."
                ),
                expected_result=(
                    "Comparación del rendimiento con diferentes "
                    "niveles de contexto."
                ),
                reversible=True,
            ),
            PlanStep(
                action=(
                    "Comparar los resultados con la configuración "
                    "actual de llama.cpp."
                ),
                expected_result=(
                    "Determinación de si la configuración actual "
                    "contribuye a la lentitud."
                ),
                reversible=True,
            ),
            PlanStep(
                action=(
                    "Identificar la causa más probable "
                    "sin cambiar todavía el modelo."
                ),
                expected_result=(
                    "Una causa priorizada y respaldada "
                    "por mediciones."
                ),
                reversible=True,
            ),
            PlanStep(
                action=(
                    "Proponer el ajuste mínimo y reversible "
                    "que pueda mejorar el rendimiento."
                ),
                expected_result=(
                    "Una acción concreta que pueda probarse "
                    "sin comprometer el sistema."
                ),
                reversible=True,
            ),
            PlanStep(
                action=(
                    "Verificar el rendimiento después del ajuste."
                ),
                expected_result=(
                    "Confirmación o rechazo de la hipótesis "
                    "mediante una nueva medición."
                ),
                reversible=True,
            ),
        ]

        known = len(problem.known_facts)
        unknown = len(problem.unknowns)
        evidence = len(problem.evidence)

        confidence = 0.45

        confidence += min(
            0.20,
            known * 0.05,
        )

        confidence += min(
            0.15,
            evidence * 0.05,
        )

        confidence -= min(
            0.20,
            unknown * 0.05,
        )

        return Plan(
            objective=objective,
            steps=steps,
            confidence=max(
                0.0,
                min(1.0, confidence),
            ),
        )

    def _investigation_plan(
        self,
        problem: Problem,
        objective: str,
    ) -> Plan:

        steps: List[PlanStep] = [
            PlanStep(
                action=(
                    "Identificar y priorizar "
                    "la información faltante."
                ),
                expected_result=(
                    "Lista de incógnitas ordenadas por impacto "
                    "sobre la decisión."
                ),
            ),
            PlanStep(
                action=(
                    "Analizar los hechos y evidencias "
                    "disponibles."
                ),
                expected_result=(
                    "Separación entre hechos, observaciones, "
                    "inferencias y supuestos."
                ),
            ),
        ]

        if problem.constraints:
            steps.append(
                PlanStep(
                    action=(
                        "Evaluar las restricciones antes "
                        "de proponer cambios."
                    ),
                    expected_result=(
                        "Límites operativos claramente definidos."
                    ),
                )
            )

        steps.extend(
            [
                PlanStep(
                    action=(
                        "Generar alternativas compatibles "
                        "con la evidencia disponible."
                    ),
                    expected_result=(
                        "Una o más estrategias candidatas."
                    ),
                ),
                PlanStep(
                    action=(
                        "Evaluar riesgos y consecuencias "
                        "de las alternativas."
                    ),
                    expected_result=(
                        "Alternativa preferida con riesgo conocido."
                    ),
                ),
                PlanStep(
                    action="Verificar la estrategia seleccionada.",
                    expected_result=(
                        "Resultado observable que permita "
                        "confirmar o rechazar la estrategia."
                    ),
                ),
            ]
        )

        confidence = (
            0.40
            + min(
                0.25,
                len(problem.known_facts) * 0.05,
            )
            + min(
                0.15,
                len(problem.evidence) * 0.05,
            )
            - min(
                0.20,
                len(problem.unknowns) * 0.05,
            )
        )

        return Plan(
            objective=objective,
            steps=steps,
            confidence=max(
                0.0,
                min(1.0, confidence),
            ),
        )

    def _generic_plan(
        self,
        problem: Problem,
        objective: str,
    ) -> Plan:

        steps = [
            PlanStep(
                action="Analizar los hechos disponibles.",
                expected_result=(
                    "Situación actual estructurada."
                ),
            ),
            PlanStep(
                action=(
                    "Evaluar restricciones y supuestos."
                ),
                expected_result=(
                    "Condiciones que limitan las alternativas."
                ),
            ),
            PlanStep(
                action="Generar alternativas de solución.",
                expected_result=(
                    "Estrategias candidatas."
                ),
            ),
            PlanStep(
                action="Evaluar riesgos y consecuencias.",
                expected_result=(
                    "Alternativa preferida."
                ),
            ),
            PlanStep(
                action="Verificar la solución.",
                expected_result=(
                    "Resultado observable."
                ),
            ),
        ]

        return Plan(
            objective=objective,
            steps=steps,
            confidence=0.45,
        )

    @staticmethod
    def _problem_text(
        problem: Problem,
    ) -> str:

        parts = [
            problem.description,
            problem.objective or "",
            *problem.constraints,
            *problem.known_facts,
            *problem.unknowns,
            *problem.assumptions,
        ]

        return " ".join(parts).lower()

    @staticmethod
    def _is_performance_problem(
        text: str,
    ) -> bool:

        keywords = (
            "lento",
            "lentitud",
            "rendimiento",
            "performance",
            "ram",
            "cpu",
            "recursos",
            "latencia",
            "velocidad",
            "contexto",
            "tokens/s",
            "tokens por segundo",
        )

        normalized = re.sub(
            r"\s+",
            " ",
            text,
        )

        return any(
            keyword in normalized
            for keyword in keywords
        )
