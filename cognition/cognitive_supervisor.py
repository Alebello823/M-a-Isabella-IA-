"""
Supervisor cognitivo de Mía Isabella.

Coordina el Cognitive Core.

No ejecuta herramientas.
No modifica archivos.
No ejecuta acciones reales.

Su función es convertir un problema en:

problema
    ↓
evidencia
    ↓
hipótesis
    ↓
plan
    ↓
simulación
    ↓
riesgo
    ↓
decisión
    ↓
confianza
"""

import logging
from typing import Optional

from cognition.confidence_calculator import ConfidenceCalculator
from cognition.decision_maker import DecisionMaker
from cognition.hypothesis_engine import HypothesisEngine
from cognition.models import CognitiveResult, Problem
from cognition.planning_engine import PlanningEngine
from cognition.reasoning_engine import ReasoningEngine
from cognition.risk_engine import RiskEngine
from cognition.simulation_engine import SimulationEngine
from cognition.uncertainty_engine import UncertaintyEngine


logger = logging.getLogger(__name__)


class CognitiveSupervisor:

    def __init__(
        self,
        reasoning_engine: Optional[ReasoningEngine] = None,
        hypothesis_engine: Optional[HypothesisEngine] = None,
        planning_engine: Optional[PlanningEngine] = None,
        simulation_engine: Optional[SimulationEngine] = None,
        risk_engine: Optional[RiskEngine] = None,
        decision_maker: Optional[DecisionMaker] = None,
        uncertainty_engine: Optional[UncertaintyEngine] = None,
        confidence_calculator: Optional[ConfidenceCalculator] = None,
    ):

        self.reasoning = (
            reasoning_engine
            or ReasoningEngine()
        )

        self.hypotheses = (
            hypothesis_engine
            or HypothesisEngine(self.reasoning)
        )

        self.planning = (
            planning_engine
            or PlanningEngine()
        )

        self.simulation = (
            simulation_engine
            or SimulationEngine()
        )

        self.risk = (
            risk_engine
            or RiskEngine()
        )

        self.decision = (
            decision_maker
            or DecisionMaker()
        )

        self.uncertainty = (
            uncertainty_engine
            or UncertaintyEngine()
        )

        self.confidence = (
            confidence_calculator
            or ConfidenceCalculator()
        )

    def solve(
        self,
        problem: Problem,
    ) -> CognitiveResult:

        logger.info(
            "Iniciando ciclo cognitivo: %s",
            problem.description,
        )

        # -----------------------------------------------------
        # 1. Comprender el problema
        # -----------------------------------------------------

        analysis = self.reasoning.analyze_problem(
            problem
        )

        # -----------------------------------------------------
        # 2. Medir incertidumbre
        # -----------------------------------------------------

        uncertainty = self.uncertainty.calculate(
            problem,
            problem.evidence,
        )

        # -----------------------------------------------------
        # 3. Generar y evaluar hipótesis
        # -----------------------------------------------------

        hypotheses = self.hypotheses.generate(
            problem
        )

        hypotheses = self.hypotheses.evaluate(
            hypotheses,
            problem.evidence,
        )

        # -----------------------------------------------------
        # 4. Construir plan
        #
        # Las hipótesis evaluadas se pasan al planificador.
        # En esta etapa PlanningEngine mantiene su comportamiento
        # actual; posteriormente utilizaremos estas hipótesis
        # para mejorar la priorización del plan.
        # -----------------------------------------------------

        plan = self.planning.create_plan(
            problem,
            hypotheses=hypotheses,
        )

        plan = self.planning.prioritize_steps(
            plan
        )

        # -----------------------------------------------------
        # 5. Simular sin ejecutar
        # -----------------------------------------------------

        simulation = self.simulation.simulate(
            plan
        )

        # -----------------------------------------------------
        # 6. Evaluar riesgo
        # -----------------------------------------------------

        risk = self.risk.assess_plan(
            plan
        )

        # -----------------------------------------------------
        # 7. Elegir estrategia
        # -----------------------------------------------------

        decision = self.decision.choose(
            plan=plan,
            risk=risk,
            hypotheses=hypotheses,
            uncertainty=uncertainty,
        )

        # -----------------------------------------------------
        # 8. Confianza de hipótesis
        # -----------------------------------------------------

        hypothesis_confidence = (
            self.confidence.from_hypotheses(
                hypotheses
            )
        )

        # -----------------------------------------------------
        # 9. Confianza global
        # -----------------------------------------------------

        global_confidence = (
            self.confidence.combine(
                analysis["confidence"],
                hypothesis_confidence,
                plan.confidence,
                simulation.success_probability,
                decision.confidence,
                1.0 - uncertainty,
            )
        )

        # -----------------------------------------------------
        # 10. Determinar necesidad de información
        # -----------------------------------------------------

        needs_information = (
            bool(analysis["needs_more_information"])
            or uncertainty >= 0.65
            or bool(analysis["contradictions"])
        )

        questions = list(
            problem.unknowns
        )

        # Si existe contradicción, convertirla
        # en una pregunta cognitiva explícita.

        for first, second in analysis["contradictions"]:
            questions.append(
                f"Resolver contradicción entre "
                f"'{first}' y '{second}'."
            )

        logger.info(
            "Ciclo cognitivo terminado: "
            "confidence=%.2f uncertainty=%.2f",
            global_confidence,
            uncertainty,
        )

        # -----------------------------------------------------
        # 11. Resultado cognitivo
        # -----------------------------------------------------

        return CognitiveResult(
            problem=problem,
            hypotheses=hypotheses,
            plan=plan,
            decision=decision,
            confidence=global_confidence,
            uncertainty=uncertainty,
            needs_more_information=needs_information,
            questions=questions,
            metadata={
                "problem_analysis": analysis,
                "hypothesis_confidence": (
                    hypothesis_confidence
                ),
                "simulation_probability": (
                    simulation.success_probability
                ),
                "risk_score": risk.score,
                "risk_level": risk.level.value,
            },
        )
