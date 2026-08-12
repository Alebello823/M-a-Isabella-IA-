"""
Calculador de confianza del Cognitive Core.

Diseño:
- determinista
- ligero
- explicable
- sin LLM
- sin dependencias externas

Principio fundamental:

    confianza != certeza

La confianza operacional representa qué tan
bien sustentada está una decisión con la
información disponible.

Nunca debe convertir automáticamente una
observación en causalidad.
"""

from typing import Iterable, Optional

from cognition.models import (
    Evidence,
    EvidenceType,
    Hypothesis,
    HypothesisStatus,
)


class ConfidenceCalculator:
    """
    Calcula confianza operacional para el Cognitive Core.

    El objetivo no es producir una probabilidad
    matemática exacta.

    El objetivo es responder:

        "¿Qué tan razonable es avanzar
         con la información disponible?"
    """

    # ---------------------------------------------------------
    # HIPÓTESIS
    # ---------------------------------------------------------

    def from_hypotheses(
        self,
        hypotheses: Iterable[Hypothesis],
    ) -> float:
        """
        Calcula la confianza derivada de las hipótesis.

        Una hipótesis no debe considerarse fuerte
        únicamente porque tenga un confidence alto.

        También importa:

        - evidencia a favor
        - evidencia en contra
        - estado de la hipótesis
        - separación respecto a alternativas
        """

        hypotheses = list(hypotheses)

        if not hypotheses:
            return 0.0

        scored = []

        for hypothesis in hypotheses:
            support = self._evidence_strength(
                hypothesis.supporting_evidence
            )

            opposition = self._evidence_strength(
                hypothesis.opposing_evidence
            )

            # La evidencia neta limita la confianza.
            evidence_balance = (
                support / (support + opposition)
                if support + opposition > 0
                else 0.0
            )

            # Una hipótesis sin evidencia no debe
            # convertirse mágicamente en certeza.
            if support <= 0:
                evidence_balance *= 0.50

            state_factor = self._status_factor(
                hypothesis.status
            )

            score = (
                hypothesis.confidence * 0.45
                + evidence_balance * 0.40
                + state_factor * 0.15
            )

            scored.append(
                self._clamp(score)
            )

        scored.sort(reverse=True)

        best = scored[0]

        if len(scored) == 1:
            return self._clamp(best)

        second = scored[1]

        separation = max(
            0.0,
            best - second,
        )

        # Una hipótesis dominante aporta
        # algo más de confianza, pero no demasiado.
        result = (
            best * 0.90
            + separation * 0.10
        )

        return self._clamp(result)

    # ---------------------------------------------------------
    # EVIDENCIA
    # ---------------------------------------------------------

    @staticmethod
    def evidence_strength(
        evidence: Iterable[Evidence],
    ) -> float:
        """
        Devuelve una medida normalizada de fuerza
        de la evidencia disponible.

        No mide cantidad solamente.

        La calidad depende del tipo de evidencia.
        """

        return ConfidenceCalculator._evidence_strength(
            evidence
        )

    @staticmethod
    def _evidence_strength(
        evidence: Iterable[Evidence],
    ) -> float:

        evidence = list(evidence)

        if not evidence:
            return 0.0

        total = 0.0
        weight = 0.0

        factors = {
            EvidenceType.FACT: 1.00,
            EvidenceType.EXTERNAL: 0.90,
            EvidenceType.OBSERVATION: 0.80,
            EvidenceType.INFERENCE: 0.55,
            EvidenceType.ASSUMPTION: 0.25,
        }

        for item in evidence:
            factor = factors.get(
                item.evidence_type,
                0.50,
            )

            total += (
                ConfidenceCalculator._clamp(
                    item.confidence
                )
                * factor
            )

            weight += factor

        if weight <= 0:
            return 0.0

        return ConfidenceCalculator._clamp(
            total / weight
        )

    # ---------------------------------------------------------
    # ESTADO DE HIPÓTESIS
    # ---------------------------------------------------------

    @staticmethod
    def _status_factor(
        status: HypothesisStatus,
    ) -> float:

        factors = {
            HypothesisStatus.OPEN: 0.40,
            HypothesisStatus.SUPPORTED: 0.70,
            HypothesisStatus.WEAKENED: 0.25,
            HypothesisStatus.REJECTED: 0.00,
            HypothesisStatus.CONFIRMED: 0.90,
        }

        return factors.get(status, 0.40)

    # ---------------------------------------------------------
    # CONFIANZA GLOBAL
    # ---------------------------------------------------------

    def combine(
        self,
        analysis_confidence: float,
        hypothesis_confidence: float,
        plan_confidence: float,
        simulation_probability: float,
        decision_confidence: float,
        information_confidence: float,
        weights: Optional[Iterable[float]] = None,
    ) -> float:
        """
        Combina las señales principales del ciclo cognitivo.

        Importante:

        La confianza global NO debe ser una simple
        media de todo.

        El análisis y la evidencia tienen mayor
        importancia que la simulación, porque una
        simulación todavía no representa un resultado real.

        Pesos por defecto:

        análisis:       25%
        hipótesis:      25%
        plan:           15%
        simulación:     10%
        decisión:       10%
        información:   15%
        """

        values = [
            self._clamp(analysis_confidence),
            self._clamp(hypothesis_confidence),
            self._clamp(plan_confidence),
            self._clamp(simulation_probability),
            self._clamp(decision_confidence),
            self._clamp(information_confidence),
        ]

        if weights is None:
            weights = [
                0.25,
                0.25,
                0.15,
                0.10,
                0.10,
                0.15,
            ]
        else:
            weights = list(weights)

        if len(values) != len(weights):
            raise ValueError(
                "values y weights deben tener la misma longitud"
            )

        if any(weight < 0 for weight in weights):
            raise ValueError(
                "Los pesos no pueden ser negativos"
            )

        total_weight = sum(weights)

        if total_weight <= 0:
            return 0.0

        result = sum(
            value * weight
            for value, weight in zip(
                values,
                weights,
            )
        )

        return self._clamp(
            result / total_weight
        )

    # ---------------------------------------------------------
    # UTILIDAD
    # ---------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:
        return max(
            0.0,
            min(1.0, float(value)),
        )
