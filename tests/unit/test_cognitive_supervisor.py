"""
Pruebas unitarias del CognitiveSupervisor de Mía Isabella.

Objetivo:

    Problem
       ↓
    Reasoning
       ↓
    Uncertainty
       ↓
    Hypotheses
       ↓
    Planning
       ↓
    Simulation
       ↓
    Risk
       ↓
    Decision
       ↓
    Confidence
       ↓
    CognitiveResult

Estas pruebas verifican la integración del ciclo cognitivo
sin utilizar LLM ni ejecutar herramientas reales.
"""

from cognition.cognitive_supervisor import CognitiveSupervisor
from cognition.models import (
    Evidence,
    EvidenceType,
    Problem,
    DecisionStatus,
    HypothesisStatus,
    RiskLevel,
)


def build_performance_problem() -> Problem:
    """
    Construye un problema representativo de rendimiento.
    """

    return Problem(
        description=(
            "Mi teléfono está lento y quiero saber por qué."
        ),
        objective=(
            "Identificar la causa probable del problema "
            "y determinar una estrategia adecuada."
        ),
        known_facts=[
            "El problema comenzó ayer.",
            "Mía Isabella está ejecutándose en Android.",
            "El modelo utiliza llama.cpp.",
        ],
        unknowns=[
            "Determinar la causa del problema.",
        ],
        assumptions=[],
        constraints=[],
        success_conditions=[
            "Identificar una causa suficientemente "
            "sustentada por evidencia.",
            "Evitar afirmar causalidad sin verificación.",
        ],
        evidence=[
            Evidence(
                content="El problema comenzó ayer.",
                evidence_type=EvidenceType.OBSERVATION,
                confidence=0.60,
                source="conversation_context",
            ),
            Evidence(
                content=(
                    "Mía Isabella está ejecutándose en Android."
                ),
                evidence_type=EvidenceType.OBSERVATION,
                confidence=0.60,
                source="conversation_context",
            ),
            Evidence(
                content="El modelo utiliza llama.cpp.",
                evidence_type=EvidenceType.OBSERVATION,
                confidence=0.60,
                source="conversation_context",
            ),
        ],
    )


def test_cognitive_supervisor_returns_result():
    """
    Verifica que el Supervisor complete todo el ciclo
    y devuelva un CognitiveResult válido.
    """

    supervisor = CognitiveSupervisor()

    problem = build_performance_problem()

    result = supervisor.solve(problem)

    assert result is not None

    assert result.problem is problem

    assert result.hypotheses

    assert result.plan is not None

    assert result.plan.steps

    assert result.decision is not None

    assert result.reflection is None

    assert 0.0 <= result.confidence <= 1.0

    assert 0.0 <= result.uncertainty <= 1.0

    assert isinstance(
        result.needs_more_information,
        bool,
    )

    assert isinstance(
        result.questions,
        list,
    )


def test_hypotheses_are_evaluated():
    """
    Las hipótesis deben salir evaluadas y ordenadas.
    """

    supervisor = CognitiveSupervisor()

    problem = build_performance_problem()

    result = supervisor.solve(problem)

    assert len(result.hypotheses) >= 1

    confidences = [
        hypothesis.confidence
        for hypothesis in result.hypotheses
    ]

    assert confidences == sorted(
        confidences,
        reverse=True,
    )

    for hypothesis in result.hypotheses:
        assert 0.0 <= hypothesis.confidence <= 1.0

        assert hypothesis.status in (
            HypothesisStatus.OPEN,
            HypothesisStatus.SUPPORTED,
            HypothesisStatus.WEAKENED,
            HypothesisStatus.REJECTED,
            HypothesisStatus.CONFIRMED,
        )


def test_performance_problem_generates_performance_plan():
    """
    Un problema de rendimiento debe producir el plan
    especializado correspondiente.
    """

    supervisor = CognitiveSupervisor()

    problem = build_performance_problem()

    result = supervisor.solve(problem)

    actions = [
        step.action.lower()
        for step in result.plan.steps
    ]

    assert any(
        "cpu" in action and "ram" in action
        for action in actions
    )

    assert any(
        "contexto" in action
        for action in actions
    )

    assert any(
        "llama.cpp" in action
        for action in actions
    )


def test_plan_steps_are_reversible():
    """
    El Cognitive Core no debe proponer cambios irreversibles
    en este escenario diagnóstico.
    """

    supervisor = CognitiveSupervisor()

    problem = build_performance_problem()

    result = supervisor.solve(problem)

    assert result.plan.steps

    assert all(
        step.reversible
        for step in result.plan.steps
    )


def test_supervisor_requests_information_when_unknowns_exist():
    """
    Si existen incógnitas relevantes, el resultado cognitivo
    debe reflejar que todavía falta información.
    """

    supervisor = CognitiveSupervisor()

    problem = build_performance_problem()

    result = supervisor.solve(problem)

    assert result.needs_more_information is True

    assert result.questions

    assert (
        "Determinar la causa del problema."
        in result.questions
    )


def test_decision_is_structured():
    """
    La decisión debe contener una acción y un estado válido.
    """

    supervisor = CognitiveSupervisor()

    problem = build_performance_problem()

    result = supervisor.solve(problem)

    decision = result.decision

    assert decision is not None

    assert decision.action.strip()

    assert 0.0 <= decision.confidence <= 1.0

    assert decision.status in (
        DecisionStatus.PROPOSED,
        DecisionStatus.ACCEPTED,
        DecisionStatus.REJECTED,
        DecisionStatus.NEEDS_CONFIRMATION,
    )


def test_risk_metadata_is_present():
    """
    El Supervisor debe conservar información del Risk Engine
    dentro de metadata.
    """

    supervisor = CognitiveSupervisor()

    problem = build_performance_problem()

    result = supervisor.solve(problem)

    metadata = result.metadata

    assert "risk_score" in metadata
    assert "risk_level" in metadata
    assert "simulation_probability" in metadata
    assert "hypothesis_confidence" in metadata
    assert "problem_analysis" in metadata

    assert 0.0 <= metadata["risk_score"] <= 1.0

    assert 0.0 <= (
        metadata["simulation_probability"]
    ) <= 1.0

    assert 0.0 <= (
        metadata["hypothesis_confidence"]
    ) <= 1.0

    assert metadata["risk_level"] in (
        RiskLevel.LOW.value,
        RiskLevel.MEDIUM.value,
        RiskLevel.HIGH.value,
        RiskLevel.CRITICAL.value,
    )


def test_cognitive_pipeline_is_deterministic():
    """
    Al ser un Cognitive Core determinista, el mismo problema
    debe producir los mismos valores principales.
    """

    supervisor = CognitiveSupervisor()

    problem_1 = build_performance_problem()
    problem_2 = build_performance_problem()

    result_1 = supervisor.solve(problem_1)
    result_2 = supervisor.solve(problem_2)

    assert result_1.confidence == result_2.confidence

    assert result_1.uncertainty == result_2.uncertainty

    assert (
        result_1.needs_more_information
        == result_2.needs_more_information
    )

    assert [
        h.statement
        for h in result_1.hypotheses
    ] == [
        h.statement
        for h in result_2.hypotheses
    ]

    assert [
        h.confidence
        for h in result_1.hypotheses
    ] == [
        h.confidence
        for h in result_2.hypotheses
    ]

    assert [
        step.action
        for step in result_1.plan.steps
    ] == [
        step.action
        for step in result_2.plan.steps
    ]
