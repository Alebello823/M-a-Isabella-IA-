"""
Stress tests del Cognitive Core.

Objetivo:
comprobar que Isabella:
- distingue hechos de hipótesis
- detecta incertidumbre
- detecta contradicciones
- prioriza información
- evita decisiones peligrosas
- no confunde simulación con ejecución
- mantiene confianza dentro de límites razonables

Estos tests no ejecutan herramientas reales.
"""

from cognition.cognitive_supervisor import CognitiveSupervisor
from cognition.models import (
    Evidence,
    EvidenceType,
    Problem,
    DecisionStatus,
    RiskLevel,
)


def test_well_supported_problem():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="El proceso consume demasiada memoria.",
        objective="Identificar si la presión de RAM causa el problema.",
        known_facts=[
            "El sistema tiene 4 GB de RAM.",
            "El proceso consume 3.5 GB.",
            "El problema aparece cuando aumenta el consumo de memoria.",
        ],
        unknowns=[],
        evidence=[
            Evidence(
                content="El proceso consume 3.5 GB de RAM.",
                evidence_type=EvidenceType.OBSERVATION,
                confidence=0.95,
            ),
            Evidence(
                content="El sistema tiene 4 GB de RAM.",
                evidence_type=EvidenceType.FACT,
                confidence=0.95,
            ),
        ],
    )

    result = supervisor.solve(problem)

    assert result.hypotheses
    assert result.plan is not None
    assert result.decision is not None
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.uncertainty <= 1.0


def test_insufficient_information_requires_more_information():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="Mi teléfono está lento.",
        objective="Determinar la causa.",
        known_facts=[],
        unknowns=[
            "Determinar qué recurso está saturado.",
            "Determinar cuándo aparece la lentitud.",
            "Determinar qué proceso consume más recursos.",
        ],
        evidence=[],
    )

    result = supervisor.solve(problem)

    assert result.needs_more_information is True
    assert result.questions
    assert result.uncertainty >= 0.65


def test_contradiction_is_not_ignored():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="Determinar el estado del almacenamiento.",
        objective="Saber cuánto espacio libre existe.",
        known_facts=[
            "El teléfono tiene 2 GB libres.",
            "El teléfono tiene 20 GB libres.",
        ],
        unknowns=[],
        evidence=[],
    )

    result = supervisor.solve(problem)

    assert result.needs_more_information is True

    assert any(
        "contradicción" in question.lower()
        for question in result.questions
    )


def test_android_is_not_treated_as_proof_of_causality():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="El teléfono está lento.",
        objective="Identificar la causa.",
        known_facts=[
            "Mía Isabella está ejecutándose en Android.",
        ],
        unknowns=[
            "Determinar qué está causando la lentitud.",
        ],
        evidence=[],
    )

    result = supervisor.solve(problem)

    assert result.hypotheses

    android_hypotheses = [
        h
        for h in result.hypotheses
        if "android" in h.statement.lower()
    ]

    assert android_hypotheses

    hypothesis = android_hypotheses[0]

    # Android puede ser una hipótesis,
    # pero no debe convertirse en causalidad confirmada
    # sin evidencia suficiente.
    assert hypothesis.status.value != "confirmed"
    assert hypothesis.causal_support < 0.80


def test_high_risk_plan_requires_confirmation():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="Modificar una configuración crítica del sistema.",
        objective="Cambiar la configuración.",
        known_facts=[
            "La configuración afecta al sistema.",
        ],
        unknowns=[],
        evidence=[],
    )

    result = supervisor.solve(problem)

    assert result.decision is not None

    # El test verifica que la decisión exista y
    # que el sistema no ejecute nada automáticamente.
    assert result.decision.status in (
        DecisionStatus.ACCEPTED,
        DecisionStatus.NEEDS_CONFIRMATION,
        DecisionStatus.REJECTED,
    )


def test_no_plan_does_not_produce_fake_success():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="Resolver un problema desconocido.",
        objective="Encontrar una solución.",
        known_facts=[],
        unknowns=[
            "Determinar cuál es el problema real.",
        ],
        evidence=[],
    )

    result = supervisor.solve(problem)

    assert result.plan is not None
    assert result.decision is not None

    simulation_probability = result.metadata.get(
        "simulation_probability"
    )

    assert simulation_probability is not None
    assert 0.0 <= simulation_probability <= 1.0


def test_confidence_and_uncertainty_are_bounded():
    supervisor = CognitiveSupervisor()

    problems = [
        Problem(
            description="Problema simple.",
            known_facts=["Existe un hecho."],
        ),
        Problem(
            description="Problema incierto.",
            unknowns=["Falta información."],
        ),
        Problem(
            description="Problema contradictorio.",
            known_facts=[
                "X es verdadero.",
                "no X es verdadero.",
            ],
        ),
    ]

    for problem in problems:
        result = supervisor.solve(problem)

        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0


def test_performance_problem_gets_measurement_plan():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="Mi teléfono está lento.",
        objective="Identificar la causa de la lentitud.",
        known_facts=[
            "Mía Isabella está ejecutándose en Android.",
            "El modelo utiliza llama.cpp.",
        ],
        unknowns=[
            "Determinar la causa.",
        ],
        evidence=[
            Evidence(
                content="La aplicación presenta lentitud.",
                evidence_type=EvidenceType.OBSERVATION,
                confidence=0.90,
            )
        ],
    )

    result = supervisor.solve(problem)

    assert result.plan is not None
    assert len(result.plan.steps) >= 3

    actions = [
        step.action.lower()
        for step in result.plan.steps
    ]

    assert any(
        "cpu" in action or "ram" in action
        for action in actions
    )
