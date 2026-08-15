from cognition.models import (
    Evidence,
    EvidenceType,
    Hypothesis,
    Problem,
    HypothesisStatus,
)

from cognition.question_engine import QuestionEngine
from cognition.cognitive_supervisor import CognitiveSupervisor


def test_unknown_becomes_diagnostic_question():
    engine = QuestionEngine()

    problem = Problem(
        description="El teléfono está lento.",
        unknowns=[
            "Determinar la causa del problema."
        ],
    )

    questions = engine.generate(problem)

    assert questions
    assert any(
        "información observable" in question.lower()
        for question in questions
    )


def test_hypothesis_generates_verification_question():
    engine = QuestionEngine()

    problem = Problem(
        description="El teléfono está lento.",
    )

    hypothesis = Hypothesis(
        statement="La presión de memoria puede estar contribuyendo.",
        verification_required=True,
    )

    questions = engine.generate(
        problem,
        [hypothesis],
    )

    assert any(
        "confirmar o descartar" in question.lower()
        for question in questions
    )


def test_performance_problem_generates_measurement_questions():
    engine = QuestionEngine()

    problem = Problem(
        description="Mi teléfono está lento.",
        objective="Determinar la causa.",
    )

    questions = engine.generate(problem)

    assert any(
        "cpu y ram" in question.lower()
        for question in questions
    )

    assert any(
        "tokens por segundo" in question.lower()
        for question in questions
    )


def test_supervisor_uses_question_engine():
    supervisor = CognitiveSupervisor()

    problem = Problem(
        description="Mi teléfono está lento.",
        objective="Identificar la causa.",
        unknowns=[
            "Determinar la causa del problema."
        ],
        known_facts=[
            "Mía Isabella está ejecutándose en Android.",
            "El modelo utiliza llama.cpp.",
        ],
        evidence=[
            Evidence(
                content="La aplicación está ejecutando inferencia.",
                evidence_type=EvidenceType.OBSERVATION,
                confidence=0.8,
            )
        ],
    )

    result = supervisor.solve(problem)

    assert result.needs_more_information is True
    assert result.questions

    assert any(
        "cpu y ram" in question.lower()
        for question in result.questions
    )
