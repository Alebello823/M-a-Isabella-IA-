from cognition.cognitive_supervisor import CognitiveSupervisor
from cognition.problem_analyzer import ProblemAnalyzer
from cognition.models import (
    DecisionStatus,
    RiskLevel,
)


def build_performance_problem():
    analyzer = ProblemAnalyzer()

    return analyzer.analyze(
        "Mi teléfono está lento y quiero saber por qué.",
        context_facts=[
            "El problema comenzó ayer.",
            "Mía Isabella está ejecutándose en Android.",
            "El modelo utiliza llama.cpp.",
        ],
    )


def test_cognitive_supervisor_produces_complete_result():
    problem = build_performance_problem()

    supervisor = CognitiveSupervisor()

    result = supervisor.solve(problem)

    assert result.problem is problem

    assert result.hypotheses
    assert result.plan is not None
    assert result.plan.steps

    assert result.decision is not None
    assert result.decision.action

    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.uncertainty <= 1.0

    assert isinstance(
        result.needs_more_information,
        bool,
    )


def test_performance_problem_prioritizes_measurement():
    problem = build_performance_problem()

    supervisor = CognitiveSupervisor()

    result = supervisor.solve(problem)

    assert result.plan is not None
    assert result.plan.steps

    first_action = (
        result.plan.steps[0].action.lower()
    )

    assert any(
        word in first_action
        for word in (
            "medir",
            "identificar",
            "observar",
            "diagnosticar",
        )
    )

    assert result.decision is not None

    assert result.decision.status in (
        DecisionStatus.ACCEPTED,
        DecisionStatus.NEEDS_CONFIRMATION,
    )
