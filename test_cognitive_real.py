from cognition.models import Problem, Evidence, EvidenceType
from cognition.cognitive_supervisor import CognitiveSupervisor


problem = Problem(
    description="La generación de texto funciona correctamente, pero la velocidad de inferencia es baja.",

    objective=(
        "Determinar la causa más probable de la baja velocidad "
        "y proponer una mejora segura y reversible."
    ),

    constraints=[
        "RAM limitada",
        "No cambiar el modelo todavía",
        "No modificar permanentemente la configuración",
        "No ejecutar acciones destructivas",
    ],

    known_facts=[
        "La inferencia utiliza llama.cpp.",
        "La generación funciona correctamente.",
        "El dispositivo tiene recursos limitados.",
        "La velocidad observada es inferior a la esperada.",
    ],

    unknowns=[
        "Consumo real de RAM durante la inferencia.",
        "Uso real de CPU durante la inferencia.",
        "Efecto del tamaño del contexto.",
        "Número de threads utilizado.",
    ],

    assumptions=[
        "La presión de memoria podría estar reduciendo el rendimiento.",
        "El tamaño del contexto podría afectar la velocidad.",
    ],

    success_conditions=[
        "Identificar una causa probable.",
        "Proponer una acción reversible.",
    ],

    evidence=[
        Evidence(
            content="La inferencia utiliza llama.cpp.",
            evidence_type=EvidenceType.FACT,
            confidence=1.0,
            source="system",
        ),
        Evidence(
            content="La generación funciona correctamente.",
            evidence_type=EvidenceType.OBSERVATION,
            confidence=0.95,
            source="runtime",
        ),
        Evidence(
            content="La velocidad observada es inferior a la esperada.",
            evidence_type=EvidenceType.OBSERVATION,
            confidence=0.95,
            source="runtime",
        ),
    ],
)


brain = CognitiveSupervisor()
result = brain.solve(problem)

print()
print("=" * 50)
print("MÍA ISABELLA - REAL COGNITIVE TEST")
print("=" * 50)

print()
print("CONFIDENCE:", round(result.confidence, 3))
print("UNCERTAINTY:", round(result.uncertainty, 3))
print("NEEDS INFORMATION:", result.needs_more_information)

print()
print("HYPOTHESES")

for h in result.hypotheses:
    print(
        f"[{h.status.value:10}] "
        f"confidence={h.confidence:.2f} | "
        f"{h.statement}"
    )

print()
print("PLAN")

for i, step in enumerate(result.plan.steps, 1):
    print(f"{i}. {step.action}")

print()
print("DECISION")
print("status:", result.decision.status.value)
print("action:", result.decision.action)
print(
    "confidence:",
    round(result.decision.confidence, 3)
)

print()
print("METADATA")
print(result.metadata)
