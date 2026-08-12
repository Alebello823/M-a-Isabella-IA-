"""
Motor de reflexión posterior a una tarea.

No modifica código.
No modifica automáticamente la arquitectura.
Registra lecciones y posibles mejoras.
"""

from typing import Iterable, List

from cognition.models import Plan, Reflection


class ReflectionEngine:

    def reflect(
        self,
        plan: Plan,
        actual_results: Iterable[str] = (),
    ) -> Reflection:

        actual_results = list(actual_results)

        observations: List[str] = []
        mistakes: List[str] = []
        lessons: List[str] = []
        improvements: List[str] = []

        expected = [
            step.expected_result
            for step in plan.steps
            if step.expected_result
        ]

        if not actual_results:
            return Reflection(
                success=False,
                observations=[
                    "No se proporcionaron resultados reales."
                ],
                mistakes=[
                    "No es posible verificar el plan."
                ],
                lessons=[
                    "Una acción debe tener un resultado observable."
                ],
                improvements=[
                    "Agregar una etapa explícita de verificación."
                ],
                confidence=0.20,
            )

        observations.extend(actual_results)

        matched = 0

        for expected_result in expected:
            for actual in actual_results:
                if self._similar(
                    expected_result,
                    actual,
                ):
                    matched += 1
                    break

        success = (
            matched >= max(
                1,
                len(expected) // 2
            )
        )

        if success:
            lessons.append(
                "La estrategia produjo resultados compatibles "
                "con las expectativas."
            )

            improvements.append(
                "Conservar los pasos que demostraron ser efectivos."
            )

        else:
            mistakes.append(
                "Los resultados no coinciden suficientemente "
                "con las expectativas."
            )

            lessons.append(
                "La hipótesis o el plan deben reevaluarse."
            )

            improvements.append(
                "Aumentar la recolección de evidencia antes "
                "de repetir la estrategia."
            )

        confidence = (
            matched / len(expected)
            if expected
            else 0.30
        )

        return Reflection(
            success=success,
            observations=observations,
            mistakes=mistakes,
            lessons=lessons,
            improvements=improvements,
            confidence=max(
                0.0,
                min(1.0, confidence)
            ),
        )

    @staticmethod
    def _similar(
        expected: str,
        actual: str,
    ) -> bool:

        expected_words = {
            word.lower()
            for word in expected.split()
            if len(word) > 4
        }

        actual_lower = actual.lower()

        if not expected_words:
            return False

        matches = sum(
            1
            for word in expected_words
            if word in actual_lower
        )

        return (
            matches / len(expected_words)
        ) >= 0.30
