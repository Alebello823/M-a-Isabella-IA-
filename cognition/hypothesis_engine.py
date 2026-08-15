"""
Motor de generación y evaluación de hipótesis.

Diseño:
- determinista
- ligero
- sin LLM
- sin dependencias externas
- explicable

El LLM podrá utilizar este resultado posteriormente para
interpretar, ampliar o comunicar el razonamiento, pero no
es necesario para construir la estructura causal básica.
"""

import re
from typing import Iterable, List, Set

from cognition.models import (
    Evidence,
    EvidenceType,
    Hypothesis,
    Problem,
)
from cognition.reasoning_engine import ReasoningEngine


class HypothesisEngine:

    def __init__(
        self,
        reasoning_engine: ReasoningEngine = None,
    ):
        self.reasoning = (
            reasoning_engine
            or ReasoningEngine()
        )

    # ==========================================================
    # GENERACIÓN
    # ==========================================================

    def generate(
        self,
        problem: Problem,
    ) -> List[Hypothesis]:
        """
        Genera hipótesis a partir de relaciones observables.

        Prioridad:

        1. Evidencia + hechos
        2. Restricciones
        3. Incógnitas
        4. Supuestos

        No pretende descubrir verdades nuevas.
        Construye candidatos que posteriormente deben evaluarse.
        """

        hypotheses: List[Hypothesis] = []
        seen: Set[str] = set()

        def add(
            statement: str,
            prior: float,
        ) -> None:
            normalized = self._normalize(statement)

            if not normalized:
                return

            if normalized in seen:
                return

            seen.add(normalized)

            hypotheses.append(
                Hypothesis(
                    statement=statement,
                    prior_probability=prior,
                )
            )

        text = self._problem_text(problem)

        # ------------------------------------------------------
        # 1. Relaciones problema -> hechos
        # ------------------------------------------------------

        for fact in problem.known_facts:
            if self._related(
                problem.description,
                fact,
            ):
                add(
                    (
                        f"El hecho '{fact}' puede contribuir "
                        f"al problema."
                    ),
                    0.65,
                )

        # ------------------------------------------------------
        # 2. Relaciones problema -> restricciones
        # ------------------------------------------------------

        for constraint in problem.constraints:
            if self._related(
                problem.description,
                constraint,
            ):
                add(
                    (
                        f"La restricción '{constraint}' "
                        "puede ser una causa relevante del problema."
                    ),
                    0.65,
                )
            else:
                add(
                    (
                        f"La restricción '{constraint}' "
                        "limita las soluciones disponibles."
                    ),
                    0.55,
                )

        # ------------------------------------------------------
        # 3. Señales técnicas conocidas
        # ------------------------------------------------------

        technical_rules = (
            (
                ("ram", "memoria", "recursos"),
                "La presión de memoria puede estar contribuyendo "
                "al problema.",
                0.60,
            ),
            (
                ("cpu", "procesador", "recursos"),
                "La carga del procesador puede estar contribuyendo "
                "al problema.",
                0.60,
            ),
            (
                ("contexto", "tokens", "ventana"),
                "El tamaño del contexto puede estar contribuyendo "
                "al problema.",
                0.55,
            ),
            (
                ("modelo", "inferencia", "llama.cpp"),
                "La configuración de inferencia puede estar "
                "contribuyendo al problema.",
                0.55,
            ),
        )

        for keywords, statement, prior in technical_rules:
            if any(
                keyword in text
                for keyword in keywords
            ):
                add(
                    statement,
                    prior,
                )

        # ------------------------------------------------------
        # 4. Incógnitas
        # ------------------------------------------------------

        for unknown in problem.unknowns:
            add(
                (
                    f"La incógnita '{unknown}' "
                    "puede contener información necesaria "
                    "para identificar la causa."
                ),
                0.45,
            )

        # ------------------------------------------------------
        # 5. Supuestos
        # ------------------------------------------------------

        for assumption in problem.assumptions:
            add(
                (
                    f"El supuesto '{assumption}' "
                    "podría explicar parcialmente el problema, "
                    "pero requiere verificación."
                ),
                0.30,
            )

        # ------------------------------------------------------
        # 6. Fallback
        # ------------------------------------------------------

        if not hypotheses:
            add(
                (
                    "La información disponible no permite "
                    "identificar todavía una causa dominante."
                ),
                0.35,
            )

        return hypotheses

    # ==========================================================
    # EVALUACIÓN
    # ==========================================================

    def evaluate(
        self,
        hypotheses: Iterable[Hypothesis],
        evidence: Iterable[Evidence],
    ) -> List[Hypothesis]:

        evidence = list(evidence)

        for hypothesis in hypotheses:

            statement_words = {
                word
                for word in self._normalize(
                    hypothesis.statement
                ).split()
                if len(word) > 5
            }

            # Evitar duplicación si la hipótesis
            # ya hubiera sido evaluada anteriormente.
            hypothesis.supporting_evidence.clear()
            hypothesis.opposing_evidence.clear()

            for item in evidence:

                text = self._normalize(
                    item.content
                )

                overlap = sum(
                    1
                    for word in statement_words
                    if word in text
                )

                # La coincidencia textual solamente
                # proporciona evidencia débil.
                if overlap >= 3:

                    hypothesis.supporting_evidence.append(
                        item
                    )

                elif overlap == 2:

                    if item.evidence_type in (
                        EvidenceType.FACT,
                        EvidenceType.OBSERVATION,
                        EvidenceType.EXTERNAL,
                    ):
                        hypothesis.supporting_evidence.append(
                            item
                        )

            # Cada hipótesis se evalúa exactamente
            # una vez en este ciclo.
            self.reasoning.evaluate_hypothesis(
                hypothesis
            )

        return sorted(
            hypotheses,
            key=lambda h: h.confidence,
            reverse=True,
        )

    # ==========================================================
    # UTILIDADES
    # ==========================================================

    def _problem_text(
        self,
        problem: Problem,
    ) -> str:
        """
        Construye una representación textual determinista
        del problema para activar reglas técnicas.
        """

        parts = [
            problem.description,
            problem.objective or "",
            *problem.constraints,
            *problem.known_facts,
            *problem.unknowns,
            *problem.assumptions,
        ]

        return self._normalize(" ".join(parts))

    def _related(
        self,
        left: str,
        right: str,
    ) -> bool:
        """
        Determina si dos textos contienen señales léxicas
        suficientemente relacionadas.

        No afirma causalidad; solamente determina
        relación temática para generar hipótesis.
        """

        left_words = {
            word
            for word in self._normalize(left).split()
            if len(word) >= 4
        }

        right_words = {
            word
            for word in self._normalize(right).split()
            if len(word) >= 4
        }

        if not left_words or not right_words:
            return False

        overlap = left_words & right_words

        return len(overlap) >= 1

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normaliza texto para comparación de reglas.

        No modifica el contenido original del problema.
        """

        text = str(text).lower()
        text = re.sub(r"\s+", " ", text)

        return text.strip()
