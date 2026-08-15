"""
Motor determinista de generación de preguntas cognitivas.

Responsabilidades:

- preservar las incógnitas originales
- convertir incógnitas en preguntas diagnósticas
- orientar la verificación de hipótesis
- priorizar contradicciones
- generar preguntas específicas para problemas técnicos
- reducir incertidumbre mediante información observable

No utiliza LLM.
No ejecuta herramientas.
No modifica el sistema.

Principio:

    desconocimiento
          ↓
    pregunta verificable
          ↓
    evidencia observable
          ↓
    reducción de incertidumbre
"""


import re
from typing import Iterable, List, Set, Tuple

from cognition.models import Hypothesis, Problem


class QuestionEngine:
    """
    Generador determinista de preguntas cognitivas.

    El motor no intenta responder las preguntas.

    Su responsabilidad es determinar qué información
    debería obtenerse para reducir incertidumbre.
    """

    # ==========================================================
    # API PRINCIPAL
    # ==========================================================

    def generate(
        self,
        problem: Problem,
        hypotheses: Iterable[Hypothesis] = (),
        contradictions: Iterable[Tuple[str, str]] = (),
    ) -> List[str]:
        """
        Genera preguntas cognitivas a partir del problema.

        Orden de prioridad:

        1. incógnitas originales
        2. preguntas diagnósticas derivadas
        3. contradicciones
        4. verificación de hipótesis
        5. preguntas técnicas
        6. fallback general

        Las incógnitas originales se conservan literalmente.
        """

        hypotheses = list(hypotheses)
        contradictions = list(contradictions)

        questions: List[str] = []
        seen: Set[str] = set()

        def add(question: str) -> None:
            """
            Añade una pregunta evitando duplicados.
            """

            question = str(question).strip()

            if not question:
                return

            key = question.casefold()

            if key in seen:
                return

            seen.add(key)
            questions.append(question)

        # ======================================================
        # 1. INCÓGNITAS ORIGINALES
        # ======================================================
        #
        # Nunca reemplazamos la incógnita original.
        #
        # Esto permite conservar exactamente la intención
        # cognitiva recibida por el supervisor.
        # ======================================================

        for unknown in problem.unknowns:

            # Preservar literalmente la incógnita.
            add(unknown)

            # Generar una pregunta observable derivada.
            diagnostic = self._diagnostic_question_from_unknown(
                unknown
            )

            add(diagnostic)

        # ======================================================
        # 2. CONTRADICCIONES
        # ======================================================
        #
        # Las contradicciones tienen prioridad cognitiva:
        #
        # si dos hechos incompatibles existen,
        # Isabella no debe tratarlos como simultáneamente válidos.
        # ======================================================

        for contradiction in contradictions:

            if len(contradiction) != 2:
                continue

            first, second = contradiction

            first = str(first).strip()
            second = str(second).strip()

            if not first or not second:
                continue

            add(
                "Resolver contradicción entre "
                f"'{first}' y '{second}'."
            )

            add(
                "¿Qué evidencia observable permite "
                "determinar cuál de los dos hechos es correcto?"
            )

        # ======================================================
        # 3. VERIFICACIÓN DE HIPÓTESIS
        # ======================================================

        for hypothesis in hypotheses:

            if not hypothesis.verification_required:
                continue

            statement = str(
                hypothesis.statement
            ).strip()

            if not statement:
                continue

            add(
                "¿Qué medición u observación permitiría "
                "confirmar o descartar la hipótesis: "
                f"'{statement}'?"
            )

        # ======================================================
        # 4. PROBLEMAS TÉCNICOS / RENDIMIENTO
        # ======================================================

        text = self._problem_text(problem)

        if self._is_performance_problem(text):

            add(
                "¿Qué consumo de CPU y RAM presenta el sistema "
                "mientras ocurre la lentitud?"
            )

            add(
                "¿La lentitud aparece durante la inferencia "
                "o también cuando Mía Isabella está inactiva?"
            )

            add(
                "¿Cuál es la velocidad actual de inferencia "
                "en tokens por segundo?"
            )

            add(
                "¿Qué tamaño de contexto está utilizando "
                "actualmente el motor de inferencia?"
            )

        # ======================================================
        # 5. FALLBACK
        # ======================================================

        if not questions:

            add(
                "¿Qué información observable falta para "
                "confirmar o descartar las hipótesis actuales?"
            )

        return questions

    # ==========================================================
    # CONVERSIÓN DE INCÓGNITAS
    # ==========================================================

    @staticmethod
    def _diagnostic_question_from_unknown(
        unknown: str,
    ) -> str:
        """
        Convierte una incógnita en una pregunta verificable.

        Ejemplo:

            Determinar la causa del problema.

        →

            ¿Qué información observable permite determinar
            la causa del problema?
        """

        text = str(unknown).strip()

        if not text:
            return ""

        # ------------------------------------------------------
        # Ya es una pregunta.
        # ------------------------------------------------------

        if text.endswith("?"):
            return text

        # ------------------------------------------------------
        # Eliminar puntuación final.
        # ------------------------------------------------------

        text = text.rstrip(".! ")

        if not text:
            return ""

        normalized = text.casefold()

        # ======================================================
        # "DETERMINAR ..."
        # ======================================================

        prefix = "determinar "

        if normalized.startswith(prefix):
            objective = text[len(prefix):].strip()

            if objective:
                return (
                    "¿Qué información observable permite "
                    f"determinar {QuestionEngine._lower_first(objective)}?"
                )

        # ======================================================
        # "IDENTIFICAR ..."
        # ======================================================

        prefix = "identificar "

        if normalized.startswith(prefix):
            objective = text[len(prefix):].strip()

            if objective:
                return (
                    "¿Qué información observable permite "
                    f"identificar {QuestionEngine._lower_first(objective)}?"
                )

        # ======================================================
        # "SABER ..."
        # ======================================================

        prefix = "saber "

        if normalized.startswith(prefix):
            objective = text[len(prefix):].strip()

            if objective:
                return (
                    "¿Qué información observable permite "
                    f"saber {QuestionEngine._lower_first(objective)}?"
                )

        # ======================================================
        # "CONOCER ..."
        # ======================================================

        prefix = "conocer "

        if normalized.startswith(prefix):
            objective = text[len(prefix):].strip()

            if objective:
                return (
                    "¿Qué información observable permite "
                    f"conocer {QuestionEngine._lower_first(objective)}?"
                )

        # ======================================================
        # "VERIFICAR ..."
        # ======================================================

        prefix = "verificar "

        if normalized.startswith(prefix):
            objective = text[len(prefix):].strip()

            if objective:
                return (
                    "¿Qué información observable permite "
                    f"verificar {QuestionEngine._lower_first(objective)}?"
                )

        # ======================================================
        # "COMPROBAR ..."
        # ======================================================

        prefix = "comprobar "

        if normalized.startswith(prefix):
            objective = text[len(prefix):].strip()

            if objective:
                return (
                    "¿Qué información observable permite "
                    f"comprobar {QuestionEngine._lower_first(objective)}?"
                )

        # ======================================================
        # "IDENTIFICAR SI ..."
        # ======================================================

        # Este caso queda cubierto por "identificar",
        # pero lo dejamos conceptualmente documentado:
        #
        # Identificar si X ocurre
        #
        # →
        #
        # ¿Qué información observable permite identificar
        # si X ocurre?
        #
        # No necesitamos una regla adicional.

        # ======================================================
        # FALLBACK
        # ======================================================

        return (
            "¿Qué información observable necesitamos "
            "para determinar "
            f"{QuestionEngine._lower_first(text)}?"
        )

    # ==========================================================
    # PROBLEMAS DE RENDIMIENTO
    # ==========================================================

    @staticmethod
    def _is_performance_problem(
        text: str,
    ) -> bool:
        """
        Detecta señales lingüísticas relacionadas
        con rendimiento o recursos.

        Es solamente detección temática.
        No constituye un diagnóstico.
        """

        normalized = str(text).casefold()

        keywords = (
            "lento",
            "lentitud",
            "rendimiento",
            "performance",
            "ram",
            "memoria",
            "cpu",
            "procesador",
            "recursos",
            "latencia",
            "velocidad",
            "contexto",
            "tokens/s",
            "tokens por segundo",
            "inferencia",
            "llama.cpp",
        )

        return any(
            keyword in normalized
            for keyword in keywords
        )

    # ==========================================================
    # REPRESENTACIÓN DEL PROBLEMA
    # ==========================================================

    @staticmethod
    def _problem_text(
        problem: Problem,
    ) -> str:
        """
        Construye una representación textual determinista
        del problema.

        Se utiliza únicamente para activar reglas temáticas.
        """

        parts = [
            problem.description,
            problem.objective or "",
            *problem.constraints,
            *problem.known_facts,
            *problem.unknowns,
            *problem.assumptions,
        ]

        text = " ".join(
            str(part)
            for part in parts
            if part is not None
        )

        return re.sub(
            r"\s+",
            " ",
            text.casefold(),
        ).strip()

    # ==========================================================
    # UTILIDADES
    # ==========================================================

    @staticmethod
    def _lower_first(
        text: str,
    ) -> str:
        """
        Convierte únicamente la primera letra a minúscula.

        Mantiene intacto el resto del texto.

        Ejemplo:

            "La causa del problema"

        →

            "la causa del problema"
        """

        text = str(text).strip()

        if not text:
            return ""

        return text[0].lower() + text[1:]
