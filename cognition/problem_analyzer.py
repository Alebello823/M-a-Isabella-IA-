"""
Analizador estructural de problemas de Mía Isabella.

Responsabilidad:

    mensaje del usuario
            ↓
        Problem estructurado

Este módulo NO:
- ejecuta herramientas
- modifica archivos
- toma decisiones finales
- ejecuta acciones
- genera la respuesta final al usuario

Su función es preparar el problema para el Cognitive Core.

Inicialmente utiliza análisis determinista ligero.
Posteriormente podrá incorporar un LLM como apoyo para
extraer información más compleja sin cambiar el contrato.
"""

import logging
import re
from typing import Iterable, List, Optional

from cognition.models import (
    Evidence,
    EvidenceType,
    Problem,
)


logger = logging.getLogger(__name__)


class ProblemAnalyzer:
    """
    Convierte una entrada de usuario y su contexto
    en un Problem estructurado.
    """

    def analyze(
        self,
        user_message: str,
        context_facts: Optional[Iterable[str]] = None,
    ) -> Problem:

        # -----------------------------------------------------
        # Validación de entrada
        # -----------------------------------------------------

        if not isinstance(user_message, str):
            raise TypeError(
                "user_message debe ser un string"
            )

        description = user_message.strip()

        if not description:
            raise ValueError(
                "El mensaje del usuario no puede estar vacío"
            )

        # -----------------------------------------------------
        # Contexto conocido
        # -----------------------------------------------------

        facts = [
            str(item).strip()
            for item in (context_facts or [])
            if str(item).strip()
        ]

        # -----------------------------------------------------
        # Evidencia derivada del contexto
        # -----------------------------------------------------

        evidence = self._build_evidence(
            facts
        )

        # -----------------------------------------------------
        # Análisis estructural
        # -----------------------------------------------------

        unknowns = self._detect_unknowns(
            description
        )

        constraints = self._detect_constraints(
            description
        )

        assumptions = self._detect_assumptions(
            description
        )

        objective = self._infer_objective(
            description
        )

        success_conditions = (
            self._infer_success_conditions(
                description
            )
        )

        # -----------------------------------------------------
        # Construcción del problema
        # -----------------------------------------------------

        problem = Problem(
            description=description,
            objective=objective,
            constraints=constraints,
            known_facts=facts,
            unknowns=unknowns,
            assumptions=assumptions,
            success_conditions=success_conditions,
            resources=[],
            evidence=evidence,
            metadata={
                "source": "user_message",
                "analyzer": "deterministic",
            },
        )

        logger.debug(
            "Problema analizado: %s",
            problem.id,
        )

        return problem

    # =========================================================
    # EVIDENCIA
    # =========================================================

    @staticmethod
    def _build_evidence(
        facts: Iterable[str],
    ) -> List[Evidence]:
        """
        Convierte hechos conocidos del contexto
        en evidencia estructurada.

        Importante:

        La evidencia proveniente del contexto no se considera
        automáticamente un hecho confirmado. Se representa como
        OBSERVATION con confianza moderada.

        Esto evita que el Cognitive Core convierta directamente
        una observación en certeza causal.
        """

        evidence: List[Evidence] = []

        for fact in facts:

            content = str(fact).strip()

            if not content:
                continue

            evidence.append(
                Evidence(
                    content=content,
                    evidence_type=EvidenceType.OBSERVATION,
                    confidence=0.60,
                    source="conversation_context",
                )
            )

        return evidence

    # =========================================================
    # OBJETIVO
    # =========================================================

    @staticmethod
    def _infer_objective(
        text: str,
    ) -> str:

        normalized = text.lower().strip()

        if normalized.startswith(
            (
                "como ",
                "cómo ",
                "que hago",
                "qué hago",
                "puedo ",
                "puedo hacer",
            )
        ):
            return (
                "Determinar la respuesta o acción "
                "más adecuada para la solicitud."
            )

        if any(
            word in normalized
            for word in (
                "error",
                "fallo",
                "problema",
                "no funciona",
                "lento",
                "falla",
            )
        ):
            return (
                "Identificar la causa probable del problema "
                "y determinar una estrategia adecuada."
            )

        return (
            "Responder correctamente a la solicitud "
            "del usuario."
        )

    # =========================================================
    # INCÓGNITAS
    # =========================================================

    @staticmethod
    def _detect_unknowns(
        text: str,
    ) -> List[str]:

        unknowns: List[str] = []

        normalized = text.lower()

        if (
            "por qué" in normalized
            or "porque" in normalized
        ):
            unknowns.append(
                "Determinar la causa del problema."
            )

        if any(
            marker in normalized
            for marker in (
                "cómo",
                "como ",
                "qué hago",
                "que hago",
            )
        ):
            unknowns.append(
                "Determinar qué información adicional "
                "puede ser necesaria para actuar correctamente."
            )

        if "no funciona" in normalized:
            unknowns.append(
                "Determinar qué componente está provocando "
                "el comportamiento observado."
            )

        return ProblemAnalyzer._unique(
            unknowns
        )

    # =========================================================
    # RESTRICCIONES
    # =========================================================

    @staticmethod
    def _detect_constraints(
        text: str,
    ) -> List[str]:

        constraints: List[str] = []

        normalized = text.lower()

        patterns = (
            (
                r"\bsin root\b",
                "No se dispone de privilegios root.",
            ),
            (
                r"\bsin internet\b",
                "No se dispone de conexión a Internet.",
            ),
            (
                r"\bsin acceso\b",
                "El acceso al recurso indicado puede estar limitado.",
            ),
            (
                r"\blimitad[oa]s?\b",
                "Existen recursos o capacidades limitadas.",
            ),
        )

        for pattern, constraint in patterns:

            if re.search(
                pattern,
                normalized,
            ):
                constraints.append(
                    constraint
                )

        return ProblemAnalyzer._unique(
            constraints
        )

    # =========================================================
    # SUPUESTOS
    # =========================================================

    @staticmethod
    def _detect_assumptions(
        text: str,
    ) -> List[str]:

        assumptions: List[str] = []

        normalized = text.lower()

        if any(
            phrase in normalized
            for phrase in (
                "creo que",
                "pienso que",
                "supongo que",
                "parece que",
            )
        ):
            assumptions.append(
                "La interpretación proporcionada por "
                "el usuario debe verificarse."
            )

        return ProblemAnalyzer._unique(
            assumptions
        )

    # =========================================================
    # CONDICIONES DE ÉXITO
    # =========================================================

    @staticmethod
    def _infer_success_conditions(
        text: str,
    ) -> List[str]:

        normalized = text.lower()

        if any(
            word in normalized
            for word in (
                "error",
                "fallo",
                "problema",
                "no funciona",
                "lento",
                "falla",
            )
        ):
            return [
                "Identificar una causa suficientemente "
                "sustentada por evidencia.",
                "Evitar afirmar causalidad sin verificación.",
            ]

        return [
            "Proporcionar una respuesta coherente "
            "y sustentada por la información disponible.",
        ]

    # =========================================================
    # UTILIDADES
    # =========================================================

    @staticmethod
    def _unique(
        values: Iterable[str],
    ) -> List[str]:

        result: List[str] = []
        seen = set()

        for value in values:

            normalized = value.strip()

            if not normalized:
                continue

            key = normalized.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result
