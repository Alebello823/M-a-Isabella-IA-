"""
Controlador de aprendizaje de Mía Isabella.

Responsabilidad
---------------
Gestionar el aprendizaje derivado de experiencias sin modificar
automáticamente el código, el modelo o la configuración del sistema.

Principio fundamental:

    EXPERIENCIA
        ↓
    EVALUACIÓN
        ↓
    EXTRACCIÓN
        ↓
    VALIDACIÓN
        ↓
    MEMORIA

El aprendizaje no modifica directamente el comportamiento del sistema.

El LLM es una fuente de inferencia, no la autoridad del aprendizaje.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


@dataclass
class LearningCandidate:
    """
    Candidato a conocimiento aprendido.

    Todavía NO es conocimiento confirmado.
    """

    content: str

    source: str = "conversation"

    confidence: float = 0.0

    relevance: float = 0.0

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    validated: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


class LearningController:
    """
    Controlador central del aprendizaje.

    No modifica:

        - código
        - modelos
        - configuración
        - herramientas

    Su función es convertir experiencias en candidatos de
    conocimiento y decidir cuáles pueden pasar a memoria.
    """

    def __init__(
        self,
        min_confidence: float = 0.75,
        min_relevance: float = 0.60,
    ):
        self.min_confidence = max(
            0.0,
            min(1.0, float(min_confidence)),
        )

        self.min_relevance = max(
            0.0,
            min(1.0, float(min_relevance)),
        )

        self._pending: List[LearningCandidate] = []

        logger.info(
            "LearningController inicializado: "
            "min_confidence=%.2f min_relevance=%.2f",
            self.min_confidence,
            self.min_relevance,
        )

    # ==========================================================
    # EVALUACIÓN
    # ==========================================================

    def evaluate(
        self,
        content: str,
        confidence: float,
        relevance: float,
        source: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LearningCandidate:
        """
        Evalúa una posible pieza de conocimiento.

        El resultado queda como candidato.
        """

        candidate = LearningCandidate(
            content=content.strip(),
            source=source,
            confidence=max(
                0.0,
                min(1.0, float(confidence)),
            ),
            relevance=max(
                0.0,
                min(1.0, float(relevance)),
            ),
            metadata=metadata or {},
        )

        candidate.validated = self._meets_thresholds(
            candidate
        )

        if candidate.validated:
            logger.info(
                "Candidato de aprendizaje validado: "
                "confidence=%.2f relevance=%.2f",
                candidate.confidence,
                candidate.relevance,
            )
        else:
            logger.debug(
                "Candidato rechazado temporalmente: "
                "confidence=%.2f relevance=%.2f",
                candidate.confidence,
                candidate.relevance,
            )

        self._pending.append(candidate)

        return candidate

    # ==========================================================
    # UMBRALES
    # ==========================================================

    def _meets_thresholds(
        self,
        candidate: LearningCandidate,
    ) -> bool:

        if not candidate.content:
            return False

        if candidate.confidence < self.min_confidence:
            return False

        if candidate.relevance < self.min_relevance:
            return False

        return True

    # ==========================================================
    # CANDIDATOS
    # ==========================================================

    def pending_candidates(
        self,
    ) -> List[LearningCandidate]:
        """
        Devuelve candidatos pendientes.

        Se devuelve una copia para evitar que consumidores externos
        modifiquen directamente el estado interno.
        """

        return list(self._pending)

    def validated_candidates(
        self,
    ) -> List[LearningCandidate]:
        """
        Devuelve solamente candidatos validados.
        """

        return [
            candidate
            for candidate in self._pending
            if candidate.validated
        ]

    # ==========================================================
    # CONFIRMACIÓN
    # ==========================================================

    def confirm(
        self,
        candidate: LearningCandidate,
    ) -> bool:
        """
        Confirma explícitamente un candidato.

        Esta operación no escribe todavía en ninguna base de datos.
        El almacenamiento queda desacoplado.

        Returns:
            True si el candidato puede pasar a memoria.
        """

        if candidate not in self._pending:
            logger.warning(
                "Intento de confirmar candidato desconocido."
            )
            return False

        if not candidate.validated:
            logger.warning(
                "Candidato no validado; no se confirma."
            )
            return False

        logger.info(
            "Candidato confirmado para almacenamiento."
        )

        return True

    # ==========================================================
    # LIMPIEZA
    # ==========================================================

    def discard(
        self,
        candidate: LearningCandidate,
    ) -> bool:
        """
        Elimina un candidato pendiente.
        """

        try:
            self._pending.remove(candidate)

        except ValueError:
            return False

        logger.debug(
            "Candidato de aprendizaje descartado."
        )

        return True

    def clear(
        self,
    ) -> None:
        """
        Limpia candidatos pendientes.
        """

        self._pending.clear()

        logger.debug(
            "Candidatos de aprendizaje limpiados."
        )
