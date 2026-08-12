"""
Orquestador central de Mía Isabella.

Flujo:

mensaje
   ↓
memoria episódica
   ↓
contexto
   ↓
Cognitive Core
   ↓
prompt
   ↓
Inference Engine
   ↓
respuesta
   ↓
memoria

El Orchestrator no conoce el modelo concreto.
"""

import logging
from datetime import datetime
from typing import List, Optional

from internal.schemas import (
    Message,
    MessageRole,
    Episode,
)

from contracts.inference import InferenceEngineProtocol

from memory.storage.episodic_repository import (
    EpisodicRepository,
)

from cognition.cognitive_supervisor import (
    CognitiveSupervisor,
)

from cognition.models import Problem


logger = logging.getLogger(__name__)


class Orchestrator:

    def __init__(
        self,
        inference_engine: InferenceEngineProtocol,
        episodic_repo: EpisodicRepository,
        knowledge_extractor=None,
        vector_repo=None,
        cognitive_supervisor: Optional[
            CognitiveSupervisor
        ] = None,
        system_prompt: str = (
            "Eres Mía Isabella, una asistente personal "
            "amigable, útil, responsable y precisa."
        ),
    ):

        self.inference = inference_engine

        self.episodic_repo = episodic_repo

        self.knowledge_extractor = (
            knowledge_extractor
        )

        self.vector_repo = vector_repo

        self.system_prompt = system_prompt

        self.cognitive = (
            cognitive_supervisor
            or CognitiveSupervisor()
        )

    # =========================================================
    # PROCESAMIENTO PRINCIPAL
    # =========================================================

    def process_message(
        self,
        user_message: Message,
    ) -> Message:

        logger.info(
            "Procesando mensaje del usuario"
        )

        # -----------------------------------------------------
        # 1. Recuperar memoria episódica
        # -----------------------------------------------------

        recent_episodes = (
            self.episodic_repo.get_recent_episodes(
                limit=5
            )
        )

        context_str = self._build_context(
            recent_episodes
        )

        # -----------------------------------------------------
        # 2. Construir problema cognitivo
        # -----------------------------------------------------

        problem = Problem(
            description=user_message.content,
            objective=(
                "Responder correctamente al usuario."
            ),
            known_facts=self._extract_context_facts(
                recent_episodes
            ),
            unknowns=[],
            constraints=[],
            assumptions=[],
        )

        # -----------------------------------------------------
        # 3. Cognitive Core
        # -----------------------------------------------------

        try:

            cognitive_result = (
                self.cognitive.solve(problem)
            )

            logger.info(
                "Cognitive Core: "
                "confidence=%.2f "
                "uncertainty=%.2f",
                cognitive_result.confidence,
                cognitive_result.uncertainty,
            )

        except Exception as exc:

            logger.exception(
                "Error en Cognitive Core: %s",
                exc,
            )

            cognitive_result = None

        # -----------------------------------------------------
        # 4. Contexto cognitivo
        # -----------------------------------------------------

        cognitive_context = ""

        if cognitive_result is not None:

            cognitive_context = (
                self._build_cognitive_context(
                    cognitive_result
                )
            )

        # -----------------------------------------------------
        # 5. Construir prompt
        # -----------------------------------------------------

        prompt = self._build_prompt(
            context=context_str,
            user_input=user_message.content,
            cognitive_context=cognitive_context,
        )

        # -----------------------------------------------------
        # 6. Inferencia
        # -----------------------------------------------------

        logger.debug(
            "Solicitando inferencia..."
        )

        try:

            response_text = (
                self.inference.generate(
                    prompt,
                    max_tokens=300,
                    temperature=0.7,
                )
            )

        except Exception as exc:

            logger.exception(
                "Error durante inferencia: %s",
                exc,
            )

            response_text = (
                "Lo siento, no pude generar "
                "una respuesta en este momento."
            )

        if not response_text.strip():

            response_text = (
                "Lo siento, no pude generar "
                "una respuesta en este momento."
            )

        # -----------------------------------------------------
        # 7. Crear mensaje del asistente
        # -----------------------------------------------------

        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response_text.strip(),
            timestamp=datetime.now(),
        )

        # -----------------------------------------------------
        # 8. Guardar episodio
        # -----------------------------------------------------

        episode = Episode(
            messages=[
                user_message,
                assistant_message,
            ],
            start_time=user_message.timestamp,
            end_time=assistant_message.timestamp,
        )

        try:

            self.episodic_repo.save_episode(
                episode
            )

        except Exception as exc:

            logger.exception(
                "No se pudo guardar el episodio: %s",
                exc,
            )

        # -----------------------------------------------------
        # 9. Extraer conocimiento
        # -----------------------------------------------------

        if self.knowledge_extractor:

            try:

                self.knowledge_extractor.extract_and_store(
                    episode
                )

            except Exception as exc:

                logger.error(
                    "Error en extracción de conocimiento: %s",
                    exc,
                )

        return assistant_message

    # =========================================================
    # MEMORIA
    # =========================================================

    def _build_context(
        self,
        episodes: List[Episode],
    ) -> str:

        lines = []

        for episode in reversed(episodes):

            for message in episode.messages:

                if message.role == MessageRole.USER:

                    role = "Usuario"

                else:

                    role = "Mía"

                content = (
                    message.content
                    .strip()
                )

                if not content:
                    continue

                lines.append(
                    f"{role}: {content}"
                )

        # Evitamos un prompt innecesariamente enorme.

        return "\n".join(
            lines[-20:]
        )

    def _extract_context_facts(
        self,
        episodes: List[Episode],
    ) -> List[str]:

        facts = []

        for episode in episodes:

            for message in episode.messages:

                content = (
                    message.content.strip()
                    if message.content
                    else ""
                )

                if content:

                    facts.append(
                        content
                    )

        return facts[-10:]

    # =========================================================
    # CONTEXTO COGNITIVO
    # =========================================================

    def _build_cognitive_context(
        self,
        result,
    ) -> str:

        lines = [
            "=== CONTEXTO COGNITIVO ===",
            (
                f"Confianza: "
                f"{result.confidence:.2f}"
            ),
            (
                f"Incertidumbre: "
                f"{result.uncertainty:.2f}"
            ),
        ]

        if result.needs_more_information:

            lines.append(
                "Se necesita información adicional."
            )

        if result.hypotheses:

            lines.append(
                "\nHipótesis consideradas:"
            )

            for hypothesis in result.hypotheses[:3]:

                lines.append(
                    "- "
                    f"{hypothesis.statement} "
                    f"(confianza="
                    f"{hypothesis.confidence:.2f}, "
                    f"estado="
                    f"{hypothesis.status.value})"
                )

        if result.plan:

            lines.append(
                "\nPlan cognitivo:"
            )

            for step in result.plan.steps[:5]:

                lines.append(
                    f"- {step.action}"
                )

        if result.decision:

            lines.append(
                "\nDecisión cognitiva:"
            )

            lines.append(
                f"- Acción: "
                f"{result.decision.action}"
            )

            lines.append(
                f"- Estado: "
                f"{result.decision.status.value}"
            )

        if result.questions:

            lines.append(
                "\nInformación pendiente:"
            )

            for question in result.questions[:5]:

                lines.append(
                    f"- {question}"
                )

        return "\n".join(lines)

    # =========================================================
    # PROMPT
    # =========================================================

    def _build_prompt(
        self,
        context: str,
        user_input: str,
        cognitive_context: str = "",
    ) -> str:

        prompt = (
            "<|system|>\n"
            f"{self.system_prompt}\n"
        )

        if context:

            prompt += (
                "\nHistorial reciente:\n"
                f"{context}\n"
            )

        if cognitive_context:

            prompt += (
                "\n"
                f"{cognitive_context}\n"
            )

        prompt += (
            "\n"
            "<|user|>\n"
            f"{user_input.strip()}\n"
            "<|assistant|>\n"
        )

        return prompt
