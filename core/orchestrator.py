import logging
from datetime import datetime
from typing import List, Optional

from internal.schemas import Message, MessageRole, Episode
from contracts.inference import InferenceEngineProtocol
from memory.storage.episodic_repository import EpisodicRepository

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        inference_engine: InferenceEngineProtocol,
        episodic_repo: EpisodicRepository,
        knowledge_extractor=None,   # opcional
        vector_repo=None,           # opcional, por claridad
        system_prompt: str = "Eres Mía Isabella, una asistente personal amigable y útil."
    ):
        self.inference = inference_engine
        self.episodic_repo = episodic_repo
        self.system_prompt = system_prompt
        self.knowledge_extractor = knowledge_extractor
        # vector_repo ya fue inyectado en episodic_repo desde el kernel

    def process_message(self, user_message: Message) -> Message:
        # Recuperar últimos episodios (contexto)
        recent_episodes = self.episodic_repo.get_recent_episodes(limit=5)
        context_str = self._build_context(recent_episodes)

        # Construir prompt
        prompt = self._build_prompt(context_str, user_message.content)

        # Inferencia
        logger.debug("Solicitando inferencia...")
        response_text = self.inference.generate(prompt, max_tokens=300, temperature=0.7)

        # Crear mensaje de respuesta
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=response_text,
            timestamp=datetime.now()
        )

        # Guardar episodio
        episode = Episode(
            messages=[user_message, assistant_message],
            start_time=user_message.timestamp,
            end_time=assistant_message.timestamp,
        )
        self.episodic_repo.save_episode(episode)

        # Extraer hechos (si el extractor está disponible)
        if self.knowledge_extractor:
            try:
                self.knowledge_extractor.extract_and_store(episode)
                logger.debug("Extracción de conocimiento completada")
            except Exception as e:
                logger.error("Error en extracción de conocimiento: %s", e)

        return assistant_message

    def _build_context(self, episodes: List[Episode]) -> str:
        lines = []
        for ep in episodes:
            for msg in ep.messages:
                role = "Usuario" if msg.role == MessageRole.USER else "Asistente"
                lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def _build_prompt(self, context: str, user_input: str) -> str:
        prompt = f"<|system|>\n{self.system_prompt}\n"
        if context:
            prompt += f"Historial reciente:\n{context}\n"
        prompt += f"<|user|>\n{user_input}\n<|assistant|>\n"
        return prompt
