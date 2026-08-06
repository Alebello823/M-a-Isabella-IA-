"""
Extrae hechos (sujeto-predicado-objeto) de las conversaciones usando el LLM.
"""
import json
import logging
from typing import List

from internal.schemas import Episode, Message, Fact
from contracts.inference import InferenceEngineProtocol
from memory.storage.semantic_repository import SemanticRepository

logger = logging.getLogger(__name__)

class KnowledgeExtractor:
    PROMPT_TEMPLATE = """
Extrae hechos en formato de tripletas (sujeto, predicado, objeto) de la siguiente conversación.
Responde únicamente con un array JSON, sin explicaciones.
Ejemplo: [{"sujeto":"usuario","predicado":"tiene_novia","objeto":"Ana","confianza":0.98}]

Conversación:
{conversation}

Tripletas:
"""
    def __init__(self, llm_engine: InferenceEngineProtocol, semantic_repo: SemanticRepository):
        self.llm = llm_engine
        self.semantic_repo = semantic_repo

    def extract_and_store(self, episode: Episode):
        # Concatenar solo los mensajes del usuario y del asistente
        texto = "\n".join([
            f"{'Usuario' if m.role.value == 'user' else 'Asistente'}: {m.content}"
            for m in episode.messages if m.role.value in ('user', 'assistant')
        ])
        if not texto.strip():
            return

        prompt = self.PROMPT_TEMPLATE.format(conversation=texto)
        response = self.llm.generate(prompt, max_tokens=200, temperature=0.3)
        try:
            facts_list = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("El LLM no devolvió JSON válido: %s", response)
            return

        for item in facts_list:
            fact = Fact(
                subject=item["sujeto"],
                predicate=item["predicado"],
                object=item["objeto"],
                confidence=item.get("confianza", 0.8),
                source=str(episode.id),
                timestamp=episode.end_time,
            )
            self.semantic_repo.save_fact(fact)
            logger.debug("Hecho guardado: %s - %s - %s", fact.subject, fact.predicate, fact.object)
