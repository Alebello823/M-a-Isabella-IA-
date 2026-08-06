# inference/engine.py
import logging
from pathlib import Path
from llama_cpp import Llama

from contracts.inference import InferenceEngineProtocol

logger = logging.getLogger(__name__)

class LlamaCppEngine(InferenceEngineProtocol):
    def __init__(self, model_path: Path, n_ctx: int = 2048, n_threads: int = 4):
        self.model_path = model_path
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        logger.info(f"Cargando modelo desde {model_path}")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        logger.info("Modelo cargado correctamente")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|user|>", "<|assistant|>", "\nUser:", "\nAssistant:"],
        )
        generated = output["choices"][0]["text"].strip()
        return generated
