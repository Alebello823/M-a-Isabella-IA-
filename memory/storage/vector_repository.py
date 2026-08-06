"""
Repositorio vectorial con FAISS y modelo ligero de embeddings.
"""
import logging
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorRepository:
    def __init__(self, index_path: str = None, dimension: int = 384):
        self.index_path = index_path or "data/vectors/episodes.faiss"
        self.dimension = dimension
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = self._load_or_create_index()
        # Para mapear posición interna -> episode_id (lo guardaremos en una tabla auxiliar)
        # Aquí por simplicidad asumiremos que el id es el índice; en producción usaríamos BD.

    def _load_or_create_index(self):
        if os.path.exists(self.index_path):
            logger.info("Cargando índice FAISS existente")
            return faiss.read_index(self.index_path)
        else:
            logger.info("Creando nuevo índice FAISS")
            return faiss.IndexFlatL2(self.dimension)

    def add_embedding(self, doc_id: str, embedding: list, metadata: dict = None):
        vector = np.array([embedding], dtype=np.float32)
        self.index.add(vector)
        # Guardar el índice para no perderlo
        faiss.write_index(self.index, self.index_path)

    def search(self, query_embedding: list, top_k: int = 5):
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                results.append({"id": str(idx), "score": float(dist)})
        return results

    def clear(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        faiss.write_index(self.index, self.index_path)
