# memory/storage/episodic_repository.py
"""
Implementación del repositorio de episodios usando SQLite.
Cumple con el contrato MemoryProtocol (parte episódica).
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from internal.schemas import Episode, Message, EmotionTone
from storage.db_connector import DatabaseConnector

logger = logging.getLogger(__name__)

class EpisodicRepository:
    """Repositorio para almacenar y recuperar episodios (conversaciones/eventos)."""

    def __init__(self):
        self.db = DatabaseConnector()
        self._table = "episodes"

    # ---------- Métodos auxiliares de serialización ----------

    def _serialize_episode(self, episode: Episode) -> Dict[str, Any]:
        """Convierte un objeto Episode a un diccionario para SQLite."""
        return {
            "id": str(episode.id),
            "title": episode.title,
            "start_time": episode.start_time.isoformat(),
            "end_time": episode.end_time.isoformat() if episode.end_time else None,
            "summary": episode.summary,
            "importance_score": episode.importance_score,
            "tags": json.dumps(episode.tags) if episode.tags else "[]",
            "device_id": episode.device_id,
            "emotion_tone": episode.emotion_tone.value if episode.emotion_tone else EmotionTone.NEUTRAL.value,
            "context_snapshot": json.dumps(episode.context_snapshot) if episode.context_snapshot else "{}",
            "messages": json.dumps([
                {
                    "id": str(m.id),
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "metadata": m.metadata
                }
                for m in episode.messages
            ]) if episode.messages else "[]",
        }

    def _deserialize_episode(self, row: Dict[str, Any]) -> Episode:
        """Convierte una fila de la base de datos a un objeto Episode."""
        messages_data = json.loads(row["messages"]) if row["messages"] else []
        messages = []
        for m in messages_data:
            msg = Message(
                id=UUID(m["id"]),
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]),
                metadata=m.get("metadata", {})
            )
            messages.append(msg)

        return Episode(
            id=UUID(row["id"]),
            title=row["title"],
            messages=messages,
            start_time=datetime.fromisoformat(row["start_time"]),
            end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
            summary=row["summary"],
            importance_score=row["importance_score"] or 0.5,
            tags=json.loads(row["tags"]) if row["tags"] else [],
            device_id=row["device_id"],
            emotion_tone=EmotionTone(row["emotion_tone"]) if row["emotion_tone"] else EmotionTone.NEUTRAL,
            context_snapshot=json.loads(row["context_snapshot"]) if row["context_snapshot"] else {},
        )

    # ---------- Métodos públicos (parte del contrato) ----------

    def save_episode(self, episode: Episode) -> str:
        """Guarda un episodio en la base de datos."""
        data = self._serialize_episode(episode)
        query = f"""
            INSERT OR REPLACE INTO {self._table} (
                id, title, start_time, end_time, summary, importance_score,
                tags, device_id, emotion_tone, context_snapshot, messages
            ) VALUES (
                :id, :title, :start_time, :end_time, :summary, :importance_score,
                :tags, :device_id, :emotion_tone, :context_snapshot, :messages
            )
        """
        self.db.execute(query, data)
        logger.debug(f"Episodio guardado: {episode.id}")
        return str(episode.id)

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Recupera un episodio por su ID."""
        query = f"SELECT * FROM {self._table} WHERE id = ?"
        row = self.db.execute_query(query, (episode_id,), fetch_one=True)
        if row:
            return self._deserialize_episode(row)
        return None

    def get_recent_episodes(self, limit: int = 10, since: Optional[str] = None) -> List[Episode]:
        """Obtiene los episodios más recientes (por start_time)."""
        query = f"SELECT * FROM {self._table}"
        params = {}
        if since:
            query += " WHERE start_time > :since"
            params["since"] = since
        query += " ORDER BY start_time DESC LIMIT :limit"
        params["limit"] = limit
        rows = self.db.execute_query(query, params, fetch_all=True)
        return [self._deserialize_episode(row) for row in rows]

    def search_episodes_by_content(self, query_text: str, limit: int = 5) -> List[Episode]:
        """Búsqueda simple por contenido en los mensajes (LIKE sobre JSON)."""
        search_pattern = f"%{query_text}%"
        sql = f"""
            SELECT * FROM {self._table}
            WHERE messages LIKE ?
            ORDER BY start_time DESC
            LIMIT ?
        """
        rows = self.db.execute_query(sql, (search_pattern, limit), fetch_all=True)
        return [self._deserialize_episode(row) for row in rows]

    def delete_episode(self, episode_id: str) -> bool:
        """Elimina un episodio por ID. Retorna True si se eliminó."""
        query = f"DELETE FROM {self._table} WHERE id = ?"
        lastrowid = self.db.execute(query, (episode_id,))
        return lastrowid is not None and lastrowid != 0

    def get_episodes_by_time_range(self, start: datetime, end: datetime) -> List[Episode]:
        """Recupera episodios dentro de un rango de tiempo."""
        query = f"""
            SELECT * FROM {self._table}
            WHERE start_time >= :start AND start_time <= :end
            ORDER BY start_time ASC
        """
        params = {"start": start.isoformat(), "end": end.isoformat()}
        rows = self.db.execute_query(query, params, fetch_all=True)
        return [self._deserialize_episode(row) for row in rows]

    def update_importance(self, episode_id: str, new_score: float) -> None:
        """Actualiza la puntuación de importancia de un episodio."""
        query = f"UPDATE {self._table} SET importance_score = ? WHERE id = ?"
        self.db.execute(query, (new_score, episode_id))

    def get_all_episodes(self, limit: Optional[int] = None) -> List[Episode]:
        """Retorna todos los episodios (útil para migraciones/backups)."""
        query = f"SELECT * FROM {self._table} ORDER BY start_time DESC"
        if limit:
            query += f" LIMIT {limit}"
        rows = self.db.execute_query(query, fetch_all=True)
        return [self._deserialize_episode(row) for row in rows]
    # Dentro de EpisodicRepository, al inicio de la clase, añade:
    def __init__(self, vector_repo=None):
        super().__init__()  # o llama al __init__ original que solo hace self.db = DatabaseConnector()
        self.vector_repo = vector_repo

# Y reemplaza el método search_episodes_by_content por:
    def search_episodes_by_content(self, query_text: str, limit: int = 5) -> List[Episode]:
        # Si hay repositorio vectorial, intentar búsqueda semántica primero
        if self.vector_repo:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = model.encode(query_text)
                neighbors = self.vector_repo.search(embedding, limit)
                if neighbors:
                    episodes = []
                    for n in neighbors:
                        # Suponiendo que el id en FAISS coincide con el episode_id (requiere mapeo)
                        # En una versión real, habría que guardar el mapeo. Por ahora, caemos a LIKE.
                        pass
                    # Por simplicidad, caeremos a búsqueda clásica.
            except Exception as e:
                logger.warning("Falló búsqueda vectorial, usando LIKE: %s", e)

        # Búsqueda clásica por LIKE en la columna messages
        search_pattern = f"%{query_text}%"
        sql = f"""
            SELECT * FROM {self._table}
            WHERE messages LIKE ?
            ORDER BY start_time DESC
            LIMIT ?
        """
        rows = self.db.execute_query(sql, (search_pattern, limit), fetch_all=True)
        return [self._deserialize_episode(row) for row in rows]

    def count_episodes(self) -> int:
        """Retorna el número total de episodios."""
        query = f"SELECT COUNT(*) as count FROM {self._table}"
        result = self.db.execute_query(query, fetch_one=True)
        return result["count"] if result else 0
