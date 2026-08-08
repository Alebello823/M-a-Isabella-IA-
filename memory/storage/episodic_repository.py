"""
Repositorio de memoria episódica de Mía Isabella.

Responsabilidades:
- Guardar episodios en SQLite.
- Recuperar episodios.
- Buscar conversaciones.
- Gestionar importancia.
- Mantener compatibilidad futura con memoria vectorial.

El repositorio debe seguir funcionando aunque el vector_repo
no esté instalado o disponible.
"""

import json
import logging

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from internal.schemas import Episode, Message, EmotionTone
from storage.db_connector import DatabaseConnector


logger = logging.getLogger(__name__)


class EpisodicRepository:
    """Repositorio SQLite para memoria episódica."""

    TABLE = "episodes"

    def __init__(self, vector_repo=None):
        """
        Inicializa el repositorio.

        vector_repo es opcional. Si no existe, la memoria episódica
        continúa funcionando normalmente utilizando SQLite.
        """

        # IMPORTANTE:
        # No usar super().__init__() aquí porque esta clase no hereda
        # de un repositorio que necesite inicialización.
        self.db = DatabaseConnector()
        self._table = self.TABLE
        self.vector_repo = vector_repo

        logger.debug(
            "EpisodicRepository inicializado: table=%s",
            self._table,
        )

    # ============================================================
    # SERIALIZACIÓN
    # ============================================================

    def _serialize_episode(
        self,
        episode: Episode,
    ) -> Dict[str, Any]:
        """Convierte Episode en datos compatibles con SQLite."""

        return {
            "id": str(episode.id),
            "title": episode.title,
            "start_time": episode.start_time.isoformat(),
            "end_time": (
                episode.end_time.isoformat()
                if episode.end_time
                else None
            ),
            "summary": episode.summary,
            "importance_score": episode.importance_score,
            "tags": (
                json.dumps(episode.tags)
                if episode.tags
                else "[]"
            ),
            "device_id": episode.device_id,
            "emotion_tone": (
                episode.emotion_tone.value
                if episode.emotion_tone
                else EmotionTone.NEUTRAL.value
            ),
            "context_snapshot": (
                json.dumps(episode.context_snapshot)
                if episode.context_snapshot
                else "{}"
            ),
            "messages": json.dumps(
                [
                    {
                        "id": str(message.id),
                        "role": message.role.value
                        if hasattr(message.role, "value")
                        else str(message.role),
                        "content": message.content,
                        "timestamp": message.timestamp.isoformat(),
                        "metadata": message.metadata,
                    }
                    for message in episode.messages
                ]
                if episode.messages
                else []
            ),
        }

    def _deserialize_episode(
        self,
        row: Dict[str, Any],
    ) -> Episode:
        """Convierte una fila SQLite en Episode."""

        messages_data = (
            json.loads(row["messages"])
            if row["messages"]
            else []
        )

        messages = []

        for item in messages_data:
            message = Message(
                id=UUID(item["id"]),
                role=item["role"],
                content=item["content"],
                timestamp=datetime.fromisoformat(
                    item["timestamp"]
                ),
                metadata=item.get("metadata", {}),
            )

            messages.append(message)

        emotion_value = row["emotion_tone"]

        try:
            emotion_tone = (
                EmotionTone(emotion_value)
                if emotion_value
                else EmotionTone.NEUTRAL
            )
        except (ValueError, TypeError):
            emotion_tone = EmotionTone.NEUTRAL

        return Episode(
            id=UUID(row["id"]),
            title=row["title"],
            messages=messages,
            start_time=datetime.fromisoformat(
                row["start_time"]
            ),
            end_time=(
                datetime.fromisoformat(row["end_time"])
                if row["end_time"]
                else None
            ),
            summary=row["summary"],
            importance_score=(
                row["importance_score"]
                if row["importance_score"] is not None
                else 0.5
            ),
            tags=(
                json.loads(row["tags"])
                if row["tags"]
                else []
            ),
            device_id=row["device_id"],
            emotion_tone=emotion_tone,
            context_snapshot=(
                json.loads(row["context_snapshot"])
                if row["context_snapshot"]
                else {}
            ),
        )

    # ============================================================
    # GUARDAR
    # ============================================================

    def save_episode(self, episode: Episode) -> str:
        """Guarda o reemplaza un episodio."""

        data = self._serialize_episode(episode)

        query = f"""
            INSERT OR REPLACE INTO {self._table} (
                id,
                title,
                start_time,
                end_time,
                summary,
                importance_score,
                tags,
                device_id,
                emotion_tone,
                context_snapshot,
                messages
            )
            VALUES (
                :id,
                :title,
                :start_time,
                :end_time,
                :summary,
                :importance_score,
                :tags,
                :device_id,
                :emotion_tone,
                :context_snapshot,
                :messages
            )
        """

        self.db.execute(query, data)

        logger.debug(
            "Episodio guardado: %s",
            episode.id,
        )

        return str(episode.id)

    # ============================================================
    # RECUPERAR UNO
    # ============================================================

    def get_episode(
        self,
        episode_id: str,
    ) -> Optional[Episode]:
        """Recupera un episodio por ID."""

        query = f"""
            SELECT *
            FROM {self._table}
            WHERE id = ?
        """

        row = self.db.execute_query(
            query,
            (episode_id,),
            fetch_one=True,
        )

        if row:
            return self._deserialize_episode(row)

        return None

    # ============================================================
    # EPISODIOS RECIENTES
    # ============================================================

    def get_recent_episodes(
        self,
        limit: int = 10,
        since: Optional[str] = None,
    ) -> List[Episode]:
        """Obtiene los episodios más recientes."""

        query = f"""
            SELECT *
            FROM {self._table}
        """

        params = {}

        if since:
            query += """
                WHERE start_time > :since
            """
            params["since"] = since

        query += """
            ORDER BY start_time DESC
            LIMIT :limit
        """

        params["limit"] = int(limit)

        rows = self.db.execute_query(
            query,
            params,
            fetch_all=True,
        )

        return [
            self._deserialize_episode(row)
            for row in rows
        ]

    # ============================================================
    # BÚSQUEDA
    # ============================================================

    def search_episodes_by_content(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Episode]:
        """
        Busca episodios.

        Actualmente SQLite LIKE es el método principal.
        Si posteriormente se conecta vector_repo, se puede
        ampliar sin romper esta función.
        """

        # --------------------------------------------------------
        # Intento opcional de memoria vectorial.
        # --------------------------------------------------------

        if self.vector_repo is not None:

            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )

                model = SentenceTransformer(
                    "all-MiniLM-L6-v2"
                )

                embedding = model.encode(
                    query_text
                )

                neighbors = self.vector_repo.search(
                    embedding,
                    limit,
                )

                # Todavía no utilizamos directamente los vecinos
                # porque necesitamos un mapeo estable:
                #
                # FAISS ID -> episode_id
                #
                # Por ahora usamos SQLite como fallback seguro.

                if neighbors:
                    logger.debug(
                        "Memoria vectorial encontró %d vecinos; "
                        "usando SQLite hasta disponer del mapeo.",
                        len(neighbors),
                    )

            except Exception as exc:
                logger.warning(
                    "Falló búsqueda vectorial; usando SQLite: %s",
                    exc,
                )

        # --------------------------------------------------------
        # Búsqueda SQLite
        # --------------------------------------------------------

        search_pattern = f"%{query_text}%"

        query = f"""
            SELECT *
            FROM {self._table}
            WHERE messages LIKE ?
            ORDER BY start_time DESC
            LIMIT ?
        """

        rows = self.db.execute_query(
            query,
            (search_pattern, int(limit)),
            fetch_all=True,
        )

        return [
            self._deserialize_episode(row)
            for row in rows
        ]

    # ============================================================
    # ELIMINAR
    # ============================================================

    def delete_episode(
        self,
        episode_id: str,
    ) -> bool:
        """Elimina un episodio."""

        query = f"""
            DELETE FROM {self._table}
            WHERE id = ?
        """

        result = self.db.execute(
            query,
            (episode_id,),
        )

        return result is not None and result != 0

    # ============================================================
    # RANGO TEMPORAL
    # ============================================================

    def get_episodes_by_time_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Episode]:
        """Obtiene episodios dentro de un intervalo temporal."""

        query = f"""
            SELECT *
            FROM {self._table}
            WHERE start_time >= :start
              AND start_time <= :end
            ORDER BY start_time ASC
        """

        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
        }

        rows = self.db.execute_query(
            query,
            params,
            fetch_all=True,
        )

        return [
            self._deserialize_episode(row)
            for row in rows
        ]

    # ============================================================
    # IMPORTANCIA
    # ============================================================

    def update_importance(
        self,
        episode_id: str,
        new_score: float,
    ) -> None:
        """Actualiza la importancia de un episodio."""

        score = max(
            0.0,
            min(1.0, float(new_score)),
        )

        query = f"""
            UPDATE {self._table}
            SET importance_score = ?
            WHERE id = ?
        """

        self.db.execute(
            query,
            (score, episode_id),
        )

    # ============================================================
    # TODOS
    # ============================================================

    def get_all_episodes(
        self,
        limit: Optional[int] = None,
    ) -> List[Episode]:
        """Obtiene episodios almacenados."""

        query = f"""
            SELECT *
            FROM {self._table}
            ORDER BY start_time DESC
        """

        if limit is not None:
            query += f" LIMIT {int(limit)}"

        rows = self.db.execute_query(
            query,
            fetch_all=True,
        )

        return [
            self._deserialize_episode(row)
            for row in rows
        ]

    # ============================================================
    # CONTADOR
    # ============================================================

    def count_episodes(self) -> int:
        """Devuelve el número total de episodios."""

        query = f"""
            SELECT COUNT(*) AS count
            FROM {self._table}
        """

        result = self.db.execute_query(
            query,
            fetch_one=True,
        )

        if result:
            return int(result["count"])

        return 0
