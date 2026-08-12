"""
Repositorio de memoria episódica de Mía Isabella.

Responsabilidades:

- Guardar episodios en SQLite.
- Recuperar episodios.
- Buscar conversaciones.
- Gestionar importancia.
- Recuperar episodios por rango temporal.
- Mantener una interfaz preparada para memoria vectorial.

IMPORTANTE:

La memoria episódica NO depende de:

- Qwen.
- Phi.
- ningún LLM.
- sentence-transformers.
- FAISS.

La memoria vectorial puede existir como una capa adicional,
pero SQLite continúa siendo la fuente primaria.
"""

import json
import logging

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from internal.schemas import Episode, Message, EmotionTone
from storage.db_connector import DatabaseConnector


logger = logging.getLogger(__name__)


class EpisodicRepository:
    """
    Repositorio SQLite para memoria episódica.
    """

    TABLE = "episodes"

    def __init__(self, vector_repo=None):
        """
        Inicializa el repositorio.

        vector_repo se conserva como extensión futura, pero la
        memoria episódica no depende de él.
        """

        self.db = DatabaseConnector()

        self._table = self.TABLE

        self.vector_repo = vector_repo

        logger.debug(
            "EpisodicRepository inicializado: table=%s",
            self._table,
        )

    # =============================================================
    # SERIALIZACIÓN
    # =============================================================

    def _serialize_episode(
        self,
        episode: Episode,
    ) -> Dict[str, Any]:
        """
        Convierte Episode en datos compatibles con SQLite.
        """

        return {
            "id": str(episode.id),

            "title": episode.title,

            "start_time": (
                episode.start_time.isoformat()
            ),

            "end_time": (
                episode.end_time.isoformat()
                if episode.end_time
                else None
            ),

            "summary": episode.summary,

            "importance_score": (
                episode.importance_score
            ),

            "tags": (
                json.dumps(
                    episode.tags,
                    ensure_ascii=False,
                )
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
                json.dumps(
                    episode.context_snapshot,
                    ensure_ascii=False,
                )
                if episode.context_snapshot
                else "{}"
            ),

            "messages": json.dumps(
                [
                    {
                        "id": str(message.id),

                        "role": (
                            message.role.value
                            if hasattr(
                                message.role,
                                "value",
                            )
                            else str(message.role)
                        ),

                        "content": message.content,

                        "timestamp": (
                            message.timestamp.isoformat()
                        ),

                        "metadata": message.metadata,
                    }
                    for message in episode.messages
                ],
                ensure_ascii=False,
            ),
        }

    # =============================================================
    # DESERIALIZACIÓN
    # =============================================================

    def _deserialize_episode(
        self,
        row: Dict[str, Any],
    ) -> Episode:
        """
        Convierte una fila SQLite en Episode.
        """

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

                metadata=item.get(
                    "metadata",
                    {},
                ),
            )

            messages.append(message)

        # ---------------------------------------------------------
        # EMOCIÓN
        # ---------------------------------------------------------

        emotion_value = row["emotion_tone"]

        try:
            emotion_tone = (
                EmotionTone(emotion_value)
                if emotion_value
                else EmotionTone.NEUTRAL
            )

        except (
            ValueError,
            TypeError,
        ):
            emotion_tone = EmotionTone.NEUTRAL

        # ---------------------------------------------------------
        # TAGS
        # ---------------------------------------------------------

        try:
            tags = (
                json.loads(row["tags"])
                if row["tags"]
                else []
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            tags = []

        # ---------------------------------------------------------
        # CONTEXTO
        # ---------------------------------------------------------

        try:
            context_snapshot = (
                json.loads(
                    row["context_snapshot"]
                )
                if row["context_snapshot"]
                else {}
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            context_snapshot = {}

        # ---------------------------------------------------------
        # EPISODIO
        # ---------------------------------------------------------

        return Episode(
            id=UUID(row["id"]),

            title=row["title"],

            messages=messages,

            start_time=datetime.fromisoformat(
                row["start_time"]
            ),

            end_time=(
                datetime.fromisoformat(
                    row["end_time"]
                )
                if row["end_time"]
                else None
            ),

            summary=row["summary"],

            importance_score=(
                row["importance_score"]
                if row["importance_score"] is not None
                else 0.5
            ),

            tags=tags,

            device_id=row["device_id"],

            emotion_tone=emotion_tone,

            context_snapshot=context_snapshot,
        )

    # =============================================================
    # GUARDAR
    # =============================================================

    def save_episode(
        self,
        episode: Episode,
    ) -> str:
        """
        Guarda o reemplaza un episodio.
        """

        data = self._serialize_episode(
            episode
        )

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

        self.db.execute(
            query,
            data,
        )

        logger.debug(
            "Episodio guardado: %s",
            episode.id,
        )

        return str(episode.id)

    # =============================================================
    # RECUPERAR UNO
    # =============================================================

    def get_episode(
        self,
        episode_id: str,
    ) -> Optional[Episode]:
        """
        Recupera un episodio por ID.
        """

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
            return self._deserialize_episode(
                row
            )

        return None

    # =============================================================
    # EPISODIOS RECIENTES
    # =============================================================

    def get_recent_episodes(
        self,
        limit: int = 10,
        since: Optional[str] = None,
    ) -> List[Episode]:
        """
        Obtiene los episodios más recientes.
        """

        limit = max(
            1,
            int(limit),
        )

        query = f"""
            SELECT *
            FROM {self._table}
        """

        params: Dict[str, Any] = {}

        if since:

            query += """
                WHERE start_time > :since
            """

            params["since"] = str(since)

        query += """
            ORDER BY start_time DESC
            LIMIT :limit
        """

        params["limit"] = limit

        rows = self.db.execute_query(
            query,
            params,
            fetch_all=True,
        )

        return [
            self._deserialize_episode(row)
            for row in rows
        ]

    # =============================================================
    # BÚSQUEDA
    # =============================================================

    def search_episodes_by_content(
        self,
        query_text: str,
        limit: int = 5,
    ) -> List[Episode]:
        """
        Busca episodios mediante SQLite.

        La memoria episódica no carga modelos de embeddings.

        Se buscan coincidencias en:

        - mensajes
        - resumen
        - título
        - etiquetas
        """

        if not query_text or not query_text.strip():
            return []

        limit = max(
            1,
            int(limit),
        )

        search_pattern = (
            f"%{query_text.strip()}%"
        )

        query = f"""
            SELECT *
            FROM {self._table}
            WHERE messages LIKE ?
               OR summary LIKE ?
               OR title LIKE ?
               OR tags LIKE ?
            ORDER BY start_time DESC
            LIMIT ?
        """

        rows = self.db.execute_query(
            query,
            (
                search_pattern,
                search_pattern,
                search_pattern,
                search_pattern,
                limit,
            ),
            fetch_all=True,
        )

        return [
            self._deserialize_episode(row)
            for row in rows
        ]

    # =============================================================
    # ELIMINAR
    # =============================================================

    def delete_episode(
        self,
        episode_id: str,
    ) -> bool:
        """
        Elimina un episodio.
        """

        query = f"""
            DELETE FROM {self._table}
            WHERE id = ?
        """

        result = self.db.execute(
            query,
            (episode_id,),
        )

        return (
            result is not None
            and result != 0
        )

    # =============================================================
    # RANGO TEMPORAL
    # =============================================================

    def get_episodes_by_time_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[Episode]:
        """
        Obtiene episodios dentro de un intervalo temporal.
        """

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

    # =============================================================
    # IMPORTANCIA
    # =============================================================

    def update_importance(
        self,
        episode_id: str,
        new_score: float,
    ) -> None:
        """
        Actualiza la importancia de un episodio.

        El valor siempre queda entre 0 y 1.
        """

        score = max(
            0.0,
            min(
                1.0,
                float(new_score),
            ),
        )

        query = f"""
            UPDATE {self._table}
            SET importance_score = ?
            WHERE id = ?
        """

        self.db.execute(
            query,
            (
                score,
                episode_id,
            ),
        )

    # =============================================================
    # TODOS
    # =============================================================

    def get_all_episodes(
        self,
        limit: Optional[int] = None,
    ) -> List[Episode]:
        """
        Obtiene episodios almacenados.
        """

        query = f"""
            SELECT *
            FROM {self._table}
            ORDER BY start_time DESC
        """

        params = ()

        if limit is not None:

            safe_limit = max(
                1,
                int(limit),
            )

            query += """
                LIMIT ?
            """

            params = (
                safe_limit,
            )

        rows = self.db.execute_query(
            query,
            params,
            fetch_all=True,
        )

        return [
            self._deserialize_episode(row)
            for row in rows
        ]

    # =============================================================
    # CONTADOR
    # =============================================================

    def count_episodes(self) -> int:
        """
        Devuelve el número total de episodios.
        """

        query = f"""
            SELECT COUNT(*) AS count
            FROM {self._table}
        """

        result = self.db.execute_query(
            query,
            fetch_one=True,
        )

        if result:
            return int(
                result["count"]
            )

        return 0
