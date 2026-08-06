"""
Implementación del repositorio semántico para guardar y consultar hechos.
Cumple con MemoryProtocol (parte semántica).
"""
import logging
from typing import Optional, List, Dict, Any
from internal.schemas import Fact
from storage.db_connector import DatabaseConnector

logger = logging.getLogger(__name__)

class SemanticRepository:
    def __init__(self):
        self.db = DatabaseConnector()
        self.table = "facts"

    def save_fact(self, fact: Fact) -> str:
        query = f"""
            INSERT INTO {self.table}
            (id, subject, predicate, object, confidence, source, timestamp, tags, relation_type)
            VALUES (:id, :subject, :predicate, :object, :confidence, :source, :timestamp, :tags, :relation_type)
        """
        # Convertir Fact a diccionario con nombres de columna
        fact_dict = {
            "id": str(fact.id),
            "subject": fact.subject,
            "predicate": fact.predicate,
            "object": fact.object,
            "confidence": fact.confidence,
            "source": fact.source,
            "timestamp": fact.timestamp.isoformat() if fact.timestamp else None,
            "tags": ",".join(fact.tags) if fact.tags else None,
            "relation_type": fact.relation_type,
        }
        self.db.execute_query(query, fact_dict)
        return str(fact.id)

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        row = self.db.execute_query(
            f"SELECT * FROM {self.table} WHERE id = :id",
            {"id": fact_id},
            fetch_one=True
        )
        if row:
            return self._row_to_fact(row)
        return None

    def get_facts_about(self, subject: str, predicate: Optional[str] = None) -> List[Fact]:
        query = f"SELECT * FROM {self.table} WHERE subject = :subject"
        params = {"subject": subject}
        if predicate:
            query += " AND predicate = :predicate"
            params["predicate"] = predicate
        rows = self.db.execute_query(query, params, fetch_all=True)
        return [self._row_to_fact(r) for r in rows]

    def search_facts(self, query_text: str, limit: int = 10) -> List[Fact]:
        # Búsqueda simple por LIKE en subject u object (podría mejorarse con vectores)
        pattern = f"%{query_text}%"
        rows = self.db.execute_query(
            f"""SELECT * FROM {self.table}
                WHERE subject LIKE :p OR object LIKE :p
                LIMIT :limit""",
            {"p": pattern, "limit": limit},
            fetch_all=True
        )
        return [self._row_to_fact(r) for r in rows]

    def _row_to_fact(self, row: Dict[str, Any]) -> Fact:
        from datetime import datetime
        return Fact(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            confidence=row["confidence"],
            source=row["source"],
            timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
            tags=row["tags"].split(",") if row["tags"] else [],
            relation_type=row["relation_type"],
        )
