# storage/db_connector.py
"""
Conexión única a la base de datos SQLite.
Gestiona la creación de tablas y proporciona métodos genéricos de ejecución.
"""

import sqlite3
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from threading import Lock

from internal.constants import DEFAULT_DB_PATH  # ← corregido

logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    Singleton para manejar la conexión a SQLite.
    Crea las tablas necesarias si no existen.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls, db_path: Optional[Path] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseConnector, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return
        self.db_path = db_path or DEFAULT_DB_PATH  # ← corregido (ya es Path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._initialized = True
        self._create_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """Retorna una conexión, asegurando que sea la misma para el singleton."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10.0
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON;")
        return self._connection

    def _create_tables(self) -> None:
        """Crea todas las tablas necesarias si no existen."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                title TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                summary TEXT,
                importance_score REAL DEFAULT 0.5,
                tags TEXT,
                device_id TEXT,
                emotion_tone TEXT,
                context_snapshot TEXT,
                messages TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_start_time ON episodes(start_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_importance ON episodes(importance_score)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                timestamp TEXT NOT NULL,
                last_reviewed TEXT,
                tags TEXT,
                relation_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_context (
                thread_id TEXT PRIMARY KEY,
                current_topic TEXT,
                user_intent TEXT,
                emotion_tone TEXT,
                messages TEXT,
                active_goal_id TEXT,
                pending_actions TEXT,
                last_interaction TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deadline TEXT,
                progress REAL DEFAULT 0.0,
                notes TEXT,
                parent_goal_id TEXT,
                FOREIGN KEY (parent_goal_id) REFERENCES goals(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id TEXT PRIMARY KEY,
                name TEXT,
                preferred_language TEXT DEFAULT 'es',
                relationship_to_ai TEXT DEFAULT 'user',
                preferences TEXT,
                created_at TEXT NOT NULL,
                last_seen TEXT
            )
        """)

        conn.commit()
        logger.info("Tablas creadas/verificadas en la base de datos.")

    def execute_query(
        self,
        query: str,
        params: Optional[Union[Tuple, Dict[str, Any]]] = None,
        fetch_one: bool = False,
        fetch_all: bool = False
    ) -> Any:
        """Ejecuta una consulta SQL con parámetros y retorna resultados según lo solicitado."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            if fetch_one:
                row = cursor.fetchone()
                return dict(row) if row else None
            elif fetch_all:
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as e:
            logger.error(f"Error en consulta SQL: {e}\nConsulta: {query}\nParámetros: {params}")
            conn.rollback()
            raise

    def execute_many(self, query: str, params_list: List[Union[Tuple, Dict[str, Any]]]) -> None:
        """Ejecuta una consulta parametrizada para múltiples filas (batch)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.executemany(query, params_list)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error en batch SQL: {e}\nConsulta: {query}")
            conn.rollback()
            raise

    def fetch_all(self, query: str, params: tuple = ()) -> list[dict]:
        """Método de conveniencia para obtener todas las filas como diccionarios."""
        conn = self._get_connection()          # ← corregido (antes decía get_connection)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Cierra la conexión si está abierta."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Conexión a la base de datos cerrada.")
    def execute(self, query: str, params: tuple = ()) -> int:
        """Ejecuta una consulta y retorna el lastrowid (para inserciones)."""
        return self.execute_query(query, params)
