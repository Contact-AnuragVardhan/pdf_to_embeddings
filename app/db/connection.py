from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector

logger = logging.getLogger(__name__)


@contextmanager
def get_connection(database_url: str, *, autocommit: bool = False) -> Iterator[psycopg.Connection]:
    conn = psycopg.connect(database_url, autocommit=autocommit)
    try:
        try:
            register_vector(conn)
        except Exception as exc:
            logger.debug("pgvector type registration skipped/failed: %s", exc)
        yield conn
    finally:
        conn.close()
