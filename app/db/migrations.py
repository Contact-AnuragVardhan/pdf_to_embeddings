from __future__ import annotations

from pathlib import Path

import psycopg


def init_schema(database_url: str, schema_path: Path) -> None:
    sql = schema_path.read_text(encoding="utf-8")
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(sql)
