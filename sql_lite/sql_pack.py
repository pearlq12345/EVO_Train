#!/usr/bin/env python3
"""MySQL helpers for tcp_read_server task storage."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://evodata:cqmygYSDSS123@rm-bp1y7lfvg5u0hxh8a.mysql.rds.aliyuncs.com:3306/evo_data?charset=utf8mb4"
)

def _load_pymysql() -> Any:
    try:
        import pymysql
        from pymysql.cursors import DictCursor
    except ImportError as exc:
        raise RuntimeError("PyMySQL is required for Aliyun RDS/MySQL. Install it with: pip install PyMySQL") from exc
    return pymysql, DictCursor


def _parse_database_url() -> dict[str, Any]:
    if not DATABASE_URL:
        raise RuntimeError("Set EVO_TRAIN_DATABASE_URL or DATABASE_URL to an Aliyun RDS MySQL connection URL.")

    parsed = urlparse(DATABASE_URL)
    if parsed.scheme not in {"mysql", "mysql+pymysql"}:
        raise RuntimeError("Only mysql/mysql+pymysql database URLs are supported.")
    if not parsed.hostname or not parsed.path.strip("/"):
        raise RuntimeError("Database URL must include host and database name.")

    query = parse_qs(parsed.query)
    charset = query.get("charset", ["utf8mb4"])[0]
    return {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username or ""),
        "password": unquote(parsed.password or ""),
        "database": unquote(parsed.path.lstrip("/")),
        "charset": charset,
    }


@contextmanager
def _connect():
    pymysql, DictCursor = _load_pymysql()
    conn = pymysql.connect(
        **_parse_database_url(),
        cursorclass=DictCursor,
        autocommit=False,
    )
    try:
        _init_db(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db(conn: Any) -> None:
    with conn.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_tasks (
                username VARCHAR(255) NOT NULL,
                task_name VARCHAR(255) NOT NULL,
                status VARCHAR(255) NOT NULL DEFAULT '',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (username, task_name),
                INDEX idx_user_tasks_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )


def sql_get_user_all_task(username: str) -> list[dict[str, str]]:
    """Return all tasks for one user in the response format expected by the TCP server."""
    with _connect() as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT task_name, status
            FROM user_tasks
            WHERE username = %s
            ORDER BY created_at ASC, task_name ASC
            """,
            (username,),
        )
        rows = cursor.fetchall()

    return [{"taskName": row["task_name"], "status": row["status"]} for row in rows]


def sql_add_user_task(username: str, task_name: str) -> bool:
    """Add a task for one user. Return False when the task already exists."""
    pymysql, _ = _load_pymysql()
    try:
        with _connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_tasks (username, task_name, status)
                VALUES (%s, %s, '')
                """,
                (username, task_name),
            )
        return True
    except pymysql.err.IntegrityError:
        return False


def sql_delete_user_task(username: str, task_name: str) -> bool:
    """Delete a task for one user. Return True only when a row was deleted."""
    with _connect() as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM user_tasks
            WHERE username = %s AND task_name = %s
            """,
            (username, task_name),
        )
        return cursor.rowcount > 0