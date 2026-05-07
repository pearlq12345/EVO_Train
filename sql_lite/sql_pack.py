#!/usr/bin/env python3
"""MySQL helpers for tcp_read_server task storage."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

DATABASE_URL = os.environ.get(
    "EVO_TRAIN_DATABASE_URL",
    os.environ.get(
        "DATABASE_URL",
    "mysql+pymysql://evodata:cqmygYSDSS123@rm-bp1y7lfvg5u0hxh8a.mysql.rds.aliyuncs.com:3306/evo_data?charset=utf8mb4"
    ),
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
                provider VARCHAR(255) NOT NULL DEFAULT '',
                remote_job_id VARCHAR(255) NOT NULL DEFAULT '',
                checkpoint_path VARCHAR(1024) NOT NULL DEFAULT '',
                dataset_path VARCHAR(1024) NOT NULL DEFAULT '',
                last_error VARCHAR(1024) NOT NULL DEFAULT '',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                PRIMARY KEY (username, task_name),
                INDEX idx_user_tasks_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        _ensure_columns(cursor)


def _ensure_columns(cursor: Any) -> None:
    cursor.execute("SHOW COLUMNS FROM user_tasks")
    existing_columns = {row["Field"] for row in cursor.fetchall()}
    required_columns = {
        "provider": "VARCHAR(255) NOT NULL DEFAULT ''",
        "remote_job_id": "VARCHAR(255) NOT NULL DEFAULT ''",
        "checkpoint_path": "VARCHAR(1024) NOT NULL DEFAULT ''",
        "dataset_path": "VARCHAR(1024) NOT NULL DEFAULT ''",
        "last_error": "VARCHAR(1024) NOT NULL DEFAULT ''",
        "updated_at": "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
    }
    for column_name, column_sql in required_columns.items():
        if column_name in existing_columns:
            continue
        cursor.execute(f"ALTER TABLE user_tasks ADD COLUMN {column_name} {column_sql}")


def _timestamp_to_string(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _row_to_task(row: dict[str, Any]) -> dict[str, str]:
    return {
        "taskName": str(row.get("task_name") or ""),
        "status": str(row.get("status") or ""),
        "provider": str(row.get("provider") or ""),
        "jobId": str(row.get("remote_job_id") or ""),
        "checkpointPath": str(row.get("checkpoint_path") or ""),
        "datasetPath": str(row.get("dataset_path") or ""),
        "error": str(row.get("last_error") or ""),
        "createdAt": _timestamp_to_string(row.get("created_at")),
        "updatedAt": _timestamp_to_string(row.get("updated_at")),
    }


def sql_get_user_all_task(username: str) -> list[dict[str, str]]:
    """Return all tasks for one user in the response format expected by the TCP server."""
    with _connect() as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                task_name,
                status,
                provider,
                remote_job_id,
                checkpoint_path,
                dataset_path,
                last_error,
                created_at,
                updated_at
            FROM user_tasks
            WHERE username = %s
            ORDER BY created_at ASC, task_name ASC
            """,
            (username,),
        )
        rows = cursor.fetchall()

    return [_row_to_task(row) for row in rows]


def sql_get_user_task(username: str, task_name: str) -> dict[str, str] | None:
    with _connect() as conn, conn.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                task_name,
                status,
                provider,
                remote_job_id,
                checkpoint_path,
                dataset_path,
                last_error,
                created_at,
                updated_at
            FROM user_tasks
            WHERE username = %s AND task_name = %s
            """,
            (username, task_name),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return _row_to_task(row)


def sql_add_user_task(
    username: str,
    task_name: str,
    *,
    status: str = "",
    provider: str = "",
    remote_job_id: str = "",
    checkpoint_path: str = "",
    dataset_path: str = "",
    last_error: str = "",
) -> bool:
    """Add a task for one user. Return False when the task already exists."""
    pymysql, _ = _load_pymysql()
    try:
        with _connect() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO user_tasks (
                    username,
                    task_name,
                    status,
                    provider,
                    remote_job_id,
                    checkpoint_path,
                    dataset_path,
                    last_error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    username,
                    task_name,
                    status,
                    provider,
                    remote_job_id,
                    checkpoint_path,
                    dataset_path,
                    last_error,
                ),
            )
        return True
    except pymysql.err.IntegrityError:
        return False


def sql_update_user_task(username: str, task_name: str, **fields: Any) -> bool:
    if not fields:
        return False

    allowed_fields = {
        "status",
        "provider",
        "remote_job_id",
        "checkpoint_path",
        "dataset_path",
        "last_error",
    }
    assignments: list[str] = []
    values: list[Any] = []
    for field_name, field_value in fields.items():
        if field_name not in allowed_fields:
            raise ValueError(f"invalid user_tasks field: {field_name}")
        assignments.append(f"{field_name} = %s")
        values.append(field_value)
    assignments.append("updated_at = CURRENT_TIMESTAMP")
    values.extend([username, task_name])

    with _connect() as conn, conn.cursor() as cursor:
        cursor.execute(
            f"""
            UPDATE user_tasks
            SET {", ".join(assignments)}
            WHERE username = %s AND task_name = %s
            """,
            values,
        )
        return cursor.rowcount > 0


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
