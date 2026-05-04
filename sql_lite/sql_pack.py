#!/usr/bin/env python3
"""SQLite helpers for tcp_read_server task storage."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "tasks.sqlite3"
DB_PATH = Path(os.environ.get("EVO_TRAIN_TASK_DB", DEFAULT_DB_PATH)).expanduser()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


@contextmanager
def _managed_connection() -> sqlite3.Connection:
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_tasks (
            username TEXT NOT NULL,
            task_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT '',
            remote_job_id TEXT NOT NULL DEFAULT '',
            checkpoint_path TEXT NOT NULL DEFAULT '',
            dataset_path TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (username, task_name)
        )
        """
    )
    _ensure_columns(
        conn,
        "user_tasks",
        {
            "provider": "TEXT NOT NULL DEFAULT ''",
            "remote_job_id": "TEXT NOT NULL DEFAULT ''",
            "checkpoint_path": "TEXT NOT NULL DEFAULT ''",
            "dataset_path": "TEXT NOT NULL DEFAULT ''",
            "last_error": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
        },
    )
    conn.commit()


def _ensure_columns(conn: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    existing_columns = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, column_sql in columns.items():
        if column_name in existing_columns:
            continue
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")


def _row_to_task(row: sqlite3.Row) -> dict[str, str]:
    return {
        "taskName": row["task_name"],
        "status": row["status"],
        "provider": row["provider"],
        "jobId": row["remote_job_id"],
        "checkpointPath": row["checkpoint_path"],
        "datasetPath": row["dataset_path"],
        "error": row["last_error"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def sql_get_user_all_task(username: str) -> list[dict[str, str]]:
    """Return all tasks for one user in the response format expected by the TCP server."""
    with _managed_connection() as conn:
        rows = conn.execute(
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
            WHERE username = ?
            ORDER BY created_at ASC, task_name ASC
            """,
            (username,),
        ).fetchall()

    return [_row_to_task(row) for row in rows]


def sql_get_user_task(username: str, task_name: str) -> dict[str, str] | None:
    with _managed_connection() as conn:
        row = conn.execute(
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
            WHERE username = ? AND task_name = ?
            """,
            (username, task_name),
        ).fetchone()
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
    try:
        with _managed_connection() as conn:
            conn.execute(
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
    except sqlite3.IntegrityError:
        return False


def sql_update_user_task(username: str, task_name: str, **fields: Any) -> bool:
    if not fields:
        return False

    assignments: list[str] = []
    values: list[Any] = []
    for field_name, field_value in fields.items():
        assignments.append(f"{field_name} = ?")
        values.append(field_value)
    assignments.append("updated_at = CURRENT_TIMESTAMP")
    values.extend([username, task_name])

    with _managed_connection() as conn:
        cursor = conn.execute(
            f"""
            UPDATE user_tasks
            SET {", ".join(assignments)}
            WHERE username = ? AND task_name = ?
            """,
            values,
        )
        return cursor.rowcount > 0


def sql_delete_user_task(username: str, task_name: str) -> bool:
    """Delete a task for one user. Return True only when a row was deleted."""
    with _managed_connection() as conn:
        cursor = conn.execute(
            """
            DELETE FROM user_tasks
            WHERE username = ? AND task_name = ?
            """,
            (username, task_name),
        )
        return cursor.rowcount > 0
