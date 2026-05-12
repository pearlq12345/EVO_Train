#!/usr/bin/env python3
"""SQLite helpers for tcp_read_server task storage."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
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
            hourly_price_cents INTEGER NOT NULL DEFAULT 0,
            frozen_until TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL DEFAULT '',
            stopped_at TEXT NOT NULL DEFAULT '',
            actual_cost_cents INTEGER NOT NULL DEFAULT 0,
            billing_status TEXT NOT NULL DEFAULT '',
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
            "hourly_price_cents": "INTEGER NOT NULL DEFAULT 0",
            "frozen_until": "TEXT NOT NULL DEFAULT ''",
            "started_at": "TEXT NOT NULL DEFAULT ''",
            "stopped_at": "TEXT NOT NULL DEFAULT ''",
            "actual_cost_cents": "INTEGER NOT NULL DEFAULT 0",
            "billing_status": "TEXT NOT NULL DEFAULT ''",
            "last_error": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
        },
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_wallets (
            username TEXT PRIMARY KEY,
            balance_cents INTEGER NOT NULL DEFAULT 0,
            frozen_cents INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gpu_prices (
            provider TEXT NOT NULL,
            gpu_spec TEXT NOT NULL,
            hourly_price_cents INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (provider, gpu_spec)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS billing_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            task_name TEXT NOT NULL DEFAULT '',
            kind TEXT NOT NULL,
            amount_cents INTEGER NOT NULL,
            balance_after_cents INTEGER NOT NULL,
            frozen_after_cents INTEGER NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
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
    task = {
        "taskName": row["task_name"],
        "status": row["status"],
        "provider": row["provider"],
        "jobId": row["remote_job_id"],
        "checkpointPath": row["checkpoint_path"],
        "datasetPath": row["dataset_path"],
        "hourlyPriceCents": str(row["hourly_price_cents"]),
        "frozenUntil": row["frozen_until"],
        "startedAt": row["started_at"],
        "stoppedAt": row["stopped_at"],
        "actualCostCents": str(row["actual_cost_cents"]),
        "billingStatus": row["billing_status"],
        "error": row["last_error"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }
    if "username" in row.keys():
        task["username"] = row["username"]
    return task


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
                hourly_price_cents,
                frozen_until,
                started_at,
                stopped_at,
                actual_cost_cents,
                billing_status,
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
                hourly_price_cents,
                frozen_until,
                started_at,
                stopped_at,
                actual_cost_cents,
                billing_status,
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


def sql_get_billing_watch_tasks() -> list[dict[str, str]]:
    """Return tasks that still need billing reconciliation or runtime enforcement."""
    with _managed_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                username,
                task_name,
                status,
                provider,
                remote_job_id,
                checkpoint_path,
                dataset_path,
                hourly_price_cents,
                frozen_until,
                started_at,
                stopped_at,
                actual_cost_cents,
                billing_status,
                last_error,
                created_at,
                updated_at
            FROM user_tasks
            WHERE hourly_price_cents > 0
              AND billing_status != 'settled'
            ORDER BY updated_at ASC, created_at ASC
            """
        ).fetchall()
    return [_row_to_task(row) for row in rows]


def sql_add_user_task(
    username: str,
    task_name: str,
    *,
    status: str = "",
    provider: str = "",
    remote_job_id: str = "",
    checkpoint_path: str = "",
    dataset_path: str = "",
    hourly_price_cents: int = 0,
    frozen_until: str = "",
    started_at: str = "",
    stopped_at: str = "",
    actual_cost_cents: int = 0,
    billing_status: str = "",
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
                    hourly_price_cents,
                    frozen_until,
                    started_at,
                    stopped_at,
                    actual_cost_cents,
                    billing_status,
                    last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    task_name,
                    status,
                    provider,
                    remote_job_id,
                    checkpoint_path,
                    dataset_path,
                    hourly_price_cents,
                    frozen_until,
                    started_at,
                    stopped_at,
                    actual_cost_cents,
                    billing_status,
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


def utc_now_text() -> str:
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat(sep=" ")


def add_hours_text(timestamp: str, hours: int) -> str:
    base = parse_timestamp(timestamp) or datetime.now(UTC).replace(tzinfo=None, microsecond=0)
    return (base + timedelta(hours=hours)).isoformat(sep=" ")


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def sql_get_wallet(username: str) -> dict[str, str]:
    with _managed_connection() as conn:
        row = conn.execute(
            """
            SELECT username, balance_cents, frozen_cents, updated_at
            FROM user_wallets
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    if row is None:
        return {"username": username, "balanceCents": "0", "frozenCents": "0", "availableCents": "0", "updatedAt": ""}
    balance = int(row["balance_cents"])
    frozen = int(row["frozen_cents"])
    return {
        "username": row["username"],
        "balanceCents": str(balance),
        "frozenCents": str(frozen),
        "availableCents": str(balance - frozen),
        "updatedAt": row["updated_at"],
    }


def sql_set_user_balance(username: str, balance_cents: int) -> None:
    with _managed_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_wallets (username, balance_cents, frozen_cents, updated_at)
            VALUES (?, ?, 0, CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET
                balance_cents = excluded.balance_cents,
                updated_at = CURRENT_TIMESTAMP
            """,
            (username, balance_cents),
        )
        _insert_billing_record(conn, username, "", "admin_set_balance", balance_cents, "manual balance update")


def sql_set_gpu_price(provider: str, gpu_spec: str, hourly_price_cents: int) -> None:
    with _managed_connection() as conn:
        conn.execute(
            """
            INSERT INTO gpu_prices (provider, gpu_spec, hourly_price_cents, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(provider, gpu_spec) DO UPDATE SET
                hourly_price_cents = excluded.hourly_price_cents,
                updated_at = CURRENT_TIMESTAMP
            """,
            (provider, gpu_spec, hourly_price_cents),
        )


def sql_get_hourly_price(provider: str, gpu_spec: str = "default") -> int:
    with _managed_connection() as conn:
        row = conn.execute(
            """
            SELECT hourly_price_cents
            FROM gpu_prices
            WHERE provider = ? AND gpu_spec = ?
            """,
            (provider, gpu_spec or "default"),
        ).fetchone()
    if row is not None:
        return int(row["hourly_price_cents"])
    return int(os.environ.get("EVO_TRAIN_DEFAULT_HOURLY_PRICE_CENTS", "1000"))


def sql_get_gpu_prices(provider: str | None = None) -> list[dict[str, str]]:
    query = """
        SELECT provider, gpu_spec, hourly_price_cents, updated_at
        FROM gpu_prices
    """
    params: tuple[Any, ...] = ()
    if provider:
        query += " WHERE provider = ?"
        params = (provider,)
    query += " ORDER BY provider ASC, gpu_spec ASC"
    with _managed_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [
        {
            "provider": row["provider"],
            "gpuSpec": row["gpu_spec"],
            "hourlyPriceCents": str(row["hourly_price_cents"]),
            "updatedAt": row["updated_at"],
        }
        for row in rows
    ]


def sql_freeze_user_balance(username: str, task_name: str, amount_cents: int, reason: str) -> bool:
    with _managed_connection() as conn:
        wallet = _wallet_for_update(conn, username)
        available = int(wallet["balance_cents"]) - int(wallet["frozen_cents"])
        if available < amount_cents:
            return False
        balance_after = int(wallet["balance_cents"])
        frozen_after = int(wallet["frozen_cents"]) + amount_cents
        conn.execute(
            """
            UPDATE user_wallets
            SET frozen_cents = ?, updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
            """,
            (frozen_after, username),
        )
        _insert_billing_record(conn, username, task_name, "freeze", amount_cents, reason, balance_after, frozen_after)
        return True


def sql_refund_frozen_balance(username: str, task_name: str, amount_cents: int, reason: str) -> None:
    with _managed_connection() as conn:
        wallet = _wallet_for_update(conn, username)
        frozen_after = max(0, int(wallet["frozen_cents"]) - amount_cents)
        balance_after = int(wallet["balance_cents"])
        conn.execute(
            """
            UPDATE user_wallets
            SET frozen_cents = ?, updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
            """,
            (frozen_after, username),
        )
        _insert_billing_record(conn, username, task_name, "refund_frozen", amount_cents, reason, balance_after, frozen_after)


def sql_charge_frozen_balance(username: str, task_name: str, amount_cents: int, reason: str) -> None:
    with _managed_connection() as conn:
        wallet = _wallet_for_update(conn, username)
        balance_after = max(0, int(wallet["balance_cents"]) - amount_cents)
        frozen_after = max(0, int(wallet["frozen_cents"]) - amount_cents)
        conn.execute(
            """
            UPDATE user_wallets
            SET balance_cents = ?, frozen_cents = ?, updated_at = CURRENT_TIMESTAMP
            WHERE username = ?
            """,
            (balance_after, frozen_after, username),
        )
        _insert_billing_record(conn, username, task_name, "charge", amount_cents, reason, balance_after, frozen_after)


def sql_get_billing_records(username: str) -> list[dict[str, str]]:
    with _managed_connection() as conn:
        rows = conn.execute(
            """
            SELECT task_name, kind, amount_cents, balance_after_cents, frozen_after_cents, reason, created_at
            FROM billing_records
            WHERE username = ?
            ORDER BY id ASC
            """,
            (username,),
        ).fetchall()
    return [
        {
            "taskName": row["task_name"],
            "kind": row["kind"],
            "amountCents": str(row["amount_cents"]),
            "balanceAfterCents": str(row["balance_after_cents"]),
            "frozenAfterCents": str(row["frozen_after_cents"]),
            "reason": row["reason"],
            "createdAt": row["created_at"],
        }
        for row in rows
    ]


def sql_get_task_frozen_cents(username: str, task_name: str) -> int:
    with _managed_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN kind = 'freeze' THEN amount_cents ELSE 0 END), 0) -
                COALESCE(SUM(CASE WHEN kind IN ('charge', 'refund_frozen') THEN amount_cents ELSE 0 END), 0)
                AS frozen_for_task
            FROM billing_records
            WHERE username = ? AND task_name = ?
            """,
            (username, task_name),
        ).fetchone()
    return max(0, int(row["frozen_for_task"] or 0))


def _wallet_for_update(conn: sqlite3.Connection, username: str) -> sqlite3.Row:
    conn.execute(
        """
        INSERT OR IGNORE INTO user_wallets (username, balance_cents, frozen_cents, updated_at)
        VALUES (?, 0, 0, CURRENT_TIMESTAMP)
        """,
        (username,),
    )
    return conn.execute(
        """
        SELECT balance_cents, frozen_cents
        FROM user_wallets
        WHERE username = ?
        """,
        (username,),
    ).fetchone()


def _insert_billing_record(
    conn: sqlite3.Connection,
    username: str,
    task_name: str,
    kind: str,
    amount_cents: int,
    reason: str,
    balance_after: int | None = None,
    frozen_after: int | None = None,
) -> None:
    if balance_after is None or frozen_after is None:
        wallet = _wallet_for_update(conn, username)
        balance_after = int(wallet["balance_cents"])
        frozen_after = int(wallet["frozen_cents"])
    conn.execute(
        """
        INSERT INTO billing_records (
            username,
            task_name,
            kind,
            amount_cents,
            balance_after_cents,
            frozen_after_cents,
            reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (username, task_name, kind, amount_cents, balance_after, frozen_after, reason),
    )
