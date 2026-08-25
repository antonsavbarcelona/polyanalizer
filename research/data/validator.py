"""Data-quality validation for immutable collector DBs."""
from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass


REQUIRED_TABLES = {
    "markets",
    "binance_trades",
    "binance_book",
    "polymarket_book",
    "chainlink_observations",
}


@dataclass
class ValidationReport:
    asset: str
    db_path: str
    ok: bool
    errors: list[str]
    markets: int
    row_counts: dict[str, int]
    invalid_books: int
    missing_metadata: int
    fingerprint: str


def open_readonly(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def validate_collector_db(asset: str, db_path: str) -> ValidationReport:
    errors: list[str] = []
    row_counts: dict[str, int] = {}
    invalid_books = 0
    missing_metadata = 0

    if not os.path.exists(db_path):
        return ValidationReport(asset, db_path, False, [f"DB does not exist: {db_path}"], 0, {}, 0, 0, "")

    try:
        conn = open_readonly(db_path)
    except sqlite3.Error as exc:
        return ValidationReport(asset, db_path, False, [f"cannot open read-only DB: {exc}"], 0, {}, 0, 0, "")

    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            errors.append(f"missing tables: {', '.join(missing)}")
        for table in sorted(REQUIRED_TABLES & tables):
            row_counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ["markets", "binance_trades", "binance_book", "polymarket_book"]:
            if row_counts.get(table, 0) == 0:
                errors.append(f"{table} is empty")

        if "markets" in tables:
            missing_metadata = conn.execute(
                """
                SELECT COUNT(*) FROM markets
                WHERE market_id IS NULL OR up_token_id IS NULL OR down_token_id IS NULL
                   OR start_ts IS NULL OR end_ts IS NULL OR start_ts >= end_ts
                """
            ).fetchone()[0]
            if missing_metadata:
                errors.append(f"invalid market metadata rows: {missing_metadata}")

        if "polymarket_book" in tables:
            invalid_books = conn.execute(
                """
                SELECT COUNT(*) FROM polymarket_book
                WHERE (up_best_bid IS NOT NULL AND up_best_ask IS NOT NULL AND up_best_bid > up_best_ask)
                   OR (down_best_bid IS NOT NULL AND down_best_ask IS NOT NULL AND down_best_bid > down_best_ask)
                """
            ).fetchone()[0]

        markets = row_counts.get("markets", 0)
    except sqlite3.Error as exc:
        errors.append(f"validation query failed: {exc}")
        markets = 0
    finally:
        conn.close()

    return ValidationReport(
        asset=asset,
        db_path=db_path,
        ok=not errors,
        errors=errors,
        markets=markets,
        row_counts=row_counts,
        invalid_books=invalid_books,
        missing_metadata=missing_metadata,
        fingerprint=fingerprint_db(db_path),
    )


def fingerprint_db(db_path: str) -> str:
    digest = hashlib.sha256()
    for suffix in ["", "-wal"]:
        path = db_path + suffix
        if not os.path.exists(path):
            continue
        digest.update(os.path.basename(path).encode("utf-8"))
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()
