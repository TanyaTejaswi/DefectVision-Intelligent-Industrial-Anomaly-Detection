"""
DefectVision – Authentication Database Module
Standalone SQLite abstraction. No imports from src/ or app/.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Ensure authentication/ is on sys.path
_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

from config.settings import (
    DATABASE_PATH,
    BOOTSTRAP_ADMIN_EMAIL,
    BOOTSTRAP_ADMIN_USERNAME,
    BOOTSTRAP_ADMIN_PASSWORD,
)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT    NOT NULL,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'inspector',
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL,
    last_login    TEXT,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until  TEXT
);
"""

_DDL_PASSWORD_RESETS = """
CREATE TABLE IF NOT EXISTS password_resets (
    reset_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    token_hash TEXT    NOT NULL UNIQUE,
    expires_at TEXT    NOT NULL,
    used       INTEGER NOT NULL DEFAULT 0
);
"""

_DDL_AUDIT_LOGS = """
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    username   TEXT,
    action     TEXT    NOT NULL,
    ip_address TEXT,
    timestamp  TEXT    NOT NULL,
    details    TEXT
);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection with row_factory."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables and bootstrap the default admin account."""
    with get_connection() as conn:
        conn.execute(_DDL_USERS)
        conn.execute(_DDL_PASSWORD_RESETS)
        conn.execute(_DDL_AUDIT_LOGS)
        conn.commit()
    bootstrap_admin()


def bootstrap_admin() -> None:
    """Create the first admin user from env-var credentials if no admin exists."""
    # Import here to avoid circular import at module level
    import bcrypt as _bcrypt  # noqa: PLC0415

    with get_connection() as conn:
        row = conn.execute("SELECT user_id FROM users WHERE role='admin' LIMIT 1").fetchone()
        if row:
            return  # Admin already exists

        pw_hash = _bcrypt.hashpw(
            BOOTSTRAP_ADMIN_PASSWORD.encode(), _bcrypt.gensalt(rounds=12)
        ).decode()

        conn.execute(
            """
            INSERT INTO users
                (full_name, username, email, password_hash, role, is_active, created_at)
            VALUES (?, ?, ?, ?, 'admin', 1, ?)
            """,
            (
                "System Administrator",
                BOOTSTRAP_ADMIN_USERNAME,
                BOOTSTRAP_ADMIN_EMAIL,
                pw_hash,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
