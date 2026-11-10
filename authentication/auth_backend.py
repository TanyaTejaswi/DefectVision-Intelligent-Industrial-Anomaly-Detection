"""
DefectVision – Core Authentication Backend
All logic is self-contained. Zero imports from src/ or app/.
"""

import hashlib
import re
import secrets
import sys
from datetime import datetime, timedelta
from pathlib import Path

_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

import bcrypt

from config.settings import (
    MAX_FAILED_ATTEMPTS,
    LOCKOUT_DURATION_MINUTES,
    RESET_TOKEN_EXPIRATION_HOURS,
    PERMITTED_REGISTRATION_ROLES,
    ALL_ROLES,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_FROM,
)
from database import get_connection, init_db

# ---------------------------------------------------------------------------
# Password utilities
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def validate_password_strength(pw: str) -> tuple[bool, str]:
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", pw):
        return False, "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", pw):
        return False, "Password must contain at least one lowercase letter."
    if not re.search(r"\d", pw):
        return False, "Password must contain at least one digit."
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", pw):
        return False, "Password must contain at least one special character."
    return True, "Strong password."


def validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_user(
    full_name: str,
    username: str,
    email: str,
    password: str,
    confirm_password: str,
    role: str,
) -> tuple[bool, str, dict | None]:
    init_db()

    # Basic field validation
    full_name = full_name.strip()
    username = username.strip().lower()
    email = email.strip().lower()
    role = role.strip().lower()

    if not full_name:
        return False, "Full name is required.", None
    if len(username) < 3:
        return False, "Username must be at least 3 characters.", None
    if not re.match(r"^[a-z0-9_]+$", username):
        return False, "Username may only contain lowercase letters, digits, and underscores.", None
    if not validate_email(email):
        return False, "Invalid email address.", None
    if role not in PERMITTED_REGISTRATION_ROLES:
        return False, "Invalid role selected.", None

    ok, msg = validate_password_strength(password)
    if not ok:
        return False, msg, None
    if password != confirm_password:
        return False, "Passwords do not match.", None

    pw_hash = hash_password(password)
    now = datetime.utcnow().isoformat()

    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users
                    (full_name, username, email, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (full_name, username, email, pw_hash, role, now),
            )
            conn.commit()
            user = dict(
                conn.execute(
                    "SELECT * FROM users WHERE username=?", (username,)
                ).fetchone()
            )
        record_audit_log("REGISTER", user["user_id"], username, None, f"New {role} account created")
        return True, "Account created successfully! You can now log in.", user
    except Exception as exc:
        msg_low = str(exc).lower()
        if "username" in msg_low:
            return False, "Username already exists.", None
        if "email" in msg_low:
            return False, "Email address already registered.", None
        return False, f"Registration failed: {exc}", None


# ---------------------------------------------------------------------------
# Authentication / Login
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def authenticate_user(
    identifier: str, password: str
) -> tuple[bool, str, dict | None]:
    """Authenticate by username OR email."""
    init_db()
    identifier = identifier.strip().lower()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE LOWER(username)=? OR LOWER(email)=?",
            (identifier, identifier),
        ).fetchone()

        if not row:
            return False, "Invalid credentials.", None

        user = dict(row)

        # Check active
        if not user["is_active"]:
            return False, "Your account has been deactivated. Contact an administrator.", None

        # Check lockout
        if user["locked_until"]:
            lock_time = datetime.fromisoformat(user["locked_until"])
            if datetime.utcnow() < lock_time:
                remaining = int((lock_time - datetime.utcnow()).total_seconds() / 60) + 1
                return False, f"Account locked. Try again in {remaining} minute(s).", None
            else:
                # Lockout expired — reset
                conn.execute(
                    "UPDATE users SET failed_attempts=0, locked_until=NULL WHERE user_id=?",
                    (user["user_id"],),
                )
                conn.commit()
                user["failed_attempts"] = 0
                user["locked_until"] = None

        # Verify password
        if not verify_password(password, user["password_hash"]):
            new_fails = user["failed_attempts"] + 1
            locked_until = None
            if new_fails >= MAX_FAILED_ATTEMPTS:
                locked_until = (
                    datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
                ).isoformat()
                msg = f"Too many failed attempts. Account locked for {LOCKOUT_DURATION_MINUTES} minutes."
            else:
                remaining = MAX_FAILED_ATTEMPTS - new_fails
                msg = f"Invalid credentials. {remaining} attempt(s) remaining before lockout."
            conn.execute(
                "UPDATE users SET failed_attempts=?, locked_until=? WHERE user_id=?",
                (new_fails, locked_until, user["user_id"]),
            )
            conn.commit()
            record_audit_log("LOGIN_FAIL", user["user_id"], user["username"], None, msg)
            return False, msg, None

        # Success — reset counters, update last_login
        conn.execute(
            "UPDATE users SET failed_attempts=0, locked_until=NULL, last_login=? WHERE user_id=?",
            (_now_iso(), user["user_id"]),
        )
        conn.commit()
        user = dict(
            conn.execute("SELECT * FROM users WHERE user_id=?", (user["user_id"],)).fetchone()
        )

    record_audit_log("LOGIN_SUCCESS", user["user_id"], user["username"], None, "Successful login")
    return True, "Login successful.", user


# ---------------------------------------------------------------------------
# Password Reset
# ---------------------------------------------------------------------------

def _sha256(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_reset_token(email: str) -> tuple[bool, str, str | None]:
    """Generate a password-reset token. Returns (ok, message, raw_token_or_None)."""
    init_db()
    email = email.strip().lower()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE LOWER(email)=? AND is_active=1", (email,)
        ).fetchone()
        if not row:
            # Security: don't reveal whether email exists
            return True, "If that email is registered, a reset link has been sent.", None

        user = dict(row)

        # Invalidate old tokens
        conn.execute(
            "UPDATE password_resets SET used=1 WHERE user_id=? AND used=0",
            (user["user_id"],),
        )

        raw_token = secrets.token_urlsafe(48)
        token_hash = _sha256(raw_token)
        expires_at = (
            datetime.utcnow() + timedelta(hours=RESET_TOKEN_EXPIRATION_HOURS)
        ).isoformat()

        conn.execute(
            "INSERT INTO password_resets (user_id, token_hash, expires_at) VALUES (?,?,?)",
            (user["user_id"], token_hash, expires_at),
        )
        conn.commit()

    record_audit_log("RESET_REQUEST", user["user_id"], user["username"], None, "Password reset requested")

    # Attempt SMTP dispatch if configured
    if SMTP_HOST and SMTP_USER:
        _send_reset_email(email, user["username"], raw_token)
        return True, "A password-reset link has been sent to your email.", None

    # Dev mode — return raw token to show on-screen
    return True, "__DEV_MODE__", raw_token


def _send_reset_email(to_email: str, username: str, raw_token: str) -> None:
    import smtplib
    from email.mime.text import MIMEText

    body = f"""Hello {username},

A password reset was requested for your DefectVision account.

Your reset token: {raw_token}

This token expires in {RESET_TOKEN_EXPIRATION_HOURS} hour(s).
If you did not request this, ignore this email.

— DefectVision Security System
"""
    msg = MIMEText(body)
    msg["Subject"] = "DefectVision — Password Reset"
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, [to_email], msg.as_string())
    except Exception:
        pass  # Silently fail — UI shows dev-mode fallback


def verify_reset_token(raw_token: str) -> tuple[bool, str, dict | None]:
    init_db()
    token_hash = _sha256(raw_token.strip())
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT pr.*, u.username, u.email
            FROM password_resets pr
            JOIN users u ON pr.user_id = u.user_id
            WHERE pr.token_hash=? AND pr.used=0
            """,
            (token_hash,),
        ).fetchone()
        if not row:
            return False, "Invalid or already-used reset token.", None
        reset = dict(row)
        if datetime.utcnow() > datetime.fromisoformat(reset["expires_at"]):
            return False, "Reset token has expired. Please request a new one.", None
    return True, "Token valid.", reset


def reset_password_with_token(
    raw_token: str, new_password: str, confirm_password: str
) -> tuple[bool, str]:
    ok, msg, reset = verify_reset_token(raw_token)
    if not ok:
        return False, msg
    if new_password != confirm_password:
        return False, "Passwords do not match."
    strength_ok, strength_msg = validate_password_strength(new_password)
    if not strength_ok:
        return False, strength_msg

    pw_hash = hash_password(new_password)
    token_hash = _sha256(raw_token.strip())
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash=?, failed_attempts=0, locked_until=NULL WHERE user_id=?",
            (pw_hash, reset["user_id"]),
        )
        conn.execute(
            "UPDATE password_resets SET used=1 WHERE token_hash=?", (token_hash,)
        )
        conn.commit()
    record_audit_log("PASSWORD_RESET", reset["user_id"], reset["username"], None, "Password reset completed")
    return True, "Password reset successfully. You can now log in."


# ---------------------------------------------------------------------------
# Admin User Management
# ---------------------------------------------------------------------------

def get_all_users() -> list[dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT user_id, full_name, username, email, role, is_active, created_at, last_login FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_user_status(admin_id: int, target_id: int, is_active: bool) -> tuple[bool, str]:
    if admin_id == target_id:
        return False, "You cannot change your own account status."
    with get_connection() as conn:
        row = conn.execute("SELECT role FROM users WHERE user_id=?", (target_id,)).fetchone()
        if row and row["role"] == "admin":
            return False, "Cannot deactivate another admin account."
        conn.execute(
            "UPDATE users SET is_active=? WHERE user_id=?",
            (1 if is_active else 0, target_id),
        )
        conn.commit()
    action = "ACTIVATE_USER" if is_active else "DEACTIVATE_USER"
    record_audit_log(action, admin_id, "admin", None, f"Target user_id={target_id}")
    return True, "User status updated."


def update_user_role(admin_id: int, target_id: int, new_role: str) -> tuple[bool, str]:
    if admin_id == target_id:
        return False, "You cannot change your own role."
    if new_role not in ALL_ROLES:
        return False, "Invalid role."
    with get_connection() as conn:
        conn.execute("UPDATE users SET role=? WHERE user_id=?", (new_role, target_id))
        conn.commit()
    record_audit_log("CHANGE_ROLE", admin_id, "admin", None, f"Target user_id={target_id} → role={new_role}")
    return True, "Role updated."


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

def record_audit_log(
    action: str,
    user_id: int | None = None,
    username: str | None = None,
    ip_address: str | None = None,
    details: str | None = None,
) -> None:
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO audit_logs (user_id, username, action, ip_address, timestamp, details) VALUES (?,?,?,?,?,?)",
                (user_id, username, action, ip_address, _now_iso(), details),
            )
            conn.commit()
    except Exception:
        pass  # Audit failures must never crash auth flow


def get_audit_logs(limit: int = 100) -> list[dict]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
