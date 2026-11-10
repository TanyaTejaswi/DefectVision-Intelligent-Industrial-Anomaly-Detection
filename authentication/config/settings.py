"""
DefectVision Authentication Configuration Module
Loads configuration from environment variables with production-ready security defaults.
"""

from pathlib import Path
import os
import secrets

AUTH_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = AUTH_ROOT.parent

# Database configuration
DATABASE_PATH = AUTH_ROOT / "database" / "auth_users.db"

# Security Parameters
SECRET_KEY = os.environ.get("DEFECTVISION_SECRET_KEY", secrets.token_hex(32))
MAX_FAILED_ATTEMPTS = int(os.environ.get("DEFECTVISION_MAX_FAILED_ATTEMPTS", "5"))
LOCKOUT_DURATION_MINUTES = int(os.environ.get("DEFECTVISION_LOCKOUT_MINUTES", "15"))
SESSION_EXPIRATION_HOURS = int(os.environ.get("DEFECTVISION_SESSION_HOURS", "12"))
REMEMBER_ME_DAYS = int(os.environ.get("DEFECTVISION_REMEMBER_DAYS", "30"))
RESET_TOKEN_EXPIRATION_HOURS = int(os.environ.get("DEFECTVISION_RESET_TOKEN_HOURS", "1"))

# Initial Bootstrap Administrator Credentials (Loaded from environment)
BOOTSTRAP_ADMIN_EMAIL = os.environ.get("DEFECTVISION_ADMIN_EMAIL", "admin@defectvision.industrial").strip().lower()
BOOTSTRAP_ADMIN_USERNAME = os.environ.get("DEFECTVISION_ADMIN_USERNAME", "admin").strip().lower()
BOOTSTRAP_ADMIN_PASSWORD = os.environ.get("DEFECTVISION_ADMIN_PASSWORD", "DefectVision@Admin2026!")

# SMTP Email Configuration (Optional - for production password reset dispatch)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587")) if os.environ.get("SMTP_PORT") else 587
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "no-reply@defectvision.industrial")

# Roles definition
PERMITTED_REGISTRATION_ROLES = {
    "inspector": "Quality Inspector",
    "engineer": "MLOps Engineer",
    "operator": "Plant Operator"
}

ALL_ROLES = {
    "admin": "System Administrator",
    "inspector": "Quality Inspector",
    "engineer": "MLOps Engineer",
    "operator": "Plant Operator"
}
