# DefectVision Authentication System

> Standalone authentication layer for the DefectVision Industrial AI Platform.
> Runs independently on port **8502** without touching any existing project files.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r authentication/requirements-auth.txt
```

> **Note:** `bcrypt` is already installed in the project's `.venv`. Run `pip install -r authentication/requirements-auth.txt` only if setting up a new environment.

### 2. Configure environment (optional)

```bash
cp authentication/.env.example authentication/.env
# Edit .env with your real admin credentials and SMTP settings
```

If you skip this step, the default bootstrap admin is:
- **Email:** `admin@defectvision.industrial`
- **Username:** `admin`
- **Password:** `DefectVision@Admin2026!`

> ⚠️ Change the default password immediately after first login.

### 3. Run the authentication app

```bash
streamlit run authentication/login.py --server.port 8502
```

Open **http://localhost:8502** in your browser.

The existing DefectVision app continues running on **http://localhost:8501**.

---

## Features

| Feature | Details |
|---------|---------|
| **Login** | Email or username + password, rate-limited |
| **Registration** | Full name, username, email, role (inspector / engineer / operator) |
| **Forgot Password** | Token-based reset; dev-mode shows token on-screen if SMTP not configured |
| **Session Management** | Configurable expiry, Remember Me (30 days), automatic logout |
| **Account Lockout** | 5 failed attempts → 15-minute lockout (configurable) |
| **Admin Panel** | User table, activate/deactivate, role changes, audit log |
| **Audit Logging** | Every login, registration, and admin action is logged |
| **Password Hashing** | bcrypt, 12 rounds |
| **Database** | SQLite WAL-mode at `authentication/database/auth_users.db` |

---

## Folder Structure

```
authentication/
├── login.py              ← Entry point (streamlit run this)
├── register.py           ← Registration page
├── forgot_password.py    ← Password recovery
├── auth_backend.py       ← Core auth logic
├── database.py           ← SQLite abstraction
├── session_manager.py    ← Streamlit session helpers
├── user_dashboard.py     ← Authenticated user home
├── admin_dashboard.py    ← Admin-only control panel
├── styles.css            ← Premium dark industrial theme
├── requirements-auth.txt ← Standalone pip requirements
├── .env.example          ← Environment variable template
├── config/
│   └── settings.py       ← All configuration constants
└── database/
    └── auth_users.db     ← Created at runtime
```

---

## Roles

| Role | Self-Registration | Description |
|------|:-----------------:|-------------|
| `inspector` | ✅ | Quality Inspector |
| `engineer`  | ✅ | MLOps Engineer |
| `operator`  | ✅ | Plant Operator |
| `admin`     | ❌ Bootstrap only | System Administrator |

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEFECTVISION_ADMIN_EMAIL` | `admin@defectvision.industrial` | Bootstrap admin email |
| `DEFECTVISION_ADMIN_USERNAME` | `admin` | Bootstrap admin username |
| `DEFECTVISION_ADMIN_PASSWORD` | `DefectVision@Admin2026!` | Bootstrap admin password |
| `DEFECTVISION_SESSION_HOURS` | `12` | Session lifetime in hours |
| `DEFECTVISION_REMEMBER_DAYS` | `30` | Remember-me lifetime in days |
| `DEFECTVISION_MAX_FAILED_ATTEMPTS` | `5` | Attempts before lockout |
| `DEFECTVISION_LOCKOUT_MINUTES` | `15` | Lockout duration |
| `SMTP_HOST` | _(blank)_ | SMTP server (blank = dev mode) |
| `SMTP_USER` | _(blank)_ | SMTP username |
| `SMTP_PASSWORD` | _(blank)_ | SMTP password |

---

## Security Notes

- Passwords are hashed with **bcrypt (12 rounds)** — never stored in plaintext.
- The SQLite database is local; for production, consider PostgreSQL.
- All admin actions are written to `audit_logs` table.
- The auth system is fully isolated from the main DefectVision codebase.
