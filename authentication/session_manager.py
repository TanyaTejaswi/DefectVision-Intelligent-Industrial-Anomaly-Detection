"""
DefectVision – Session Manager
Manages Streamlit session state for the auth system.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

import streamlit as st
from config.settings import SESSION_EXPIRATION_HOURS, REMEMBER_ME_DAYS

_SESSION_KEYS = [
    "auth_logged_in",
    "auth_user",
    "auth_role",
    "auth_session_started",
    "auth_remember_me",
    "auth_page",
    "auth_message",
    "auth_message_type",
]


def init_session() -> None:
    """Initialise all session state keys if not already set."""
    defaults = {
        "auth_logged_in": False,
        "auth_user": None,
        "auth_role": None,
        "auth_session_started": None,
        "auth_remember_me": False,
        "auth_page": "login",
        "auth_message": None,
        "auth_message_type": "info",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def login_user(user_dict: dict, remember_me: bool = False) -> None:
    st.session_state.auth_logged_in = True
    st.session_state.auth_user = user_dict
    st.session_state.auth_role = user_dict.get("role", "inspector")
    st.session_state.auth_session_started = datetime.utcnow().isoformat()
    st.session_state.auth_remember_me = remember_me


def logout_user() -> None:
    for key in _SESSION_KEYS:
        if key in st.session_state:
            del st.session_state[key]
    init_session()


def is_logged_in() -> bool:
    init_session()
    if not st.session_state.auth_logged_in:
        return False
    if not st.session_state.auth_session_started:
        return False

    started = datetime.fromisoformat(st.session_state.auth_session_started)
    remember = st.session_state.auth_remember_me
    expiry_hours = REMEMBER_ME_DAYS * 24 if remember else SESSION_EXPIRATION_HOURS

    if datetime.utcnow() > started + timedelta(hours=expiry_hours):
        logout_user()
        set_message("Your session has expired. Please log in again.", "warning")
        return False
    return True


def require_auth() -> None:
    """Redirect to login if not authenticated. Call at top of protected pages."""
    if not is_logged_in():
        st.session_state.auth_page = "login"
        st.rerun()


def require_admin() -> None:
    """Redirect if not admin."""
    require_auth()
    if st.session_state.get("auth_role") != "admin":
        st.error("⛔ Access denied. Administrator privileges required.")
        st.stop()


def get_current_user() -> dict | None:
    if is_logged_in():
        return st.session_state.auth_user
    return None


def set_message(msg: str, msg_type: str = "info") -> None:
    st.session_state.auth_message = msg
    st.session_state.auth_message_type = msg_type


def show_message() -> None:
    """Display and clear any queued message."""
    msg = st.session_state.get("auth_message")
    if msg:
        mtype = st.session_state.get("auth_message_type", "info")
        getattr(st, mtype)(msg)
        st.session_state.auth_message = None
        st.session_state.auth_message_type = "info"
