"""
DefectVision – Login Entry Point
Run: streamlit run authentication/login.py --server.port 8502
"""

import sys
from pathlib import Path

_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

import streamlit as st

st.set_page_config(
    page_title="DefectVision — Secure Login",
    page_icon="🔐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load CSS ───────────────────────────────────────────────
_CSS_PATH = _AUTH_DIR / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

from session_manager import init_session, login_user, is_logged_in, show_message, set_message
from auth_backend import authenticate_user
from database import init_db

init_db()
init_session()


# ── Routing ────────────────────────────────────────────────
def _goto(page: str, msg: str = None, msg_type: str = "info"):
    if msg:
        set_message(msg, msg_type)
    st.session_state.auth_page = page
    st.rerun()


# Already logged in → go to dashboard
if is_logged_in():
    role = st.session_state.auth_role
    if role == "admin":
        import admin_dashboard as _dash
        _dash.show()
    else:
        import user_dashboard as _dash
        _dash.show()
    st.stop()

page = st.session_state.get("auth_page", "login")

if page == "register":
    import register as _reg
    _reg.show()
elif page == "forgot_password":
    import forgot_password as _fp
    _fp.show()
else:
    # ── LOGIN PAGE ─────────────────────────────────────────
    # Brand Hero
    st.markdown("""
    <div class="dv-brand">
        <div class="dv-logo-ring">🔬</div>
        <div class="dv-title">DEFECTVISION</div>
        <div class="dv-subtitle">Intelligent Industrial Anomaly Detection</div>
        <div class="dv-status">
            <div class="dv-status-dot"></div>
            SECURE PORTAL · AUTHENTICATED ACCESS ONLY
        </div>
    </div>
    """, unsafe_allow_html=True)

    show_message()

    st.markdown('<div class="dv-card">', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-title">🔐 Sign In to Your Account</div>', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-desc">Enter your credentials to access the DefectVision platform.</div>', unsafe_allow_html=True)

    identifier = st.text_input("📧 Email or Username", placeholder="you@company.com or username", key="login_id")
    password   = st.text_input("🔑 Password", type="password", placeholder="••••••••••", key="login_pw")

    col_rem, col_forgot = st.columns([1, 1])
    with col_rem:
        remember_me = st.checkbox("Remember me", key="login_rem")
    with col_forgot:
        st.markdown('<div style="text-align:right; margin-top:4px;">', unsafe_allow_html=True)
        if st.button("Forgot password?", key="go_forgot"):
            _goto("forgot_password")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Sign In", key="login_btn"):
        if not identifier or not password:
            st.error("⚠️ Please enter both email/username and password.")
        else:
            with st.spinner("Authenticating…"):
                ok, msg, user = authenticate_user(identifier, password)
            if ok:
                login_user(user, remember_me=remember_me)
                st.success("✅ Login successful! Redirecting…")
                st.rerun()
            else:
                st.error(f"❌ {msg}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Register link
    st.markdown('<div style="text-align:center; margin-top:1rem;">', unsafe_allow_html=True)
    st.markdown('<span style="color:#64748B; font-size:0.85rem;">Don\'t have an account?&nbsp;</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="dv-link-btn">', unsafe_allow_html=True)
    if st.button("✨ Create Account", key="go_register"):
        _goto("register")
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center; margin-top:2rem; color:#334155; font-size:0.72rem; letter-spacing:0.08em;">
        DEFECTVISION · INDUSTRIAL AI PLATFORM · SECURED BY BCRYPT
    </div>
    """, unsafe_allow_html=True)
