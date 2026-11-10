"""
DefectVision – User Dashboard
Shown after successful login for non-admin users.
"""

import sys
from pathlib import Path
from datetime import datetime

_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

import streamlit as st
from session_manager import require_auth, get_current_user, logout_user
from config.settings import ALL_ROLES


_ROLE_CSS_CLASS = {
    "admin":     "dv-role-admin",
    "engineer":  "dv-role-engineer",
    "inspector": "dv-role-inspector",
    "operator":  "dv-role-operator",
}


def _fmt_dt(iso_str: str | None) -> str:
    if not iso_str:
        return "—"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return iso_str


def show():
    require_auth()
    user = get_current_user()
    if not user:
        st.rerun()
        return

    # ── Load CSS ───────────────────────────────────────────
    _CSS = _AUTH_DIR / "styles.css"
    if _CSS.exists():
        st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

    role_key   = user.get("role", "inspector")
    role_label = ALL_ROLES.get(role_key, role_key.title())
    role_cls   = _ROLE_CSS_CLASS.get(role_key, "dv-role-inspector")

    # ── Header ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem;">
        <div>
            <div style="font-size:1.6rem; font-weight:800; background:linear-gradient(90deg,#00D2FF,#FFFFFF);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                DEFECTVISION
            </div>
            <div style="color:#64748B; font-size:0.78rem; letter-spacing:0.1em;">INDUSTRIAL AI PLATFORM</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.9rem; font-weight:600; color:#E2E8F0;">
                {user.get('full_name', user.get('username', 'User'))}
            </div>
            <span class="dv-role-badge {role_cls}">{role_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Welcome Banner ─────────────────────────────────────
    st.markdown(f"""
    <div class="dv-card" style="background:linear-gradient(135deg,rgba(0,102,255,0.15),rgba(0,210,255,0.08));
                                border-color:rgba(0,210,255,0.3);">
        <div style="font-size:1.3rem; font-weight:700; color:#F1F5F9;">
            👋 Welcome back, {user.get('full_name', user.get('username', 'User'))}!
        </div>
        <div style="color:#94A3B8; font-size:0.85rem; margin-top:0.5rem;">
            You are authenticated as <strong style="color:#00D2FF;">{role_label}</strong>.
            Session started at {_fmt_dt(st.session_state.get('auth_session_started'))}.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──────────────────────────────────────────
    st.markdown("""
    <div class="dv-kpi-grid">
        <div class="dv-kpi">
            <div class="dv-kpi-value">🔬</div>
            <div class="dv-kpi-label">Anomaly Detection</div>
        </div>
        <div class="dv-kpi">
            <div class="dv-kpi-value">99.7%</div>
            <div class="dv-kpi-label">Top AUROC Score</div>
        </div>
        <div class="dv-kpi">
            <div class="dv-kpi-value">1,167</div>
            <div class="dv-kpi-label">Indexed Samples</div>
        </div>
        <div class="dv-kpi">
            <div class="dv-kpi-value">3</div>
            <div class="dv-kpi-label">Product Lines</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Launch DefectVision ────────────────────────────────
    st.markdown('<div class="dv-card">', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-title">🚀 Launch DefectVision Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#94A3B8; font-size:0.85rem; margin-bottom:1rem;">
        Access the full Anomaly Detection dashboard — Live Defect Inspector,
        Cluster Explorer, and Centroid Gallery.
    </div>
    """, unsafe_allow_html=True)
    st.link_button("🔬 Open DefectVision Dashboard →", "http://localhost:8501", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Profile Card ───────────────────────────────────────
    st.markdown('<div class="dv-card">', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-title">👤 Account Profile</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Full Name**  \n{user.get('full_name', '—')}")
        st.markdown(f"**Username**  \n`{user.get('username', '—')}`")
        st.markdown(f"**Role**  \n{role_label}")
    with col2:
        st.markdown(f"**Email**  \n{user.get('email', '—')}")
        st.markdown(f"**Member Since**  \n{_fmt_dt(user.get('created_at'))}")
        st.markdown(f"**Last Login**  \n{_fmt_dt(user.get('last_login'))}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Logout ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out", key="user_logout"):
        logout_user()
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:1rem; color:#334155; font-size:0.72rem; letter-spacing:0.08em;">
        DEFECTVISION · INDUSTRIAL AI PLATFORM · ALL SESSIONS ENCRYPTED
    </div>
    """, unsafe_allow_html=True)
