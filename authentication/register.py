"""
DefectVision – Registration Page (embedded, not standalone)
Called from login.py router.
"""

import sys
import re
from pathlib import Path

_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

import streamlit as st
from auth_backend import register_user, validate_password_strength
from session_manager import set_message, show_message
from config.settings import PERMITTED_REGISTRATION_ROLES


def _pw_strength_score(pw: str) -> int:
    score = 0
    if len(pw) >= 8:            score += 1
    if re.search(r"[A-Z]", pw): score += 1
    if re.search(r"[a-z]", pw): score += 1
    if re.search(r"\d", pw):    score += 1
    if re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", pw): score += 1
    return score


_STRENGTH_LABELS = ["", "Very Weak", "Weak", "Fair", "Good", "Strong"]
_STRENGTH_COLORS = ["", "#EF4444", "#F97316", "#EAB308", "#22C55E", "#00D2FF"]


def show():
    # Brand header (compact)
    st.markdown("""
    <div class="dv-brand" style="margin-bottom:1rem;">
        <div class="dv-logo-ring" style="width:52px;height:52px;font-size:1.4rem;">🔬</div>
        <div class="dv-title" style="font-size:1.5rem;">DEFECTVISION</div>
        <div class="dv-subtitle">Create Your Account</div>
    </div>
    """, unsafe_allow_html=True)

    show_message()

    st.markdown('<div class="dv-card">', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-title">✨ New Account Registration</div>', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-desc">Fill in the details below to create your DefectVision account.</div>', unsafe_allow_html=True)

    full_name = st.text_input("👤 Full Name", placeholder="Jane Smith", key="reg_name")
    username  = st.text_input("🏷️ Username", placeholder="jane_smith (letters, digits, underscores)", key="reg_user")
    email     = st.text_input("📧 Email Address", placeholder="jane@company.com", key="reg_email")

    role_options = {v: k for k, v in PERMITTED_REGISTRATION_ROLES.items()}
    role_label = st.selectbox(
        "🎯 Role",
        list(role_options.keys()),
        help="Select your job role. Administrators are assigned separately.",
        key="reg_role",
    )
    role_key = role_options[role_label]

    password = st.text_input("🔑 Password", type="password", placeholder="Minimum 8 chars, uppercase, digit, symbol", key="reg_pw")

    # Password strength indicator
    if password:
        score = _pw_strength_score(password)
        segs_html = "".join(
            f'<div class="pw-seg {"active-" + str(i) if i <= score else ""}"></div>'
            for i in range(1, 6)
        )
        label = _STRENGTH_LABELS[score]
        color = _STRENGTH_COLORS[score]
        st.markdown(
            f'<div class="pw-strength-bar">{segs_html}</div>'
            f'<div class="pw-label" style="color:{color};">{label}</div>',
            unsafe_allow_html=True,
        )

    confirm_pw = st.text_input("🔑 Confirm Password", type="password", placeholder="Re-enter password", key="reg_confirm")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Create Account", key="reg_btn"):
        with st.spinner("Creating account…"):
            ok, msg, user = register_user(full_name, username, email, password, confirm_pw, role_key)
        if ok:
            set_message(f"✅ {msg}", "success")
            st.session_state.auth_page = "login"
            st.rerun()
        else:
            st.error(f"❌ {msg}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dv-link-btn" style="margin-top:0.75rem;">', unsafe_allow_html=True)
    if st.button("← Back to Login", key="reg_back"):
        st.session_state.auth_page = "login"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top:1.5rem; color:#334155; font-size:0.72rem;">
        DEFECTVISION · SECURE REGISTRATION · ALL ACCOUNTS ARE REVIEWED
    </div>
    """, unsafe_allow_html=True)
