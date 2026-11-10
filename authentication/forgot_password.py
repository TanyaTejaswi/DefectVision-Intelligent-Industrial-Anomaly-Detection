"""
DefectVision – Forgot Password Page (embedded, not standalone)
Called from login.py router.
"""

import sys
from pathlib import Path

_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

import streamlit as st
from auth_backend import create_reset_token, reset_password_with_token
from session_manager import show_message, set_message


def show():
    st.markdown("""
    <div class="dv-brand" style="margin-bottom:1rem;">
        <div class="dv-logo-ring" style="width:52px;height:52px;font-size:1.4rem;">🔒</div>
        <div class="dv-title" style="font-size:1.5rem;">DEFECTVISION</div>
        <div class="dv-subtitle">Password Recovery</div>
    </div>
    """, unsafe_allow_html=True)

    show_message()

    step = st.session_state.get("fp_step", 1)

    if step == 1:
        # ── Step 1: Enter email ──────────────────────────────
        st.markdown('<div class="dv-card">', unsafe_allow_html=True)
        st.markdown('<div class="dv-card-title">🔑 Reset Your Password</div>', unsafe_allow_html=True)
        st.markdown('<div class="dv-card-desc">Enter your registered email address to receive a reset token.</div>', unsafe_allow_html=True)

        email = st.text_input("📧 Email Address", placeholder="you@company.com", key="fp_email")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("📤 Send Reset Token", key="fp_send"):
            if not email.strip():
                st.error("⚠️ Please enter your email address.")
            else:
                with st.spinner("Processing…"):
                    ok, msg, raw_token = create_reset_token(email)
                if ok:
                    if msg == "__DEV_MODE__":
                        st.warning("⚠️ **DEVELOPMENT MODE** — SMTP not configured. Your reset token is shown below instead of being emailed. Remove this in production.")
                        st.code(raw_token, language=None)
                        st.info("Copy the token above, then click **Enter Reset Token** below.")
                        st.session_state.fp_step = 2
                        st.rerun()
                    else:
                        set_message(msg, "success")
                        st.session_state.fp_step = 2
                        st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ── Step 2: Enter token + new password ──────────────
        st.markdown('<div class="dv-card">', unsafe_allow_html=True)
        st.markdown('<div class="dv-card-title">🔐 Enter New Password</div>', unsafe_allow_html=True)
        st.markdown('<div class="dv-card-desc">Paste your reset token and choose a new password.</div>', unsafe_allow_html=True)

        show_message()

        token    = st.text_input("🎫 Reset Token", placeholder="Paste your reset token here", key="fp_token")
        new_pw   = st.text_input("🔑 New Password", type="password", placeholder="Min 8 chars, uppercase, digit, symbol", key="fp_new_pw")
        confirm  = st.text_input("🔑 Confirm Password", type="password", placeholder="Re-enter new password", key="fp_confirm")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("✅ Reset Password", key="fp_reset_btn"):
            if not token or not new_pw or not confirm:
                st.error("⚠️ Please fill in all fields.")
            else:
                with st.spinner("Resetting password…"):
                    ok, msg = reset_password_with_token(token, new_pw, confirm)
                if ok:
                    set_message(f"✅ {msg}", "success")
                    st.session_state.fp_step = 1
                    st.session_state.auth_page = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="dv-link-btn" style="margin-top:0.5rem;">', unsafe_allow_html=True)
        if st.button("← Try a different email", key="fp_back_step1"):
            st.session_state.fp_step = 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Back to login
    st.markdown('<div class="dv-link-btn" style="margin-top:0.75rem;">', unsafe_allow_html=True)
    if st.button("← Back to Login", key="fp_back_login"):
        st.session_state.fp_step = 1
        st.session_state.auth_page = "login"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
