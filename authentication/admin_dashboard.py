"""
DefectVision – Admin Dashboard
Admin-only panel for user management and audit logs.
"""

import sys
from pathlib import Path
from datetime import datetime

_AUTH_DIR = Path(__file__).resolve().parent
if str(_AUTH_DIR) not in sys.path:
    sys.path.insert(0, str(_AUTH_DIR))

import streamlit as st
import pandas as pd
from session_manager import require_admin, get_current_user, logout_user
from auth_backend import get_all_users, get_audit_logs, update_user_status, update_user_role
from config.settings import ALL_ROLES


def _fmt_dt(iso_str):
    if not iso_str:
        return "—"
    try:
        return datetime.fromisoformat(iso_str).strftime("%d %b %Y, %H:%M")
    except Exception:
        return iso_str


def show():
    require_admin()
    user = get_current_user()

    _CSS = _AUTH_DIR / "styles.css"
    if _CSS.exists():
        st.markdown(f"<style>{_CSS.read_text()}</style>", unsafe_allow_html=True)

    # ── Admin Header ────────────────────────────────────────
    st.markdown(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem;">
        <div>
            <div style="font-size:1.6rem; font-weight:800; background:linear-gradient(90deg,#EF4444,#FF6B6B);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
                ADMIN CONSOLE
            </div>
            <div style="color:#64748B; font-size:0.78rem; letter-spacing:0.1em;">DEFECTVISION · SYSTEM ADMINISTRATION</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:0.9rem; font-weight:600; color:#E2E8F0;">{user.get('full_name','Administrator')}</div>
            <span class="dv-role-badge dv-role-admin">System Administrator</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fetch data ──────────────────────────────────────────
    all_users = get_all_users()
    total      = len(all_users)
    active     = sum(1 for u in all_users if u["is_active"])
    inactive   = total - active
    admins     = sum(1 for u in all_users if u["role"] == "admin")

    # ── KPI Cards ───────────────────────────────────────────
    st.markdown(f"""
    <div class="dv-kpi-grid" style="grid-template-columns: repeat(4,1fr); margin-bottom:1.5rem;">
        <div class="dv-kpi">
            <div class="dv-kpi-value">{total}</div>
            <div class="dv-kpi-label">Total Users</div>
        </div>
        <div class="dv-kpi">
            <div class="dv-kpi-value" style="font-size:1.6rem; color:#22C55E;">{active}</div>
            <div class="dv-kpi-label">Active</div>
        </div>
        <div class="dv-kpi">
            <div class="dv-kpi-value" style="font-size:1.6rem; color:#EF4444;">{inactive}</div>
            <div class="dv-kpi-label">Inactive</div>
        </div>
        <div class="dv-kpi">
            <div class="dv-kpi-value" style="font-size:1.6rem; color:#EF4444;">{admins}</div>
            <div class="dv-kpi-label">Admins</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Launch main app ─────────────────────────────────────
    st.link_button("🔬 Open DefectVision Platform →", "http://localhost:8501", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── User Management Table ───────────────────────────────
    st.markdown('<div class="dv-card">', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-title">👥 User Management</div>', unsafe_allow_html=True)

    if not all_users:
        st.info("No users found.")
    else:
        for u in all_users:
            is_self = u["user_id"] == user["user_id"]
            with st.expander(
                f"{'🟢' if u['is_active'] else '🔴'} {u['full_name']} — @{u['username']} ({u['role']})"
                + (" · YOU" if is_self else ""),
                expanded=False,
            ):
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(f"**Email:** {u['email']}")
                    st.markdown(f"**Joined:** {_fmt_dt(u['created_at'])}")
                    st.markdown(f"**Last login:** {_fmt_dt(u['last_login'])}")

                with c2:
                    if not is_self:
                        new_role = st.selectbox(
                            "Role",
                            list(ALL_ROLES.keys()),
                            index=list(ALL_ROLES.keys()).index(u["role"]),
                            key=f"role_{u['user_id']}",
                        )
                        if new_role != u["role"]:
                            if st.button("💾 Update Role", key=f"updrole_{u['user_id']}"):
                                ok, msg = update_user_role(user["user_id"], u["user_id"], new_role)
                                st.success(msg) if ok else st.error(msg)
                                st.rerun()
                    else:
                        st.markdown(f"**Role:** {u['role']}")

                with c3:
                    if not is_self and u["role"] != "admin":
                        label = "🚫 Deactivate" if u["is_active"] else "✅ Activate"
                        if st.button(label, key=f"status_{u['user_id']}"):
                            ok, msg = update_user_status(user["user_id"], u["user_id"], not u["is_active"])
                            st.success(msg) if ok else st.error(msg)
                            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Audit Logs ──────────────────────────────────────────
    st.markdown('<div class="dv-card">', unsafe_allow_html=True)
    st.markdown('<div class="dv-card-title">📋 Audit Log (last 50)</div>', unsafe_allow_html=True)

    logs = get_audit_logs(limit=50)
    if logs:
        df = pd.DataFrame(logs)[["timestamp", "username", "action", "details"]]
        df["timestamp"] = df["timestamp"].apply(_fmt_dt)
        df.columns = ["Timestamp", "User", "Action", "Details"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No audit events recorded yet.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Logout ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Sign Out", key="admin_logout"):
        logout_user()
        st.rerun()

    st.markdown("""
    <div style="text-align:center; margin-top:1rem; color:#334155; font-size:0.72rem; letter-spacing:0.08em;">
        DEFECTVISION ADMIN CONSOLE · ALL ACTIONS ARE LOGGED
    </div>
    """, unsafe_allow_html=True)
