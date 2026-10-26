"""BoltSound Assistant: customer chat with 'under the hood', plus a monitoring dashboard.

Run:  streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_settings  # noqa: E402
from app.factory import build_assistant  # noqa: E402
from app.monitoring import summarize  # noqa: E402

ICONS = {"ok": "✅", "warn": "⚠️", "blocked": "⛔", "error": "❌"}

st.set_page_config(page_title="BoltSound Assistant", page_icon="🎧", layout="wide")


@st.cache_resource
def get_assistant():
    return build_assistant(load_settings())


def chat_tab(customer: str) -> None:
    chat_col, hood_col = st.columns([3, 2], gap="large")
    with chat_col:
        for message, reply in st.session_state.history:
            st.chat_message("user").write(message)
            st.chat_message("assistant", avatar="🤖").write(reply.text)
    with hood_col:
        st.subheader("Under the hood")
        if not st.session_state.history:
            st.caption("Send a message to see every step it goes through.")
            return
        _, last = st.session_state.history[-1]
        for step in last.trace:
            st.markdown(f"{ICONS.get(step.status, '•')} **{step.name}** · {step.ms:.0f} ms  \n{step.detail}")
        st.caption(
            f"Route: {last.route} · total {last.total_ms:.0f} ms · estimated cost ${last.cost_usd:.5f}"
        )


def dashboard_tab() -> None:
    records = get_assistant().monitor.read()
    if not records:
        st.caption("No traffic yet.")
        return
    s = summarize(records)
    cols = st.columns(5)
    cols[0].metric("Messages", s.messages)
    cols[1].metric("Median time", f"{s.p50_ms / 1000:.1f} s")
    cols[2].metric("Slowest 5% over", f"{s.p95_ms / 1000:.1f} s")
    cols[3].metric("Cost", f"${s.cost_usd:.4f}")
    cols[4].metric("Handed to humans", f"{s.handover_rate:.0%}")
    left, right = st.columns(2)
    left.markdown("**Messages by route**")
    left.bar_chart(s.routes, horizontal=True)
    right.markdown("**Time per message (s)**")
    right.line_chart([r["ms"] / 1000 for r in records])
    st.markdown(f"**Blocked:** {s.blocked} · **Model errors:** {s.errors}")
    st.dataframe(
        [
            {k: r[k] for k in ("time", "customer", "route", "ms", "cost_usd", "message")}
            for r in reversed(records)
        ],
        use_container_width=True,
    )


if "history" not in st.session_state:
    st.session_state.history = []  # list of (message, reply)

customer = st.sidebar.selectbox("Logged in as", ["Riya", "Arjun", "Meera"])
st.title("🎧 BoltSound Support")
chat, dashboard = st.tabs(["Chat", "Dashboard"])
with chat:
    chat_tab(customer)
with dashboard:
    dashboard_tab()

if prompt := st.chat_input("Ask BoltSound support…"):
    st.session_state.history.append((prompt, get_assistant().answer(prompt, customer)))
    st.rerun()
