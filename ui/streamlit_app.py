"""BoltSound Assistant: customer chat on the left, 'under the hood' on the right.

Run:  streamlit run ui/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_settings  # noqa: E402
from app.factory import build_assistant  # noqa: E402

ICONS = {"ok": "✅", "warn": "⚠️", "blocked": "⛔", "error": "❌"}

st.set_page_config(page_title="BoltSound Assistant", page_icon="🎧", layout="wide")


@st.cache_resource
def get_assistant():
    return build_assistant(load_settings())


if "history" not in st.session_state:
    st.session_state.history = []  # list of (message, reply)

chat_col, hood_col = st.columns([3, 2], gap="large")

with chat_col:
    st.title("🎧 BoltSound Support")
    for message, reply in st.session_state.history:
        st.chat_message("user").write(message)
        st.chat_message("assistant", avatar="🤖").write(reply.text)

with hood_col:
    st.subheader("Under the hood")
    if st.session_state.history:
        _, last = st.session_state.history[-1]
        for step in last.trace:
            st.markdown(f"{ICONS.get(step.status, '•')} **{step.name}** · {step.ms:.0f} ms  \n{step.detail}")
        st.caption(f"Total {last.total_ms:.0f} ms · estimated cost ${last.cost_usd:.5f}")
    else:
        st.caption("Send a message to see every step it goes through.")

customer = st.sidebar.selectbox("Logged in as", ["Riya", "Arjun", "Meera"])

if prompt := st.chat_input("Ask BoltSound support…"):
    st.session_state.history.append((prompt, get_assistant().answer(prompt, customer)))
    st.rerun()
