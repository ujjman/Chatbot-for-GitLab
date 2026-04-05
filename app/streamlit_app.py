from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure sibling package imports work when Streamlit executes from the app directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.rag_service import RagService
from chatbot.settings import settings


st.set_page_config(page_title="GitLab Handbook Chatbot", page_icon="💬", layout="wide")
st.title("GitLab Handbook & Direction Chatbot")
st.caption("Live chatbot powered by Groq Llama 4 Scout + Firecrawl MCP")


@st.cache_resource
def get_rag_service() -> RagService:
    return RagService()


def _render_sources(sources: list[str]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for source in sources:
        st.markdown(f"- {source}")


def _render_debug_logs(debug_logs: list[str]) -> None:
    if not debug_logs:
        return
    with st.expander("Internal logs", expanded=False):
        st.code("\n".join(debug_logs), language="text")


def _to_chat_history(messages: list[dict]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content:
            history.append({"role": role, "content": content})
    return history


with st.sidebar:
    st.header("Settings")
    site_filter = st.selectbox("Source filter", ["all", "handbook", "direction", "other"], index=0)
    show_internal_logs = st.checkbox("Show internal logs", value=True)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if not settings.groq_api_key:
    st.error("GROQ_API_KEY is missing. Add it to your environment or .env file.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))
            if show_internal_logs:
                _render_debug_logs(msg.get("debug_logs", []))

user_prompt = st.chat_input("Ask about GitLab handbook or upcoming release features...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                rag = get_rag_service()
                result = rag.answer(
                    question=user_prompt,
                    chat_history=_to_chat_history(st.session_state.messages[:-1]),
                    site_filter=site_filter,
                    include_debug=show_internal_logs,
                )
                answer = result["answer"]
                sources = result["sources"]
                debug_logs = result.get("debug_logs", [])
            except Exception as exc:  # noqa: BLE001
                answer = f"Error: {exc}"
                sources = []
                debug_logs = [f"Streamlit layer exception: {exc}"]

        st.markdown(answer)
        _render_sources(sources)
        if show_internal_logs:
            _render_debug_logs(debug_logs)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "debug_logs": debug_logs,
        }
    )
