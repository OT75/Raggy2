import streamlit as st
import os
from dotenv import load_dotenv
from rag_engine import get_pdf_text, get_text_chunks, get_vectorstore, answer_question

load_dotenv(override=True)

st.set_page_config(
    page_title="Raggy AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💠 Raggy")
st.caption("Your own customized RAG system")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("⚙️ Control Panel")

    st.caption("AI Provider")
    api_choice = st.selectbox("Choose Provider", ["Groq (Free)", "OpenAI"], label_visibility="collapsed")

    env_key = GROQ_API_KEY if api_choice == "Groq (Free)" else OPENAI_API_KEY

    st.caption("API Key")
    user_key_input = st.text_input(
        "Bring your own key (optional)",
        type="password",
        placeholder="Leave blank to use the app's default key",
        label_visibility="collapsed"
    )

    active_key = user_key_input if user_key_input else env_key

    if user_key_input:
        st.success(f"✅ Using your own {api_choice} key")
    elif env_key:
        st.success(f"✅ Using default {api_choice} key")
    else:
        st.error(f"⚠️ No key available for {api_choice} — enter one above")

    st.markdown("---")

    st.caption("📎 Documents")
    pdf_docs = st.file_uploader(
        "Drop PDFs here",
        accept_multiple_files=True,
        type="pdf",
        label_visibility="collapsed"
    )

    if st.button("✨ Analyze Documents", type="primary", use_container_width=True):
        if not active_key:
            st.error("No API key available.")
        elif not pdf_docs:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting text..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text) if raw_text.strip() else None
                if not chunks:
                    st.error("Couldn't extract text — PDFs may be scanned images.")

            if chunks:
                with st.spinner("Building knowledge base..."):
                    st.session_state.vectorstore = get_vectorstore(chunks, api_choice, active_key)
                    st.success(f"Knowledge base ready — {len(chunks)} chunks indexed.")

    st.markdown("---")
    if st.button("↺ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

MODE_LABELS = {
    "general": None,
    "general_fallback": "⚠️ Your documents don't cover this — answering from general knowledge.",
    "rag": None,
}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        note = MODE_LABELS.get(msg.get("mode"))
        if note:
            st.caption(note)
        if msg.get("sources"):
            with st.expander("🔍 View source chunks"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i+1}**")
                    st.info(doc.page_content[:300] + "...")

user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    history_for_call = st.session_state.messages.copy()  # snapshot BEFORE this turn
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not active_key:
        with st.chat_message("assistant"):
            st.warning("Please provide an API key in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources, mode, debug_info = answer_question(
                    st.session_state.vectorstore,
                    user_input,
                    api_choice,
                    active_key,
                    chat_history=history_for_call
                )
                st.markdown(answer)
                note = MODE_LABELS.get(mode)
                if note:
                    st.caption(note)
                if debug_info["search_query"]:
                    st.caption(f"🔧 debug — search query: `{debug_info['search_query']}` | best score: {debug_info['best_score']}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "mode": mode,
            "sources": sources
        })