from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import uuid

from rag_engine import get_pdf_text, get_text_chunks, get_vectorstore, answer_question

load_dotenv(override=True)

app = FastAPI(title="Raggy API")

# Allow the React dev server to call this API. Covers both common Vite ports
# while developing — in production this should be locked to the actual frontend's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> {vectorstore, messages}
# Fine for a single-user local demo. A real deployment would need persistent,
# multi-user-safe storage (e.g. Redis) instead of a plain process-memory dict.
sessions = {}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def resolve_key(api_choice: str, user_key: str | None) -> str | None:
    env_key = GROQ_API_KEY if api_choice == "Groq (Free)" else OPENAI_API_KEY
    return user_key if user_key else env_key


@app.post("/api/analyze")
async def analyze_documents(
    files: List[UploadFile],
    api_choice: str = Form(...),
    user_key: str = Form(None),
):
    active_key = resolve_key(api_choice, user_key)
    if not active_key:
        raise HTTPException(status_code=400, detail="No API key available.")
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one PDF.")

    raw_text = get_pdf_text([f.file for f in files])
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Couldn't extract text — PDFs may be scanned images.")

    chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(chunks, api_choice, active_key)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "vectorstore": vectorstore,
        "messages": [],
    }

    return {
        "session_id": session_id,
        "chunk_count": len(chunks),
    }


@app.post("/api/ask")
async def ask_question(
    session_id: str = Form(None),
    question: str = Form(...),
    api_choice: str = Form(...),
    user_key: str = Form(None),
):
    active_key = resolve_key(api_choice, user_key)
    if not active_key:
        raise HTTPException(status_code=400, detail="No API key available.")

    if session_id is None or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"vectorstore": None, "messages": []}

    session = sessions[session_id]
    history_for_call = session["messages"].copy()

    answer, sources, mode, debug_info = answer_question(
        session["vectorstore"],
        question,
        api_choice,
        active_key,
        chat_history=history_for_call,
    )

    session["messages"].append({"role": "user", "content": question})
    session["messages"].append({
        "role": "assistant",
        "content": answer,
        "mode": mode,
        "sources": [doc.page_content[:300] for doc in sources],
    })

    return {
        "session_id": session_id,
        "answer": answer,
        "mode": mode,
        "sources": [doc.page_content[:300] for doc in sources],
        "debug": {
            "search_query": debug_info["search_query"],
            "best_score": float(debug_info["best_score"]) if debug_info["best_score"] is not None else None,
        },
    }