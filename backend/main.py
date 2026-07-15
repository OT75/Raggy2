from typing import List
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
import uuid

from rag_engine import get_pdf_text, get_text_chunks, get_embeddings, answer_question

load_dotenv(override=True)

app = FastAPI(title="Raggy API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store. Each session tracks:
#   vectorstore: the live FAISS index (or None if no documents yet)
#   documents:   doc_id -> {filename, chunk_count, chunks} — "chunks" is kept
#                so a delete can rebuild the index from the remaining documents
#                without re-reading or re-extracting any PDF.
#   messages:    chat history, same shape as before
# Fine for a single-user local demo — a real deployment needs persistent,
# multi-user-safe storage instead of a process-memory dict.
sessions = {}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def resolve_key(api_choice: str, user_key: str | None) -> str | None:
    env_key = GROQ_API_KEY if api_choice == "Groq (Free)" else OPENAI_API_KEY
    return user_key if user_key else env_key


def get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    if session_id is None or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"vectorstore": None, "documents": {}, "messages": []}
    return session_id, sessions[session_id]


def serialize_documents(session: dict):
    return [
        {"id": doc_id, "filename": d["filename"], "chunk_count": d["chunk_count"]}
        for doc_id, d in session["documents"].items()
    ]


@app.post("/api/documents")
async def add_document(
    file: UploadFile,
    session_id: str = Form(None),
    api_choice: str = Form(...),
    user_key: str = Form(None),
):
    """Adds a single document to the session's knowledge base. Only this
    document's chunks get embedded — existing documents in the vectorstore
    are untouched, via FAISS's add_texts rather than rebuilding from scratch."""
    active_key = resolve_key(api_choice, user_key)
    if not active_key:
        raise HTTPException(status_code=400, detail="No API key available.")

    session_id, session = get_or_create_session(session_id)

    raw_text = get_pdf_text([file.file])
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Couldn't extract text — file may be a scanned image.")

    chunks = get_text_chunks(raw_text)
    embeddings = get_embeddings(api_choice, active_key)

    if session["vectorstore"] is None:
        session["vectorstore"] = FAISS.from_texts(texts=chunks, embedding=embeddings)
    else:
        session["vectorstore"].add_texts(chunks)

    doc_id = str(uuid.uuid4())
    session["documents"][doc_id] = {
        "filename": file.filename,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    return {
        "session_id": session_id,
        "documents": serialize_documents(session),
    }


@app.get("/api/documents/{session_id}")
async def list_documents(session_id: str):
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"documents": serialize_documents(session)}


@app.delete("/api/documents/{session_id}/{doc_id}")
async def delete_document(
    session_id: str,
    doc_id: str,
    api_choice: str = "Groq (Free)",
    user_key: str = None,
):
    """Removes one document and rebuilds the vectorstore from the remaining
    documents' already-chunked text. This re-embeds the survivors (cheap —
    no PDF re-parsing, and local embeddings are free) rather than attempting
    row-level deletion inside FAISS, which is more fragile to get right."""
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if doc_id not in session["documents"]:
        raise HTTPException(status_code=404, detail="Document not found.")

    active_key = resolve_key(api_choice, user_key)
    if not active_key:
        raise HTTPException(status_code=400, detail="No API key available to rebuild the index.")

    del session["documents"][doc_id]

    remaining_chunks = [
        chunk for doc in session["documents"].values() for chunk in doc["chunks"]
    ]

    if remaining_chunks:
        embeddings = get_embeddings(api_choice, active_key)
        session["vectorstore"] = FAISS.from_texts(texts=remaining_chunks, embedding=embeddings)
    else:
        session["vectorstore"] = None

    return {"documents": serialize_documents(session)}


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

    session_id, session = get_or_create_session(session_id)
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
            "best_score": debug_info["best_score"],
        },
    }