# 💠 Raggy

A document-grounded assistant that answers questions from uploaded PDFs — with
transparent citations, and honest disclosure when an answer *isn't* backed by
your documents.

## The problem this solves

Most simple RAG demos silently blend "answered from your documents" and
"answered from the model's general knowledge" into one response, with no way
to tell which is which. That's a real trust problem: if the tool's whole
value is "I'll tell you what's in your documents," a user needs to know when
it *isn't* doing that.

Raggy routes every question through one of three explicit modes:

| Mode | When | Behavior |
|---|---|---|
| `general` | No documents uploaded | Plain LLM chat |
| `general_fallback` | Documents exist, but don't cover the question | General-knowledge answer, clearly labeled as not grounded |
| `rag` | Documents exist and cover the question | Answer grounded in retrieved chunks, with sources shown |

Routing is decided by a FAISS similarity score against a threshold — not by
asking the LLM to self-report relevance — so it's an inspectable number, not
a black box.

## Architecture

```
Raggy2/
├── backend/           FastAPI API — session-based, wraps rag_engine
│   ├── main.py
│   ├── rag_engine.py  Framework-agnostic RAG logic (no web imports)
│   └── requirements.txt
├── frontend/           React + TypeScript (Vite)
│   └── src/App.tsx
└── streamlit_app/       Original Streamlit prototype, kept as a fallback demo
    └── app.py
```

`rag_engine.py` has zero dependency on Streamlit or FastAPI — both `main.py`
and the Streamlit app call the exact same functions. That separation is
deliberate: the RAG logic required no changes at all when the project moved
from a single-script prototype to a client/server architecture.

### Backend (`backend/`)

- **FastAPI** — session-based (in-memory, single-user-per-session), with
  endpoints to add/list/delete documents and to ask questions.
- **Incremental indexing** — adding a document only embeds *that* document's
  chunks (`FAISS.add_texts`), not the whole knowledge base. Deleting a
  document rebuilds the index from the remaining documents' already-chunked
  text (cheap — no re-parsing PDFs, no re-embedding what wasn't touched
  conceptually, just the survivors).
- **Query contextualization** — follow-up questions ("what about that team?")
  are rewritten into standalone questions using recent chat history before
  retrieval, with a guardrail that discards rewrites that balloon in length
  (a real failure mode encountered during development: the rewrite step
  occasionally over-expanded simple questions instead of doing a minimal
  edit).
- **Fallback safety** — when a question isn't grounded, the model is
  explicitly instructed not to fabricate specific facts, names, or figures —
  added after observing the model invent a plausible-but-fake WiFi password
  rather than admit it didn't know.

### Frontend (`frontend/`)

React + TypeScript via Vite, talking to the backend over plain `fetch` +
`FormData`. No component framework — a from-scratch UI with a document
manager (add/remove individual PDFs, running index) and mode badges that
surface which of the three response modes produced each answer.

### Streamlit app (`streamlit_app/`)

The original prototype, kept working and unchanged in logic. Useful as a
fast fallback demo and as a before/after reference point for the
architecture evolution.

## Running it

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Streamlit (standalone alternative):**
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

All three need a `.env` file with `GROQ_API_KEY` and/or `OPENAI_API_KEY`
(`backend/.env`, mirrored in `streamlit_app/.env`). A key can also be
supplied per-session in the UI, which takes priority over the `.env`
default.

## Known limitations / deliberate tradeoffs

- **Short queries can under-match.** A bare "what's the salary?" can score
  worse against document text than a phrased-out equivalent, since a small
  local embedding model doesn't always bridge vocabulary gaps (e.g.
  "salary" vs. the document's own word, "stipend"). A topic-summary
  enrichment step is used as a fallback retrieval attempt for this case,
  gated behind a stricter score bar so it doesn't inflate matches for truly
  unrelated questions.
- **In-memory sessions.** Fine for a local demo; a real deployment would need
  persistent, multi-user-safe session storage.
- **Hand-rolled retrieval pipeline**, not LangChain's built-in
  `ConversationalRetrievalChain` / `create_history_aware_retriever`. This was
  a deliberate choice to get full visibility into the retrieval →
  generation boundary while debugging the grounding-disclosure bug above —
  a prebuilt chain would have hidden that step. A production version would
  likely migrate to the built-in retriever now that the logic is stable.

## Stack

Python, FastAPI, LangChain, FAISS, HuggingFace/OpenAI embeddings, Groq
(Llama 3.3 70B) / OpenAI (GPT-4o mini), React, TypeScript, Vite, Streamlit.
