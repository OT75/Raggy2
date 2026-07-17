import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

type Message = {
  role: "user" | "assistant";
  content: string;
  mode?: string;
  sources?: string[];
};

type DocumentInfo = {
  id: string;
  filename: string;
  chunk_count: number;
};

function App() {
  const [apiChoice, setApiChoice] = useState("Groq (Free)");
  const [userKey, setUserKey] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<{ text: string; ok: boolean } | null>(null);
  const [loading, setLoading] = useState(false);

  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [showDocsModal, setShowDocsModal] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [asking, setAsking] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Adds each selected file as its own document. Only the newly-selected
  // file(s) get embedded — existing documents in the session are untouched,
  // since /api/documents adds to the vectorstore rather than rebuilding it.
  const handleAnalyze = async () => {
    if (!files || files.length === 0) {
      setStatus({ text: "Choose at least one PDF first.", ok: false });
      return;
    }

    setLoading(true);
    setStatus(null);

    let currentSession = sessionId;
    let latestDocs = documents;

    for (const file of Array.from(files)) {
      const formData = new FormData();
      formData.append("file", file);
      if (currentSession) formData.append("session_id", currentSession);
      formData.append("api_choice", apiChoice);
      if (userKey) formData.append("user_key", userKey);

      try {
        const res = await fetch(`${API_BASE}/api/documents`, { method: "POST", body: formData });

        if (!res.ok) {
          const err = await res.json();
          setStatus({ text: `${file.name}: ${err.detail}`, ok: false });
          setLoading(false);
          return;
        }

        const data = await res.json();
        currentSession = data.session_id;
        latestDocs = data.documents;
      } catch {
        setStatus({ text: "Can't reach the backend. Is uvicorn running?", ok: false });
        setLoading(false);
        return;
      }
    }

    setSessionId(currentSession);
    setDocuments(latestDocs);
    setFiles(null);
    setStatus({ text: `${latestDocs.length} document(s) indexed — ready.`, ok: true });
    setLoading(false);
  };

  const handleDeleteDocument = async (docId: string) => {
    if (!sessionId) return;
    setDeletingId(docId);

    try {
      const res = await fetch(`${API_BASE}/api/documents/${sessionId}/${docId}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const err = await res.json();
        setStatus({ text: err.detail, ok: false });
        setDeletingId(null);
        return;
      }

      const data = await res.json();
      setDocuments(data.documents);

      // NEW — refresh the status line to match the actual current count
      setStatus({
        text: data.documents.length > 0
          ? `${data.documents.length} document(s) indexed — ready.`
          : "No documents indexed yet.",
        ok: data.documents.length > 0,
      });
    } catch {
      setStatus({ text: "Couldn't reach the backend to delete.", ok: false });
    }

    setDeletingId(null);
  };
  const handleAsk = async () => {
    if (!input.trim()) return;

    const question = input;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setAsking(true);

    const formData = new FormData();
    if (sessionId) formData.append("session_id", sessionId);
    formData.append("question", question);
    formData.append("api_choice", apiChoice);
    if (userKey) formData.append("user_key", userKey);

    try {
      const res = await fetch(`${API_BASE}/api/ask`, { method: "POST", body: formData });

      if (!res.ok) {
        const err = await res.json();
        setMessages((prev) => [...prev, { role: "assistant", content: `Error: ${err.detail}` }]);
        setAsking(false);
        return;
      }

      const data = await res.json();
      setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer, mode: data.mode, sources: data.sources },
      ]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Couldn't reach the backend." }]);
    }

    setAsking(false);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">💠</div>
          <div>
            <h1>Raggy</h1>
            <p>Document-grounded assistant</p>
          </div>
        </div>

        <div className="field-group">
          <span className="field-label">Provider</span>
          <select value={apiChoice} onChange={(e) => setApiChoice(e.target.value)}>
            <option value="Groq (Free)">Groq (Free) — Llama 3.3 70B</option>
            <option value="OpenAI">OpenAI — GPT-4o mini</option>
          </select>
        </div>

        <div className="field-group">
          <span className="field-label">API Key (optional)</span>
          <div className="key-field-wrap">
            <input
              type="password"
              value={userKey}
              onChange={(e) => setUserKey(e.target.value)}
              placeholder="Uses server default if blank"
            />
            <span className={`key-indicator ${userKey ? "present" : "absent"}`}>
              {userKey ? "●" : "○"}
            </span>
          </div>
        </div>

        <div className="field-group">
          <span className="field-label">Documents</span>
          <label className="dropzone">
            <input
              type="file"
              accept=".pdf"
              multiple
              onChange={(e) => setFiles(e.target.files)}
            />
            <div className="dropzone-icon">📎</div>
            <div className="dropzone-text">
              {files && files.length > 0 ? `${files.length} file(s) selected` : "Drop PDFs or click to browse"}
            </div>
          </label>
          <button className="btn btn-primary" onClick={handleAnalyze} disabled={loading}>
            {loading ? <span className="spinner" /> : "Analyze documents"}
          </button>

          {documents.length > 0 && (
            <button className="docs-btn" onClick={() => setShowDocsModal(true)}>
              <span>📁 Manage documents</span>
              <span className="docs-count">{documents.length}</span>
            </button>
          )}
        </div>

        {status && (
          <div className={`status-line ${status.ok ? "status-ok" : "status-error"}`}>
            {status.text}
          </div>
        )}

        {sessionId && (
          <div className="session-tag mono">session · {sessionId.slice(0, 8)}</div>
        )}

        <div className="sidebar-footer">
          Answers are grounded in your uploaded documents when possible. Ungrounded
          answers are always labeled.
        </div>
      </aside>

      <main className="main">
        <div className="chat-scroll">
          {messages.length === 0 && (
            <div className="empty-state">
              <h2>Ask something</h2>
              <p>
                Upload a document for grounded answers with sources, or just start
                chatting — Raggy works either way.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`msg-row ${msg.role}`}>
              <div className="msg-sender">{msg.role === "user" ? "You" : "Raggy"}</div>
              <div className="bubble">
                {msg.content}
                {msg.mode === "general_fallback" && (
                  <div className="mode-badge fallback">⚠ Not in your documents — general knowledge</div>
                )}
                {msg.mode === "rag" && (
                  <div className="mode-badge rag">✓ Grounded in your documents</div>
                )}
                {msg.sources && msg.sources.length > 0 && (
                  <details className="sources-toggle">
                    <summary>View source chunks ({msg.sources.length})</summary>
                    {msg.sources.map((s, j) => (
                      <div key={j} className="source-chunk mono">{s}...</div>
                    ))}
                  </details>
                )}
              </div>
            </div>
          ))}
          <div ref={scrollRef} />
        </div>

        <div className="composer">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            placeholder="Ask me anything..."
          />
          <button className="btn btn-primary" onClick={handleAsk} disabled={asking}>
            {asking ? <span className="spinner" /> : "Send"}
          </button>
        </div>
      </main>

      {showDocsModal && (
        <div className="modal-overlay" onClick={() => setShowDocsModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Documents in this session</h3>
              <button className="modal-close" onClick={() => setShowDocsModal(false)}>✕</button>
            </div>
            <div className="modal-body">
              {documents.length === 0 ? (
                <div className="modal-empty">No documents uploaded yet.</div>
              ) : (
                documents.map((doc) => (
                  <div key={doc.id} className="doc-row">
                    <div className="doc-info">
                      <span className="doc-name">{doc.filename}</span>
                      <span className="doc-meta mono">{doc.chunk_count} chunks</span>
                    </div>
                    <button
                      className="doc-delete"
                      onClick={() => handleDeleteDocument(doc.id)}
                      disabled={deletingId === doc.id}
                      title="Remove this document"
                    >
                      {deletingId === doc.id ? "…" : "🗑"}
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;