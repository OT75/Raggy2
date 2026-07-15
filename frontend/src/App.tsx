import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

type Message = {
  role: "user" | "assistant";
  content: string;
  mode?: string;
  sources?: string[];
};

const MODE_LABELS: Record<string, string> = {
  general_fallback: "⚠️ Your documents don't cover this — answering from general knowledge.",
};

function App() {
  const [apiChoice, setApiChoice] = useState("Groq (Free)");
  const [userKey, setUserKey] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [asking, setAsking] = useState(false);

  const handleAnalyze = async () => {
    if (!files || files.length === 0) {
      setStatus("Please choose at least one PDF.");
      return;
    }

    setLoading(true);
    setStatus("");

    const formData = new FormData();
    for (const file of Array.from(files)) {
      formData.append("files", file);
    }
    formData.append("api_choice", apiChoice);
    if (userKey) formData.append("user_key", userKey);

    try {
      const res = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        setStatus(`Error: ${err.detail}`);
        setLoading(false);
        return;
      }

      const data = await res.json();
      setSessionId(data.session_id);
      setStatus(`Knowledge base ready — ${data.chunk_count} chunks indexed.`);
    } catch (e) {
      setStatus("Failed to reach the backend. Is uvicorn running?");
    }

    setLoading(false);
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
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        body: formData,
      });

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
    } catch (e) {
      setMessages((prev) => [...prev, { role: "assistant", content: "Failed to reach the backend." }]);
    }

    setAsking(false);
  };

  return (
    <div style={{ maxWidth: 700, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>💠 Raggy</h1>
      <p>Your own customized RAG system</p>

      <div style={{ marginBottom: 16 }}>
        <label>
          Provider:{" "}
          <select value={apiChoice} onChange={(e) => setApiChoice(e.target.value)}>
            <option value="Groq (Free)">Groq (Free)</option>
            <option value="OpenAI">OpenAI</option>
          </select>
        </label>
      </div>

      <div style={{ marginBottom: 16 }}>
        <label>
          API Key (optional):{" "}
          <input
            type="password"
            value={userKey}
            onChange={(e) => setUserKey(e.target.value)}
            placeholder="Leave blank for default"
          />
        </label>
      </div>

      <div style={{ marginBottom: 16 }}>
        <input type="file" accept=".pdf" multiple onChange={(e) => setFiles(e.target.files)} />
        <button onClick={handleAnalyze} disabled={loading} style={{ marginLeft: 8 }}>
          {loading ? "Analyzing..." : "✨ Analyze Documents"}
        </button>
      </div>

      {status && <p>{status}</p>}

      <hr style={{ margin: "24px 0" }} />

      <div style={{ minHeight: 200, marginBottom: 16 }}>
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              marginBottom: 12,
              padding: 10,
              borderRadius: 8,
              background: msg.role === "user" ? "#eef" : "#f5f5f5",
            }}
          >
            <strong>{msg.role === "user" ? "You" : "Raggy"}:</strong> {msg.content}
            {msg.mode && MODE_LABELS[msg.mode] && (
              <div style={{ fontSize: 12, color: "#a60", marginTop: 4 }}>
                {MODE_LABELS[msg.mode]}
              </div>
            )}
            {msg.sources && msg.sources.length > 0 && (
              <details style={{ marginTop: 6, fontSize: 12 }}>
                <summary>🔍 View source chunks</summary>
                {msg.sources.map((s, j) => (
                  <p key={j} style={{ background: "#fff", padding: 6, borderRadius: 4 }}>
                    {s}...
                  </p>
                ))}
              </details>
            )}
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          placeholder="Ask me anything..."
          style={{ flex: 1, padding: 8 }}
        />
        <button onClick={handleAsk} disabled={asking}>
          {asking ? "Thinking..." : "Send"}
        </button>
      </div>
    </div>
  );
}

export default App;