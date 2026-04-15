(function () {
  const e = React.createElement;
  const { useEffect, useRef, useState } = React;

  function App() {
    const [messages, setMessages] = useState([
      {
        role: "assistant",
        text:
          "Document chat is ready. Ask a question about your indexed documents.",
        sources: [],
      },
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const feedRef = useRef(null);

    useEffect(() => {
      const node = feedRef.current;
      if (node) {
        node.scrollTop = node.scrollHeight;
      }
    }, [messages, loading]);

    async function sendMessage() {
      const trimmed = input.trim();
      if (!trimmed || loading) return;

      setError("");
      setLoading(true);
      setMessages((prev) => prev.concat([{ role: "user", text: trimmed }]));
      setInput("");

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: trimmed }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Request failed");
        }
        setMessages((prev) =>
          prev.concat([
            {
              role: "assistant",
              text: data.answer || "",
              sources: Array.isArray(data.sources) ? data.sources : [],
            },
          ]),
        );
      } catch (err) {
        setError(err.message || "Unexpected error");
      } finally {
        setLoading(false);
      }
    }

    function onKeyDown(event) {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    }

    return e(
      "div",
      { className: "app-shell" },
      e(
        "aside",
        { className: "sidebar" },
        e("div", { className: "sidebar-title" }, "Document Chat"),
        e(
          "button",
          {
            className: "new-chat-btn",
            onClick: () => {
              setMessages([
                {
                  role: "assistant",
                  text:
                    "Document chat is ready. Ask a question about your indexed documents.",
                  sources: [],
                },
              ]);
              setError("");
            },
          },
          "+ New chat",
        ),
        e(
          "p",
          { className: "sidebar-note" },
          "RAG-backed answers with source citations.",
        ),
      ),
      e(
        "main",
        { className: "main" },
        e(
          "header",
          { className: "topbar" },
          e("h1", { className: "topbar-title" }, "Documents"),
          e("span", { className: "model-pill" }, "qwen2.5:7b"),
        ),
        e(
          "section",
          { className: "chat-area", ref: feedRef },
          messages.map((msg, idx) => {
            const isAssistant = msg.role === "assistant";
            return e(
              "article",
              { className: `row ${msg.role}`, key: `${msg.role}-${idx}` },
              e("div", { className: "avatar" }, isAssistant ? "AI" : "You"),
              e(
                "div",
                { className: "bubble-wrap" },
                e("div", { className: `bubble ${msg.role}` }, msg.text),
                isAssistant &&
                  msg.sources &&
                  msg.sources.length > 0 &&
                  e(
                    "div",
                    { className: "sources" },
                    e("strong", null, "Sources"),
                    msg.sources.map((s, sidx) =>
                      e(
                        "div",
                        { className: "source-item", key: `source-${idx}-${sidx}` },
                        `${sidx + 1}. ${s.title || "Untitled"} | ${
                          s.section || "N/A"
                        } | ${s.procedure_code || "N/A"}`,
                      ),
                    ),
                  ),
              ),
            );
          }),
          loading &&
            e(
              "article",
              { className: "row assistant" },
              e("div", { className: "avatar" }, "AI"),
              e("div", { className: "bubble assistant typing" }, "Analyzing documents..."),
            ),
        ),
        e(
          "section",
          { className: "composer-wrap" },
          e(
            "div",
            { className: "composer" },
            e("textarea", {
              value: input,
              placeholder: "Message Document Chat...",
              onChange: (ev) => setInput(ev.target.value),
              onKeyDown,
            }),
            e(
              "button",
              { onClick: sendMessage, disabled: loading || !input.trim() },
              "Send",
            ),
          ),
          error ? e("p", { className: "error" }, error) : null,
        ),
      ),
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(e(App));
})();
