import json
import os
import sys
import textwrap
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service.qa_service import QAService

HOST = "127.0.0.1"
PORT = 8787
WEB_DIR = Path(__file__).resolve().parent / "web"


def render_box(title: str, body: str) -> None:
    width = 78
    print("+" + "-" * width + "+")
    print(f"| {title[: width - 2].ljust(width - 1)}|")
    print("+" + "-" * width + "+")
    for line in textwrap.wrap(body, width=width - 2) or [""]:
        print(f"| {line.ljust(width - 2)} |")
    print("+" + "-" * width + "+")


class DocumentChatHandler(SimpleHTTPRequestHandler):
    qa_service: QAService | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/chat":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        if content_len <= 0:
            self._send_json({"error": "Request body is required"}, HTTPStatus.BAD_REQUEST)
            return

        try:
            body = self.rfile.read(content_len)
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Invalid JSON payload"}, HTTPStatus.BAD_REQUEST)
            return

        question = str(payload.get("message", "")).strip()
        if not question:
            self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
            return

        history = payload.get("history", [])
        if not isinstance(history, list):
            history = []

        if self.qa_service is None:
            self._send_json({"error": "QA service not initialized"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        try:
            result = self.qa_service.ask(question, history=history)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {
                    "error": "Failed to generate an answer. "
                    "Ensure corpus/index exist and Ollama is running.",
                    "details": str(exc),
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json(
            {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
            },
            HTTPStatus.OK,
        )

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"status": "ok"}, HTTPStatus.OK)
            return
        super().do_GET()

    def _send_json(self, payload: dict, status: HTTPStatus) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_cli() -> None:
    service = QAService()
    render_box(
        "DOCUMENT CHAT (RAG CLI)",
        "Ask document questions.\nCommands: /exit",
    )
    history: list[dict] = []
    while True:
        try:
            question = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question in {"/exit", "exit", "quit"}:
            print("Goodbye.")
            break

        try:
            result = service.ask(question, history=history)
        except Exception as exc:  # noqa: BLE001
            render_box("ERROR", str(exc))
            continue

        answer = result.get("answer", "")
        render_box("ASSISTANT", answer)
        sources = result.get("sources", [])
        if sources:
            lines = []
            for idx, source in enumerate(sources, start=1):
                lines.append(
                    f"[{idx}] {source.get('title')} | {source.get('section')} | "
                    f"{source.get('procedure_code')}"
                )
                lines.append(f"    {source.get('source_url')}")
            render_box("SOURCES", "\n".join(lines))

        history.append({"role": "user", "text": question})
        history.append({"role": "assistant", "text": answer})


def run_server(host: str = HOST, port: int = PORT) -> None:
    if not WEB_DIR.exists():
        raise RuntimeError(f"Missing frontend directory: {WEB_DIR}")

    DocumentChatHandler.qa_service = QAService()
    server = ThreadingHTTPServer((host, port), DocumentChatHandler)
    print(f"Document chat running on http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


def main() -> None:
    os.chdir(PROJECT_ROOT)

    mode = "serve"
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()

    if mode == "cli":
        run_cli()
        return
    if mode == "serve":
        run_server()
        return

    raise SystemExit("Usage: python app.py [serve|cli]")


if __name__ == "__main__":
    main()
