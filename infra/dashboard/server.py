#!/usr/bin/env python3
"""DriftWatch dashboard server — stdlib only, no extra deps."""

import json
import subprocess
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

PORT = 8765
ROOT = Path(__file__).resolve().parent.parent.parent  # infra/dashboard/ -> infra/ -> project root
API = "http://localhost:8000"

ALLOWED_COMMANDS = {
    "demo-drift-feature",
    "demo-black-friday",
    "gen-base",
    "gen-feature",
    "gen-blackfriday",
    "train",
    "promote-prod",
    "monitor",
    "control",
    "rollback",
}

COMMAND_LOCK = Lock()

SERVICE_MATCHES = {
    "mlflow": "mlflow",
    "api": "-api-",
    "prometheus": "prometheus",
    "grafana": "grafana",
}

# ──────────────────────────────────────────────────────────────────────────────
HTML_PATH = ROOT / "dashboard.html"


# ──────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", ""):
            self._html()
        elif path == "/api/status":
            self._status()
        elif path == "/api/health":
            self._proxy_get(f"{API}/health")
        else:
            self._respond(404, b"Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        if path.startswith("/api/run/"):
            self._stream(path[len("/api/run/") :])
        elif path == "/api/predict":
            self._proxy(f"{API}/predict")
        elif path == "/api/reload":
            self._proxy(f"{API}/reload")
        else:
            self._respond(404, b"Not found")

    # ── Routes ────────────────────────────────────────────────────────────────

    def _html(self):
        body = HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _status(self):
        try:
            out = subprocess.check_output(
                ["docker", "ps", "--format", "{{.Names}}"],
                timeout=5,
                text=True,
            )
            running = [line for line in out.strip().split("\n") if line]
            status = {svc: any(m in c for c in running) for svc, m in SERVICE_MATCHES.items()}
        except Exception:
            status = {svc: False for svc in SERVICE_MATCHES}
        body = json.dumps(status).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _proxy_get(self, url: str):
        try:
            with urllib.request.urlopen(url, timeout=4) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data)
        except Exception as e:
            self._respond(502, json.dumps({"error": str(e), "model_loaded": False}).encode())

    def _proxy(self, url: str):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._respond(502, json.dumps({"error": str(e)}).encode())

    def _stream(self, cmd: str):
        if cmd not in ALLOWED_COMMANDS:
            self._respond(400, b"Command not allowed")
            return
        if not COMMAND_LOCK.acquire(blocking=False):
            self._respond(409, b"Another pipeline command is already running")
            return
        proc = None
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            proc = subprocess.Popen(
                ["make", "-C", str(ROOT), cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            connected = True
            for line in proc.stdout:
                if connected:
                    try:
                        self._sse({"line": line.rstrip()})
                    except (BrokenPipeError, ConnectionResetError):
                        # Drain output so a disconnected browser cannot block make.
                        connected = False
            proc.wait()
            if connected:
                self._sse({"exit": proc.returncode})
        except OSError as e:
            try:
                self._sse({"line": str(e)})
                self._sse({"exit": 1})
            except (BrokenPipeError, ConnectionResetError):
                pass
        finally:
            if proc is not None:
                proc.stdout.close()
                proc.wait()
            COMMAND_LOCK.release()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sse(self, data: dict):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def _respond(self, code: int, body: bytes):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"DriftWatch dashboard → http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
