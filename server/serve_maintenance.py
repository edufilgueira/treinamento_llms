#!/usr/bin/env python3
"""
Serve apenas a página de manutenção em HTTP 503 para qualquer rota.

Uso típico na VPS (parar o Oráculo e ocupar a mesma porta):

  # Parar o app principal (ex.: kill do uvicorn ou systemctl stop ...)
  python3 server/serve_maintenance.py --host 0.0.0.0 --port 8765

A porta omissão 8765 iguala a do `server/main.py`. Só usa a biblioteca padrão.

  # Windows (raiz do repo):
  python server/serve_maintenance.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
_HTML_PATH = _SERVER_DIR / "static" / "maintenance.html"


class MaintenanceHandler(BaseHTTPRequestHandler):
    _body_html: bytes = b""
    _body_json: bytes = b""

    @classmethod
    def load_files(cls) -> None:
        if not _HTML_PATH.is_file():
            raise FileNotFoundError(f"Ficheiro em falta: {_HTML_PATH}")
        cls._body_html = _HTML_PATH.read_bytes()
        payload = {
            "error": "service_unavailable",
            "message": "Serviço em manutenção. Volta a tentar mais tarde.",
        }
        cls._body_json = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _want_json(self) -> bool:
        accept = (self.headers.get("Accept") or "").lower()
        return "application/json" in accept

    def _send(self) -> None:
        if self._want_json():
            self.send_response(503, "Service Unavailable")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Retry-After", "3600")
            self.end_headers()
            self.wfile.write(self.__class__._body_json)
            return
        self.send_response(503, "Service Unavailable")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Retry-After", "3600")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(self.__class__._body_html)

    def do_GET(self) -> None:
        self._send()

    def do_HEAD(self) -> None:
        if self._want_json():
            self.send_response(503, "Service Unavailable")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Retry-After", "3600")
            self.end_headers()
            return
        self.send_response(503, "Service Unavailable")
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Retry-After", "3600")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_POST(self) -> None:
        self._send()


def main() -> None:
    MaintenanceHandler.load_files()
    p = argparse.ArgumentParser(description="Página de manutenção (HTTP 503).")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    server = HTTPServer((args.host, args.port), MaintenanceHandler)
    print(
        f"Manutenção activa em http://{args.host}:{args.port}/  (503 em todas as rotas). Ctrl+C para parar.",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
