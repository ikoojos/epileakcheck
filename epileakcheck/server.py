"""Minimal HTTP API (standard library only).

    POST /scan   {"source": "<python code>", "filename": "train.py", "rules": ["CL-07"]}
                 or {"path": "/abs/path/on/server"}            -> report JSON
    GET  /rules                                                -> rule list
    GET  /explain/<rule-or-pattern>                            -> what / why / how / example
    GET  /health                                               -> {"status": "ok"}

Intended for local or internal use; put it behind your own auth before exposing it.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import __version__, list_rules, scan_path, scan_source

MAX_BODY = 5 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = f"EpiLeakCheck/{__version__}"

    def _send(self, code: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            return self._send(200, {"status": "ok", "version": __version__})
        if self.path == "/rules":
            return self._send(200, list_rules())
        if self.path.startswith("/explain/"):
            from .explain import E
            key = self.path.split("/explain/", 1)[1]
            if key in E:
                return self._send(200, {"pattern": key, **E[key]})
            return self._send(404, {"error": f"unknown pattern '{key}'"})
        return self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/scan":
            return self._send(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            return self._send(413, {"error": "body too large"})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        rules = req.get("rules")
        conf = req.get("min_confidence", "low")
        try:
            if "source" in req:
                rep = scan_source(req["source"], req.get("filename", "<request>"), rules, conf)
            elif "path" in req and self.server.allow_paths:  # type: ignore[attr-defined]
                rep = scan_path(req["path"], rules, conf)
            else:
                return self._send(400, {"error": "send 'source' (or 'path' when the server allows it)"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        return self._send(200, rep.to_dict())

    def log_message(self, fmt, *args):  # quieter logs
        pass


def serve(host: str = "127.0.0.1", port: int = 8765, allow_paths: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.allow_paths = allow_paths  # type: ignore[attr-defined]
    print(f"EpiLeakCheck API on http://{host}:{port}  (POST /scan, GET /rules)")
    httpd.serve_forever()
