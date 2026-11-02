"""`thinkjev-serve`: a local HTTP server with Jev's request format and a demo page.

    POST /v1/systemone   {"state": ..., "questions": {...}, "model": ...}   (TypeSafe's public format)
    GET  /v1/thinkjev/info   thresholds, checkpoints, budget, and the calibration report's test curve
    GET  /               demo page: System 1, then one thought read at every checkpoint, live

With `"thinkjev": {"stream": true}` the same POST answers as a server-sent event stream: one
`data:` line per event (`s1`, `thinking`, `checkpoint`, `answer`), then `done` with the full
response, so the demo page can draw each budget's answer while the model is still thinking.

Existing code written against TypeSafe's SDK can point at this server through the SDK's base-URL
setting (`TYPESAFE_BASE_URL=http://127.0.0.1:8000`). The `model` field is accepted and ignored;
the response names the local model. An optional `"thinkjev": {"think": "auto"|"never"|"always",
"include_thought": bool}` field in the request controls escalation per call; `--think-default`
sets the mode for requests without it (clients such as benchmark harnesses that send only
TypeSafe's fields). Standard library only; requests are served one model call at a time.
"""

from __future__ import annotations

import argparse
import queue
import threading
import json
import socketserver
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .api import THINK_MODES, RequestError, ThinkJev, preset, presets
from ._demo import DEMO_PAGE
from .cli import parse_checkpoints

DEFAULT_CHECKPOINTS = (0, 128, 256, 512, 1024, 2048)
MAX_BODY = 4 * 1024 * 1024


class Server(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse-DNS lookup in `server_bind`, which can stall startup."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def handle_systemone(engine, body: dict, think_default: str = "auto") -> dict:
    """The response for one parsed request body; raises RequestError on invalid input."""
    state, questions, think, include_thought = parse_systemone(body, think_default)
    return engine.system_one(state, questions, think=think, include_thought=include_thought)


def parse_systemone(body: dict, think_default: str = "auto") -> tuple:
    """(state, questions, think, include_thought) of a request body; raises RequestError on invalid input."""
    if not isinstance(body, dict) or "state" not in body or "questions" not in body:
        raise RequestError("Request needs state and questions")
    options = body.get("thinkjev") or {}
    if not isinstance(options, dict):
        raise RequestError("thinkjev must be an object")
    return body["state"], body["questions"], options.get("think", think_default), bool(options.get("include_thought", False))


def systemone_events(engine, body: dict, think_default: str = "auto"):
    """The events of one request as they happen, ending with `done` (the full response) or `error`.

    The engine runs in a worker thread and hands events over a queue, so the HTTP handler can write
    each one as soon as it exists.
    """
    state, questions, think, include_thought = parse_systemone(body, think_default)
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            response = engine.system_one(state, questions, think, include_thought, on_event=events.put)
            events.put({"type": "done", "response": response})
        except RequestError as error:
            events.put({"type": "error", "detail": str(error)})
        except Exception as error:  # noqa: BLE001 - reported to the client, then re-raised in the log
            events.put({"type": "error", "detail": f"{type(error).__name__}: {error}"})

    threading.Thread(target=work, daemon=True).start()
    while True:
        event = events.get()
        yield event
        if event["type"] in ("done", "error"):
            return


def make_handler(engine, api_key: str | None = None, think_default: str = "auto"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "thinkjev"

        def _send(self, status: int, payload, content_type: str = "application/json") -> None:
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - http.server naming
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, DEMO_PAGE, "text/html; charset=utf-8")
            elif self.path.rstrip("/") == "/v1/thinkjev/info":
                self._send(HTTPStatus.OK, engine.info())
            else:
                self._send(HTTPStatus.NOT_FOUND, {"detail": "Not found"})

        def _stream(self, body: dict) -> None:
            """Server-sent events: one `data:` line per event, flushed as it happens."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            for event in systemone_events(engine, body, think_default):
                self.wfile.write(f"data: {json.dumps(event, allow_nan=False)}\n\n".encode())
                self.wfile.flush()

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/systemone":
                return self._send(HTTPStatus.NOT_FOUND, {"detail": "Not found"})
            if api_key and self.headers.get("Authorization") != f"Bearer {api_key}":
                return self._send(HTTPStatus.UNAUTHORIZED, {"detail": "Missing or invalid API key"})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"detail": "Request too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"null")
                options = body.get("thinkjev") if isinstance(body, dict) else None
                if isinstance(options, dict) and options.get("stream"):
                    parse_systemone(body, think_default)  # reject bad requests with 422 before streaming
                    return self._stream(body)
                return self._send(HTTPStatus.OK, handle_systemone(engine, body, think_default))
            except (json.JSONDecodeError, RequestError) as error:
                return self._send(HTTPStatus.UNPROCESSABLE_ENTITY, {"detail": str(error)})

        def log_message(self, format, *args):  # quieter than the default per-request stderr line
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=presets(),
                        help="A release model with its bundled calibration (sets model, revision, calibration, "
                             "rotations, and think budget)")
    parser.add_argument("--model", help="Hugging Face model (with --revision; instead of --preset)")
    parser.add_argument("--revision", help="Exact model revision (commit hash)")
    parser.add_argument("--name", help="Model name reported in responses (default thinkjev-<model>)")
    parser.add_argument("--calibration", type=Path,
                        help="evaluate_cascade report: S1/S2 temperatures and the dev-selected gate threshold")
    parser.add_argument("--s1-temperature", type=float)
    parser.add_argument("--s2-temperature", type=float)
    parser.add_argument("--threshold", type=float, help="Think when calibrated S1 confidence is below this")
    parser.add_argument("--think-budget", type=int, help="Maximum thought tokens (default: the preset's, else 1024)")
    parser.add_argument("--checkpoints", default=None,
                        help="Thought lengths at which the answer slots are also read, e.g. 0,128,256,512 "
                             "(default: the calibration report's checkpoints, else 0,128,256,512,1024,2048 "
                             "below the budget; 'none' disables)")
    parser.add_argument("--rotations", action="store_true", help="Debias S1 over cyclic option shifts")
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", help="Require this bearer token (default: no auth, local use)")
    parser.add_argument("--think-default", choices=THINK_MODES, default="auto",
                        help="Thinking mode for requests that carry no thinkjev field")
    args = parser.parse_args()
    if args.preset:
        if args.model or args.revision or args.calibration:
            parser.error("--preset sets the model, revision, and calibration; do not pass them too")
        fixed = preset(args.preset)
        args.model, args.revision, args.calibration = fixed["model"], fixed["revision"], fixed["calibration"]
        args.rotations = args.rotations or fixed["rotations"]
        args.think_budget = args.think_budget or fixed["think_budget"]
    elif not (args.model and args.revision):
        parser.error("pass --preset, or --model and --revision")
    args.think_budget = args.think_budget or 1024
    explicit = {key: value for key, value in (("s1_temperature", args.s1_temperature),
                                               ("s2_temperature", args.s2_temperature),
                                               ("threshold", args.threshold)) if value is not None}
    if args.calibration is None and len(explicit) < 3:
        print("No --calibration report: unset temperatures default to 1.0 and the threshold to 0.9 "
              "(uncalibrated; fit them on labeled rows from your own workload)")
    if args.checkpoints is not None:
        explicit["checkpoints"] = () if args.checkpoints == "none" else tuple(parse_checkpoints(parser, args))
    elif args.calibration is None:
        explicit["checkpoints"] = tuple(t for t in DEFAULT_CHECKPOINTS if t < args.think_budget)
    engine = ThinkJev.load(args.model, args.revision, name=args.name, device=args.device, dtype=args.dtype,
                           calibration=args.calibration, think_budget=args.think_budget,
                           rotations=args.rotations, **explicit)
    if args.checkpoints is None and args.calibration is not None and not engine.checkpoints:
        print("The calibration report has no anytime readout; pass --checkpoints to read the thought at budgets")
    server = Server((args.host, args.port), make_handler(engine, args.api_key, args.think_default))
    print(f"{engine.name} on http://{args.host}:{args.port}  (POST /v1/systemone, demo at /)")
    server.serve_forever()




if __name__ == "__main__":
    main()
