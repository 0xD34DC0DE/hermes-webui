"""Hermes WebUI Runner Server (Phase 2 of graceful-restart plan).

The runner server owns AIAgent instances out-of-process from the WebUI.
The WebUI talks to it over HTTP via ``api.runner_client.HttpRunnerClient``;
this module implements the matching server side.

Why out-of-process:
  AIAgent is constructed and runs inside whichever Python process
  imports ``run_agent``. Today that is the WebUI's ``server.py``, so a
  WebUI restart kills the in-flight agent. This runner holds the
  AIAgent on its own process; the WebUI's runtime-adapter
  (``HERMES_WEBUI_RUNTIME_ADAPTER=runner-local``) proxies chat turns
  through here, and a WebUI restart becomes transparent to a running
  conversation because the agent is alive elsewhere.

Wire format:
  The HTTP contract is fully specified by ``api.runner_client.py``. Every
  endpoint path, payload field, and event shape is pinned down there; we
  do NOT add new fields in Phase 2. The event shape is:

    {"event_id": "<uuid>", "seq": <int>, "event": "<sse_name>", "data": {...}}

  and ``seq`` is a monotonically increasing integer cursor.

Process model:
  - One in-memory ``RunStore`` keyed by ``run_id``. Phase 2 keeps state
    in RAM; the runner restart trade-off is documented in
    ``.hermes/plans/2026-06-18_181200-phase2-runner-server-design.md``.
  - AIAgent runs on a daemon thread per run. Cancellation is cooperative
    via a ``threading.Event`` flag the agent polls inside its loop.
  - Approval / clarify waits use ``threading.Event`` waiters keyed by
    id, so the ``respond_*`` endpoints can unblock the agent from
    another HTTP request.

Stdlib only:
  Matches the WebUI's ``ThreadingHTTPServer`` shape so no new deps are
  needed. FastAPI was considered and rejected — adding a framework for
  one internal subprocess is the wrong trade.

Operators:
  The runner is started by ``start-runner.ps1`` and supervised by
  ``restart.ps1``. There is no inbound auth on the runner HTTP socket;
  bind it to loopback (``127.0.0.1``) and let the WebUI talk to it
  locally. ``runner.pid`` mirrors ``server.pid`` for restart tooling.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Deque, Optional


# ── Run state ───────────────────────────────────────────────────────────────


class Run:
    """One agent run, in-memory.

    Holds the AIAgent instance, its event queue, and the control-plane
    primitives (cancel flag, approval/clarify waiters, pending-message
    queue). Tests poke at these attributes directly to drive the agent
    through its lifecycle without a live LLM provider.
    """

    def __init__(self, *, run_id: str, session_id: str, request: dict,
                 agent_factory: Callable[..., Any]):
        self.run_id = run_id
        self.session_id = session_id
        self.request = dict(request)
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None
        self.terminal_state: Optional[str] = None
        self._seq = 0
        self._seq_lock = threading.Lock()
        self.events: Deque[dict] = deque(maxlen=10_000)
        self._events_lock = threading.Lock()

        # Control-plane waiters. Keyed by id so multiple clarifications
        # can be in flight at once (rare but possible during parallel
        # tool calls).
        self.clarify_waiters: dict[str, threading.Event] = {}
        self.clarify_responses: dict[str, str] = {}
        self.approval_waiters: dict[str, threading.Event] = {}
        self.approval_responses: dict[str, str] = {}

        # Pending messages queued via POST /messages.
        self.pending_messages: list[str] = []
        self._messages_lock = threading.Lock()

        # Cancellation is cooperative.
        self.cancel_event = threading.Event()
        self._agent_thread: Optional[threading.Thread] = None
        self._agent: Optional[Any] = None
        self._agent_factory = agent_factory
        self._agent_lock = threading.Lock()

    # ── events ─────────────────────────────────────────────────────────────

    def next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def enqueue_event(self, event: dict) -> dict:
        """Append an event, stamping event_id and seq if not present.

        Returns the (possibly mutated) event. Safe to call from any
        thread; held briefly under ``_events_lock`` so a concurrent
        ``events`` snapshot from the HTTP layer doesn't tear.
        """
        with self._events_lock:
            if "event_id" not in event:
                event["event_id"] = uuid.uuid4().hex
            if "seq" not in event:
                event["seq"] = self.next_seq()
            self.events.append(event)
            return event

    def snapshot_events(self, *, after_seq: int) -> list[dict]:
        with self._events_lock:
            return [dict(e) for e in self.events if e.get("seq", 0) > after_seq]

    # ── control plane ──────────────────────────────────────────────────────

    def wait_for_clarify(self, clarify_id: str, timeout: float = 300.0) -> Optional[str]:
        ev = threading.Event()
        self.clarify_waiters[clarify_id] = ev
        try:
            if not ev.wait(timeout=timeout):
                return None
            return self.clarify_responses.get(clarify_id)
        finally:
            self.clarify_waiters.pop(clarify_id, None)

    def respond_clarify(self, clarify_id: str, response: str) -> bool:
        self.clarify_responses[clarify_id] = response
        ev = self.clarify_waiters.pop(clarify_id, None)
        if ev is not None:
            ev.set()
            return True
        # No live waiter — response is still recorded so a waiter that
        # appears later (or a status poll) can see it. This makes the
        # respond endpoint a true idempotent setter rather than
        # requiring a paired wait_for_clarify.
        return True

    def wait_for_approval(self, approval_id: str, timeout: float = 300.0) -> Optional[str]:
        ev = threading.Event()
        self.approval_waiters[approval_id] = ev
        try:
            if not ev.wait(timeout=timeout):
                return None
            return self.approval_responses.get(approval_id)
        finally:
            self.approval_waiters.pop(approval_id, None)

    def respond_approval(self, approval_id: str, choice: str) -> bool:
        self.approval_responses[approval_id] = choice
        ev = self.approval_waiters.pop(approval_id, None)
        if ev is not None:
            ev.set()
            return True
        return True

    def queue_message(self, message: str) -> None:
        with self._messages_lock:
            self.pending_messages.append(message)

    def drain_messages(self) -> list[str]:
        with self._messages_lock:
            out = list(self.pending_messages)
            self.pending_messages.clear()
            return out

    # ── agent thread ───────────────────────────────────────────────────────

    def start_agent(self) -> None:
        """Construct the AIAgent and start the conversation thread.

        The agent factory is injected so tests can substitute a fake.
        Real deployment constructs ``from run_agent import AIAgent``.
        """
        with self._agent_lock:
            if self._agent_thread is not None:
                return
            self.started_at = time.time()
            # The agent factory is responsible for passing the callbacks
            # that route through ``enqueue_event`` /
            # ``wait_for_clarify`` / etc. We hand the factory the run
            # itself so it can wire them up however it likes.
            self._agent = self._agent_factory(run=self, request=self.request)
            self._agent_thread = threading.Thread(
                target=self._run_agent_safely,
                name=f"runner-agent-{self.run_id}",
                daemon=True,
            )
            self._agent_thread.start()

    def _run_agent_safely(self) -> None:
        agent = self._agent
        try:
            if hasattr(agent, "run_conversation"):
                agent.run_conversation(user_message=self.request.get("message", ""))
            self.terminal_state = "completed"
        except Exception as exc:  # pragma: no cover - defensive
            self.enqueue_event({"event": "error", "data": {"message": str(exc)}})
            self.terminal_state = "error"
        finally:
            self.ended_at = time.time()
            self.enqueue_event({
                "event": "stream_end",
                "data": {"terminal_state": self.terminal_state},
            })

    def cancel(self) -> None:
        self.cancel_event.set()
        agent = self._agent
        if agent is not None and hasattr(agent, "cancel"):
            try:
                agent.cancel()
            except Exception:
                pass
        # Even if the agent doesn't respect cancellation, mark the run
        # terminal so get_run() reports it.
        if self.terminal_state is None:
            self.terminal_state = "cancelled"
            self.ended_at = time.time()
            self.enqueue_event({
                "event": "cancel",
                "data": {"reason": "user_requested"},
            })


# ── Store ───────────────────────────────────────────────────────────────────


class RunStore:
    """In-memory registry of active runs.

    Single-process, single-threaded for state mutations (the HTTP handler
    thread); agent threads append events via ``Run.enqueue_event``
    directly, which holds its own lock.
    """

    def __init__(self, *, agent_factory: Optional[Callable[..., Any]] = None):
        self._runs: dict[str, Run] = {}
        self._lock = threading.Lock()
        self._agent_factory: Callable[..., Any] = agent_factory or _default_agent_factory

    def create(self, request: dict) -> Run:
        run_id = uuid.uuid4().hex
        session_id = str(request.get("session_id") or "")
        run = Run(
            run_id=run_id,
            session_id=session_id,
            request=request,
            agent_factory=self._agent_factory,
        )
        with self._lock:
            self._runs[run_id] = run
        run.start_agent()
        return run

    def get(self, run_id: str) -> Optional[Run]:
        with self._lock:
            return self._runs.get(run_id)

    def all_runs(self) -> list[Run]:
        with self._lock:
            return list(self._runs.values())


# ── HTTP layer ──────────────────────────────────────────────────────────────


_VALID_GOAL_ACTIONS = {"set", "pause", "resume", "clear", "status", "edit"}


def _build_handler(store: RunStore, *, goal_handler: Optional[Callable] = None):
    """Construct the BaseHTTPRequestHandler subclass bound to ``store``.

    ``goal_handler`` is injected so tests can drive goal state without
    requiring hermes-cli on sys.path. Production injects
    ``_default_goal_handler``.
    """
    goal_handler = goal_handler or _default_goal_handler

    class _RunnerHandler(BaseHTTPRequestHandler):
        # Silence the default Apache-style log; the runner emits JSON
        # request lines on its own via log_request.
        def log_message(self, fmt, *args):
            return

        def log_request(self, code: str = "-", size: str = "-") -> None:
            try:
                duration_ms = round((time.time() - getattr(self, "_req_t0", time.time())) * 1000, 1)
            except Exception:
                duration_ms = -1.0
            try:
                print(
                    f'[runner] {{"method":"{self.command}","path":"{self.path}",'
                    f'"status":"{code}","ms":{duration_ms}}}',
                    flush=True,
                )
            except Exception:
                pass

        # ── dispatch ────────────────────────────────────────────────────

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "invalid JSON"})
                raise _StopProcessing()

        def _send(self, status: int, payload: Any) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self._req_t0 = time.time()
            try:
                parts = self.path.split("?", 1)
                path = parts[0]
                if path.startswith("/v1/runs/"):
                    run_id = path[len("/v1/runs/"):]
                    # /v1/runs/{id} or /v1/runs/{id}/events
                    sub = run_id.split("/", 1)
                    if len(sub) == 1:
                        return self._handle_get_run(sub[0])
                    if len(sub) == 2 and sub[1] == "events":
                        return self._handle_get_events(sub[0], parts[1] if len(parts) > 1 else "")
                return self._send(404, {"error": "not found"})
            except _StopProcessing:
                return
            except Exception:
                self._send(500, {"error": "internal"})

        def do_POST(self) -> None:
            self._req_t0 = time.time()
            try:
                if self.path == "/v1/runs":
                    return self._handle_post_runs()
                if self.path.startswith("/v1/runs/"):
                    return self._handle_post_subpath()
                if self.path.startswith("/v1/sessions/"):
                    return self._handle_post_goal()
                return self._send(404, {"error": "not found"})
            except _StopProcessing:
                return
            except Exception:
                self._send(500, {"error": "internal"})

        def do_PUT(self) -> None:  # noqa: N802
            self._send(405, {"error": "method not allowed"})

        def do_DELETE(self) -> None:  # noqa: N802
            self._send(405, {"error": "method not allowed"})

        # ── POST /v1/runs ───────────────────────────────────────────────

        def _handle_post_runs(self) -> None:
            body = self._read_json()
            session_id = str(body.get("session_id") or "").strip()
            if not session_id:
                return self._send(400, {"error": "session_id required"})
            run = store.create(body)
            return self._send(200, {
                "run_id": run.run_id,
                "stream_id": run.run_id,
                "session_id": run.session_id,
                "status": "started",
                "started_at": run.started_at,
                "cursor": "0",
                "active_controls": ["cancel"],
            })

        # ── POST /v1/runs/{id}/... ──────────────────────────────────────

        def _handle_post_subpath(self) -> None:
            body = self._read_json()
            rest = self.path[len("/v1/runs/"):]
            parts = rest.split("/")
            if len(parts) < 2:
                return self._send(404, {"error": "not found"})
            run_id, action = parts[0], parts[1]
            run = store.get(run_id)
            if run is None:
                return self._send(404, {"error": "run not found"})
            if action == "cancel":
                run.cancel()
                return self._send(200, {"ok": True, "status": "cancelling"})
            if action == "messages" and len(parts) == 2:
                msg = str(body.get("message") or "")
                if not msg:
                    return self._send(400, {"error": "message required"})
                run.queue_message(msg)
                return self._send(200, {"ok": True, "status": "queued", "mode": body.get("mode", "queue")})
            if action == "approvals" and len(parts) == 4 and parts[3] == "respond":
                choice = str(body.get("choice") or "")
                if not choice:
                    return self._send(400, {"error": "choice required"})
                ok = run.respond_approval(parts[2], choice)
                return self._send(200 if ok else 404, {
                    "ok": ok,
                    "status": "accepted" if ok else "no_pending_approval",
                })
            if action == "clarifications" and len(parts) == 4 and parts[3] == "respond":
                resp = str(body.get("response") or "")
                if not resp:
                    return self._send(400, {"error": "response required"})
                ok = run.respond_clarify(parts[2], resp)
                return self._send(200 if ok else 404, {
                    "ok": ok,
                    "status": "accepted" if ok else "no_pending_clarify",
                })
            return self._send(404, {"error": "not found"})

        # ── GET /v1/runs/{id} ───────────────────────────────────────────

        def _handle_get_run(self, run_id: str) -> None:
            run = store.get(run_id)
            if run is None:
                return self._send(404, {"error": "run not found"})
            active: list[str] = []
            if run.terminal_state is None:
                active.append("cancel")
                if run.clarify_waiters:
                    active.append("clarify")
                if run.approval_waiters:
                    active.append("approval")
            last_event_id = run.events[-1].get("event_id") if run.events else None
            pending_clarify = next(iter(run.clarify_waiters), None)
            pending_approval = next(iter(run.approval_waiters), None)
            status = "running" if run.terminal_state is None else run.terminal_state
            return self._send(200, {
                "run_id": run.run_id,
                "session_id": run.session_id,
                "status": status,
                "last_event_id": last_event_id,
                "terminal_state": run.terminal_state,
                "active_controls": active,
                "pending_clarify_id": pending_clarify,
                "pending_approval_id": pending_approval,
            })

        # ── GET /v1/runs/{id}/events ────────────────────────────────────

        def _handle_get_events(self, run_id: str, query: str) -> None:
            run = store.get(run_id)
            if run is None:
                return self._send(404, {"error": "run not found"})
            after_seq = 0
            if query:
                for kv in query.split("&"):
                    if kv.startswith("cursor="):
                        try:
                            after_seq = int(kv.split("=", 1)[1])
                        except ValueError:
                            pass
            events = run.snapshot_events(after_seq=after_seq)
            last_event_id = events[-1].get("event_id") if events else (
                run.events[-1].get("event_id") if run.events else None
            )
            return self._send(200, {
                "run_id": run.run_id,
                "events": events,
                "cursor": str(events[-1].get("seq", after_seq)) if events else str(after_seq),
                "last_event_id": last_event_id,
            })

        # ── POST /v1/sessions/{id}/goal ─────────────────────────────────

        def _handle_post_goal(self) -> None:
            body = self._read_json()
            session_id = self.path[len("/v1/sessions/"):].rstrip("/").split("/", 1)[0]
            action = str(body.get("action") or "").strip().lower()
            if action not in _VALID_GOAL_ACTIONS:
                return self._send(400, {
                    "error": f"action must be one of {sorted(_VALID_GOAL_ACTIONS)}",
                    "got": action,
                })
            text = str(body.get("text") or "")
            try:
                result = goal_handler(session_id=session_id, action=action, text=text)
            except Exception as exc:
                return self._send(500, {"error": f"goal handler failed: {exc}"})
            return self._send(200, result)

    return _RunnerHandler


class _StopProcessing(Exception):
    """Raised by ``_read_json`` to abort the handler after sending 400."""


# ── Default agent factory + goal handler ───────────────────────────────────


def _default_agent_factory(*, run: Run, request: dict) -> Any:
    """Construct a real AIAgent. Lazy import so tests without hermes-cli
    on sys.path don't fail at import time.
    """
    from run_agent import AIAgent  # type: ignore
    return AIAgent(
        session_id=run.session_id,
        event_callback=lambda event_name, data: run.enqueue_event({
            "event": event_name,
            "data": data,
        }),
        stream_delta_callback=lambda text: run.enqueue_event({
            "event": "delta",
            "data": {"text": text},
        }),
        tool_progress_callback=lambda name, **kw: run.enqueue_event({
            "event": "tool_progress",
            "data": {"name": name, **kw},
        }),
        clarify_callback=lambda prompt, clarify_id="clarify-default": (
            run.wait_for_clarify(clarify_id) or ""
        ),
    )


def _default_goal_handler(*, session_id: str, action: str, text: str) -> dict:
    """Proxy to hermes_cli.goals over a session-scoped DB.

    Lazy imports so tests can monkeypatch the goal module before the
    first request, and so the runner starts even if hermes_cli is
    partially importable. If hermes_cli's SessionDB isn't on sys.path,
    we fall back to a simple JSON file under $HERMES_HOME/runner/goals/
    so the endpoint still works in isolated test environments.
    """
    try:
        from hermes_cli.goals import load_goal, save_goal, clear_goal, GoalState  # type: ignore
    except Exception:
        return _fallback_goal_handler(session_id=session_id, action=action, text=text)

    if action == "set":
        state = GoalState(goal=text, status="active")
        save_goal(session_id, state)
        return {"ok": True, "action": action, "text": text}
    if action == "edit":
        existing = load_goal(session_id)
        if existing is None:
            existing = GoalState(goal=text, status="active")
        else:
            existing.goal = text
        save_goal(session_id, existing)
        return {"ok": True, "action": action, "text": text}
    if action == "pause":
        existing = load_goal(session_id) or GoalState(goal="", status="paused")
        existing.status = "paused"
        save_goal(session_id, existing)
        return {"ok": True, "action": action}
    if action == "resume":
        existing = load_goal(session_id) or GoalState(goal="", status="active")
        existing.status = "active"
        save_goal(session_id, existing)
        return {"ok": True, "action": action}
    if action == "clear":
        clear_goal(session_id)
        return {"ok": True, "action": action}
    # status
    existing = load_goal(session_id)
    return {
        "ok": True,
        "action": "status",
        "text": existing.goal if existing else "",
        "status": existing.status if existing else "",
    }


def _fallback_goal_handler(*, session_id: str, action: str, text: str) -> dict:
    """Minimal JSON-file-backed goal store for environments without hermes_cli.

    Tests that monkeypatch HERMES_HOME and don't have hermes-cli on
    sys.path use this transparently. Production hits the real
    hermes_cli.goals path; this is a defensive fallback only.
    """
    import pathlib
    home = pathlib.Path(os.environ.get("HERMES_HOME") or (pathlib.Path.home() / ".hermes"))
    goals_dir = home / "runner" / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)
    path = goals_dir / f"{session_id}.json"
    state: dict = {}
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    if action == "set":
        state = {"text": text, "status": "active"}
        path.write_text(json.dumps(state), encoding="utf-8")
        return {"ok": True, "action": action, "text": text}
    if action == "edit":
        state["text"] = text
        path.write_text(json.dumps(state), encoding="utf-8")
        return {"ok": True, "action": action, "text": text}
    if action == "pause":
        state["status"] = "paused"
        path.write_text(json.dumps(state), encoding="utf-8")
        return {"ok": True, "action": action}
    if action == "resume":
        state["status"] = "active"
        path.write_text(json.dumps(state), encoding="utf-8")
        return {"ok": True, "action": action}
    if action == "clear":
        if path.exists():
            path.unlink()
        return {"ok": True, "action": action}
    return {
        "ok": True,
        "action": "status",
        "text": state.get("text", ""),
        "status": state.get("status", ""),
    }


# ── Public API ─────────────────────────────────────────────────────────────


def make_server(*, host: str = "127.0.0.1", port: int = 8788,
                agent_factory: Optional[Callable] = None,
                goal_handler: Optional[Callable] = None) -> tuple[ThreadingHTTPServer, RunStore]:
    """Build a configured (server, store) pair. Does not call serve_forever.

    Returns the bound server (caller invokes ``serve_forever()``) and the
    underlying RunStore for tests that want to poke at state directly.
    """
    store = RunStore(agent_factory=agent_factory)
    handler_cls = _build_handler(store, goal_handler=goal_handler)

    server = ThreadingHTTPServer((host, port), handler_cls)
    server.daemon_threads = True
    return server, store


def start_server_in_process(*, host: str = "127.0.0.1", port: int = 8788,
                             agent_factory: Optional[Callable] = None,
                             goal_handler: Optional[Callable] = None) -> tuple[ThreadingHTTPServer, str, RunStore]:
    """Start the runner server in a background thread; return (server, base_url, store).

    Used by tests; production should call ``make_server`` then
    ``serve_forever()`` in its own main thread.
    """
    server, store = make_server(host=host, port=port, agent_factory=agent_factory, goal_handler=goal_handler)
    t = threading.Thread(target=server.serve_forever, name="runner-http", daemon=True)
    t.start()
    actual_port = server.server_address[1]
    base_url = f"http://{host}:{actual_port}"
    return server, base_url, store


def shutdown_in_process_server(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()


def get_run(run_id: str) -> Optional[Run]:
    """Module-level helper used by tests to fetch a Run from any store.

    Tests construct their own server fixture but poke the run directly.
    Rather than thread the store through, we expose a process-global
    last-created-store handle for tests; production code does not use
    this.
    """
    return _LAST_STORE.get(run_id) if _LAST_STORE is not None else None


_LAST_STORE: Optional[RunStore] = None


def _register_test_store(store: RunStore) -> None:
    """Tests call this from a fixture to expose ``get_run`` globally."""
    global _LAST_STORE
    _LAST_STORE = store