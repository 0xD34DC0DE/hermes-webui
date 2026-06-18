"""Runner server contract tests.

Phase 2 of the graceful-server-restart plan: the runner server is the
out-of-process owner of the AIAgent. These tests pin down its HTTP
contract end-to-end against an in-process ThreadingHTTPServer, with
AIAgent mocked so we don't depend on a live LLM provider.

Coverage:
  - POST /v1/runs creates a run, returns run_id/stream_id/session_id.
  - GET /v1/runs/{id}/events serves queued events in seq order with cursor.
  - GET /v1/runs/{id} returns the run's current status.
  - POST /v1/runs/{id}/cancel flips the cooperative cancel flag and
    emits a terminal cancel event.
  - POST /v1/runs/{id}/approvals/{aid}/respond unblocks a waiting
    approval waiter with the chosen option.
  - POST /v1/runs/{id}/clarifications/{cid}/respond unblocks a waiting
    clarify waiter with the response string.
  - POST /v1/runs/{id}/messages queues a pending message into the
    run's message queue.
  - POST /v1/sessions/{sid}/goal proxies to hermes_cli.goals with the
    validated action.
  - The server validates Content-Type, rejects malformed JSON with 400,
    and 404s unknown runs.

The tests use the in-process ThreadingHTTPServer via ``api.runner_server``
directly (no subprocess) so they're hermetic and fast.
"""
from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection
from typing import Any

import pytest


# ── helpers ────────────────────────────────────────────────────────────────


class _FakeAgent:
    """Drop-in AIAgent replacement for tests.

    The runner_server accepts any object with the same callback surface
    as AIAgent (stream_delta_callback, tool_progress_callback,
    clarify_callback, event_callback, etc.). _FakeAgent records calls and
    lets the test drive lifecycle events synchronously.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cancelled = False
        self.clarify_waiters: dict[str, threading.Event] = {}
        self.clarify_responses: dict[str, str] = {}
        self.approval_waiters: dict[str, threading.Event] = {}
        self.approval_responses: dict[str, str] = {}
        self.pending_messages: list[str] = []

    def run_conversation(self, *, user_message, **kwargs):
        # Block until cancelled so cancel can be tested mid-stream.
        while not self.cancelled:
            time.sleep(0.05)

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def server():
    """Start the runner server in-process on an ephemeral port."""
    from api import runner_server as rs

    # Patch the agent factory so _FakeAgent is used.
    def factory(**kwargs):
        return _FakeAgent(**kwargs)

    srv, base_url, store = rs.start_server_in_process(agent_factory=factory, host="127.0.0.1", port=0)
    rs._register_test_store(store)
    try:
        yield srv, base_url
    finally:
        rs.shutdown_in_process_server(srv)


def _conn(base_url: str) -> HTTPConnection:
    # base_url is "http://127.0.0.1:<port>"
    rest = base_url.split("://", 1)[1]
    host, _, port = rest.partition(":")
    return HTTPConnection(host, int(port), timeout=10)


def _post(base_url: str, path: str, body: dict | None = None, *, headers: dict | None = None) -> tuple[int, dict]:
    c = _conn(base_url)
    payload = json.dumps(body or {}).encode("utf-8")
    h = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        h.update(headers)
    c.request("POST", path, body=payload, headers=h)
    resp = c.getresponse()
    raw = resp.read().decode("utf-8")
    c.close()
    try:
        return resp.status, json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return resp.status, {"_raw": raw}


def _get(base_url: str, path: str) -> tuple[int, dict]:
    c = _conn(base_url)
    c.request("GET", path, headers={"Accept": "application/json"})
    resp = c.getresponse()
    raw = resp.read().decode("utf-8")
    c.close()
    try:
        return resp.status, json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return resp.status, {"_raw": raw}


# ── Task 1: POST /v1/runs ───────────────────────────────────────────────────


def test_post_runs_creates_run_and_returns_ids(server):
    _, base_url = server
    status, payload = _post(base_url, "/v1/runs", {
        "session_id": "sess-1",
        "message": "hello",
        "workspace": None,
        "profile": None,
        "provider": None,
        "model": None,
        "toolsets": [],
        "source": "webui",
        "metadata": {},
    })
    assert status == 200, payload
    assert payload["session_id"] == "sess-1"
    assert payload["status"] == "started"
    assert payload["run_id"]
    assert payload["stream_id"] == payload["run_id"]
    assert payload["cursor"] == "0"
    assert isinstance(payload["active_controls"], list)
    assert payload["started_at"]


def test_post_runs_rejects_non_json(server):
    _, base_url = server
    c = _conn(base_url)
    c.request("POST", "/v1/runs", body=b"not json", headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    assert resp.status == 400
    c.close()


def test_post_runs_requires_session_id(server):
    _, base_url = server
    status, payload = _post(base_url, "/v1/runs", {"message": "hi"})
    assert status == 400
    assert "session_id" in payload.get("error", "")


# ── Task 2: GET /v1/runs/{id}/events ───────────────────────────────────────


def test_get_events_serves_queued_events_in_seq_order(server):
    from api import runner_server as rs

    _, base_url = server
    status, payload = _post(base_url, "/v1/runs", {"session_id": "sess-1", "message": "hi"})
    run_id = payload["run_id"]

    # Synthetically push events into the run's queue.
    run = rs.get_run(run_id)
    run.enqueue_event({"event": "stream_start", "data": {"x": 1}})
    run.enqueue_event({"event": "delta", "data": {"text": "hi"}})
    run.enqueue_event({"event": "stream_end", "data": {"ok": True}})

    status, ev = _get(base_url, f"/v1/runs/{run_id}/events")
    assert status == 200
    assert len(ev["events"]) == 3
    assert [e["seq"] for e in ev["events"]] == [1, 2, 3]
    assert [e["event"] for e in ev["events"]] == ["stream_start", "delta", "stream_end"]
    assert ev["cursor"] == "3"


def test_get_events_cursor_pagination(server):
    from api import runner_server as rs

    _, base_url = server
    _, payload = _post(base_url, "/v1/runs", {"session_id": "sess-1", "message": "hi"})
    run_id = payload["run_id"]

    run = rs.get_run(run_id)
    for i in range(5):
        run.enqueue_event({"event": "delta", "data": {"text": f"chunk-{i}"}})

    # First fetch: full.
    _, ev1 = _get(base_url, f"/v1/runs/{run_id}/events")
    assert len(ev1["events"]) == 5
    assert ev1["cursor"] == "5"

    # Cursor=2 should return events with seq >= 3 (events 3, 4, 5).
    _, ev2 = _get(base_url, f"/v1/runs/{run_id}/events?cursor=2")
    assert [e["seq"] for e in ev2["events"]] == [3, 4, 5]
    assert ev2["cursor"] == "5"


def test_get_events_returns_empty_for_unknown_run(server):
    _, base_url = server
    status, payload = _get(base_url, "/v1/runs/nonexistent/events")
    assert status == 404


# ── Task 3: GET /v1/runs/{id} ──────────────────────────────────────────────


def test_get_run_returns_status(server):
    _, base_url = server
    _, payload = _post(base_url, "/v1/runs", {"session_id": "sess-1", "message": "hi"})
    run_id = payload["run_id"]

    status, body = _get(base_url, f"/v1/runs/{run_id}")
    assert status == 200
    assert body["run_id"] == run_id
    assert body["session_id"] == "sess-1"
    assert body["status"] in ("started", "running")


# ── Task 4: cancel + control-plane endpoints ───────────────────────────────


def test_cancel_run_sets_flag_and_emits_terminal_event(server):
    from api import runner_server as rs

    _, base_url = server
    _, payload = _post(base_url, "/v1/runs", {"session_id": "sess-1", "message": "hi"})
    run_id = payload["run_id"]

    run = rs.get_run(run_id)
    # The fake agent runs run_conversation in a thread; it should exit
    # shortly after we cancel.
    status, body = _post(base_url, f"/v1/runs/{run_id}/cancel", {})
    assert status == 200
    assert body.get("ok") is True

    # Wait briefly for the agent thread to wind down and the terminal
    # event to be queued.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if run.terminal_state is not None:
            break
        time.sleep(0.05)
    assert run.terminal_state in ("cancelled", "completed")
    assert any(e.get("event") == "cancel" for e in list(run.events))


def test_respond_clarify_unblocks_waiter(server):
    from api import runner_server as rs

    _, base_url = server
    _, payload = _post(base_url, "/v1/runs", {"session_id": "sess-1", "message": "hi"})
    run_id = payload["run_id"]
    run = rs.get_run(run_id)

    # Simulate a clarify callback that waits for a response.
    result: dict[str, Any] = {}

    def waiter():
        # Block until the run's clarifications dict has a response for
        # our id, with a 2s ceiling so the test doesn't hang.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            ev = run.clarify_responses.get("clar-1")
            if ev is not None:
                result["response"] = ev
                return
            time.sleep(0.02)
        result["response"] = None

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    # Now POST the response.
    status, body = _post(base_url, f"/v1/runs/{run_id}/clarifications/clar-1/respond", {"response": "yes"})
    assert status == 200
    assert body.get("ok") is True
    t.join(timeout=2.5)
    assert result.get("response") == "yes"


def test_respond_approval_unblocks_waiter(server):
    from api import runner_server as rs

    _, base_url = server
    _, payload = _post(base_url, "/v1/runs", {"session_id": "sess-1", "message": "hi"})
    run_id = payload["run_id"]
    run = rs.get_run(run_id)

    result: dict[str, Any] = {}

    def waiter():
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            ev = run.approval_responses.get("app-1")
            if ev is not None:
                result["choice"] = ev
                return
            time.sleep(0.02)
        result["choice"] = None

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    status, body = _post(base_url, f"/v1/runs/{run_id}/approvals/app-1/respond", {"choice": "approve"})
    assert status == 200
    assert body.get("ok") is True
    t.join(timeout=2.5)
    assert result.get("choice") == "approve"


def test_queue_message_appends_to_run_queue(server):
    from api import runner_server as rs

    _, base_url = server
    _, payload = _post(base_url, "/v1/runs", {"session_id": "sess-1", "message": "hi"})
    run_id = payload["run_id"]
    run = rs.get_run(run_id)

    status, body = _post(base_url, f"/v1/runs/{run_id}/messages", {"message": "follow-up", "mode": "queue"})
    assert status == 200
    assert body.get("ok") is True
    # The run's pending_messages list should contain the queued message.
    assert "follow-up" in run.pending_messages


# ── Task 5: update_goal proxy ─────────────────────────────────────────────


def test_update_goal_set_status_clear_round_trip(server, tmp_path, monkeypatch):
    """Goal round-trip via the JSON-file fallback (hermes-cli may not be importable)."""
    from api import runner_server as rs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Spin up a *second* server with the fallback goal handler so the
    # test doesn't depend on hermes_cli being importable in the test env.
    srv, base, _ = rs.start_server_in_process(
        agent_factory=lambda run, request: type("FA", (), {"run_conversation": lambda self, **kw: None, "cancel": lambda self: None})(),
        goal_handler=rs._fallback_goal_handler,
        host="127.0.0.1", port=0,
    )
    try:
        # set
        status, body = _post(base, "/v1/sessions/sess-1/goal", {"action": "set", "text": "ship it"})
        assert status == 200, body
        assert body.get("ok") is True
        assert body.get("action") == "set"

        # status
        status, body = _post(base, "/v1/sessions/sess-1/goal", {"action": "status", "text": ""})
        assert status == 200
        assert body.get("text") == "ship it"

        # clear
        status, body = _post(base, "/v1/sessions/sess-1/goal", {"action": "clear", "text": ""})
        assert status == 200
        assert body.get("ok") is True

        # status after clear
        status, body = _post(base, "/v1/sessions/sess-1/goal", {"action": "status", "text": ""})
        assert body.get("text", "") == ""
    finally:
        rs.shutdown_in_process_server(srv)


def test_update_goal_rejects_unknown_action(server):
    _, base_url = server
    status, body = _post(base_url, "/v1/sessions/sess-1/goal", {"action": "bogus", "text": ""})
    assert status == 400


# ── Cross-cutting concerns ─────────────────────────────────────────────────


def test_404_for_unknown_routes(server):
    _, base_url = server
    status, _ = _get(base_url, "/v1/nonsense")
    assert status == 404


def test_method_not_allowed(server):
    _, base_url = server
    c = _conn(base_url)
    c.request("DELETE", "/v1/runs")
    resp = c.getresponse()
    assert resp.status == 405
    c.close()