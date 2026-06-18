"""End-to-end integration test for out-of-process runner restart.

Phase 2 of graceful-restart. The core invariant: when the WebUI
restarts, in-flight agent runs continue on the runner side, and a
fresh WebUI can re-attach via /v1/runs/{id}/events without losing
events.

This test exercises the runner's HTTP contract directly (the
runner_client.py / runtime_adapter.py wiring is covered by the
WebUI's existing test_runtime_adapter_seam.py). The "WebUI restart"
is simulated by tearing down the client half and bringing up a fresh
one while the runner keeps running and emitting events.

Coverage:
  - Run survives a "WebUI" restart: events emitted while no client
    is attached are still observable when a new client reconnects
    with the right cursor.
  - Cancel issued from a fresh client cancels the run on the runner.
  - Goal state persists across "WebUI" restarts because the runner
    owns it.
"""
from __future__ import annotations

import json
import threading
import time
from http.client import HTTPConnection

import pytest


class _StreamingFakeAgent:
    """Emits a delta every 50ms until cancelled.

    The agent factory receives the Run object via kwargs (``run=self``)
    so this fake can enqueue events directly without needing the runner
    to inject callbacks. This mirrors the production wiring where the
    AIAgent's event_callback routes back to ``run.enqueue_event``.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.cancelled = False
        self._run = kwargs.get("run")

    def run_conversation(self, *, user_message, **kwargs):
        for i in range(100):
            if self.cancelled:
                break
            if self._run is not None:
                self._run.enqueue_event({"event": "delta", "data": {"text": f"chunk-{i}"}})
            time.sleep(0.05)

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def runner():
    """Start the runner on an ephemeral port with a streaming fake agent."""
    from api import runner_server as rs

    srv, base, store = rs.start_server_in_process(
        agent_factory=_StreamingFakeAgent,
        host="127.0.0.1", port=0,
    )
    rs._register_test_store(store)
    try:
        yield srv, base
    finally:
        rs.shutdown_in_process_server(srv)


def _conn(base_url: str) -> HTTPConnection:
    rest = base_url.split("://", 1)[1]
    host, _, port = rest.partition(":")
    return HTTPConnection(host, int(port), timeout=10)


def _post(base_url: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    c = _conn(base_url)
    c.request("POST", path, body=json.dumps(body or {}),
              headers={"Content-Type": "application/json"})
    resp = c.getresponse()
    raw = resp.read().decode("utf-8")
    c.close()
    return resp.status, json.loads(raw) if raw else {}


def _get(base_url: str, path: str) -> tuple[int, dict]:
    c = _conn(base_url)
    c.request("GET", path, headers={"Accept": "application/json"})
    resp = c.getresponse()
    raw = resp.read().decode("utf-8")
    c.close()
    return resp.status, json.loads(raw) if raw else {}


# ── core invariant: events survive a "WebUI" restart ─────────────────────


def test_events_survive_client_disconnect(runner):
    """Mid-run, drop the client. Bring up a fresh client. Verify it can
    reattach and read events that arrived while it was down.
    """
    _, base = runner

    # Phase 1: WebUI client A starts the run.
    status, payload = _post(base, "/v1/runs", {"session_id": "sess-A", "message": "hi"})
    assert status == 200
    run_id = payload["run_id"]

    # Drain a few events as client A.
    time.sleep(0.2)
    status, ev = _get(base, f"/v1/runs/{run_id}/events")
    assert status == 200
    cursor_a = ev["cursor"]
    assert int(cursor_a) > 0, f"expected some events to have arrived, got cursor={cursor_a}"

    # Phase 2: client A "restarts" — we drop the connection. The runner
    # keeps running and emitting events.
    time.sleep(0.3)
    status, ev_more = _get(base, f"/v1/runs/{run_id}/events?cursor={cursor_a}")
    assert status == 200
    events_during_disconnect = ev_more["events"]
    cursor_b = ev_more["cursor"]

    # Phase 3: fresh client B reconnects with the latest cursor and
    # sees no gap.
    assert int(cursor_b) > int(cursor_a), "cursor should advance while no client is attached"
    # All events emitted during disconnect are present.
    assert all(e["seq"] > int(cursor_a) for e in events_during_disconnect)


def test_client_can_re_attach_with_correct_cursor(runner):
    """Same scenario as above but verifies a fresh client can resume."""
    _, base = runner

    _, payload = _post(base, "/v1/runs", {"session_id": "sess-B", "message": "hi"})
    run_id = payload["run_id"]

    time.sleep(0.15)
    _, ev = _get(base, f"/v1/runs/{run_id}/events")
    cursor = ev["cursor"]

    # Simulate restart: bring up a "fresh" client and fetch with cursor.
    time.sleep(0.2)
    _, ev2 = _get(base, f"/v1/runs/{run_id}/events?cursor={cursor}")
    new_events = ev2["events"]

    # The fresh client sees only events after its cursor.
    assert all(int(e["seq"]) > int(cursor) for e in new_events)
    # The cursor advanced.
    assert int(ev2["cursor"]) > int(cursor)


# ── cancel from fresh client ──────────────────────────────────────────────


def test_cancel_from_fresh_client_stops_the_run(runner):
    """A fresh WebUI client can cancel an existing run started by another."""
    from api import runner_server as rs

    _, base = runner

    _, payload = _post(base, "/v1/runs", {"session_id": "sess-C", "message": "hi"})
    run_id = payload["run_id"]

    # Fresh client (simulating post-restart) issues cancel.
    status, body = _post(base, f"/v1/runs/{run_id}/cancel", {})
    assert status == 200
    assert body.get("ok") is True

    # The run's terminal_state should reflect cancellation.
    run = rs.get_run(run_id)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and run.terminal_state is None:
        time.sleep(0.05)
    assert run.terminal_state in ("cancelled", "completed")

    # And the events stream contains a terminal event.
    has_terminal = any(
        e.get("event") in ("cancel", "stream_end")
        for e in run.events
    )
    assert has_terminal


# ── goal state survives "WebUI" restart because the runner owns it ───────


def test_goal_state_survives_client_reconnect(runner, tmp_path, monkeypatch):
    """Goal state is owned by the runner, not the WebUI client. Set it,
    drop the client, bring up a fresh one, the goal is still there.
    """
    from api import runner_server as rs

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Restart the runner with the JSON fallback so this test doesn't
    # depend on hermes_cli being importable.
    rs.shutdown_in_process_server(runner[0])
    srv, base, store = rs.start_server_in_process(
        agent_factory=_StreamingFakeAgent,
        goal_handler=rs._fallback_goal_handler,
        host="127.0.0.1", port=0,
    )
    rs._register_test_store(store)
    try:
        # Phase 1: client A sets a goal.
        status, body = _post(base, "/v1/sessions/sess-D/goal",
                             {"action": "set", "text": "persist me"})
        assert status == 200
        assert body.get("text") == "persist me"

        # Phase 2: client A "restarts". A brand-new client (which is
        # what we already have here, since HTTPConnection is stateless)
        # reads the goal back.
        status, body = _post(base, "/v1/sessions/sess-D/goal",
                             {"action": "status", "text": ""})
        assert status == 200
        assert body.get("text") == "persist me"
    finally:
        rs.shutdown_in_process_server(srv)