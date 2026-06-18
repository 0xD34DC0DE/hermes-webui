"""Server-lifecycle helpers: PID file + port-free polling + state machine.

Phase 1 of the "graceful server restart" plan. Phase 2 (runner-local) will
replace this with a runner-owned lifecycle; until then this module owns:

  - ``server.pid`` under HERMES_WEBUI_STATE_DIR (JSON, not raw int, so we can
    extend without a schema break).
  - A process-wide restart-state machine readable via ``/api/server/restart/status``.
  - A port-free polling helper used by ``restart.ps1`` on Windows / POSIX.

Design constraints (mirrors the existing ``api/gateway_watcher.py`` and
``api/background_process.py`` shape):

  - Pure stdlib. No new deps.
  - All public functions are thread-safe via a single module-level ``_LOCK``.
  - Every best-effort write/read is wrapped in ``try/except`` so a corrupt or
    stale pid file never blocks server startup or shutdown.
  - The state-machine writes use ``datetime.now(timezone.utc).isoformat()``
    exclusively — no naive datetimes anywhere in the lifecycle payload, so
    log-grepping across timezones is unambiguous.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.config import HOST, PORT, STATE_DIR
from api.updates import WEBUI_VERSION

_PID_FILE_NAME = "server.pid"
_LOCK = threading.Lock()

# Process-wide state machine. Tests monkeypatch this directly. Keys are stable
# so the /api/server/restart/status contract can be locked down:
#
#   state          - "idle" | "scheduled" | "shutting_down" | "relaunching" | "back"
#   restarts_at    - ISO-8601 timestamp the SIGINT is scheduled to fire, or None
#   old_pid        - PID of the server that scheduled the restart, or None
#   new_pid        - PID of the replacement server once it's bound, or None
#   port           - bind port that must remain reachable across the restart
#   last_error     - last error message surfaced from this module, or None
#   updated_at     - ISO-8601 timestamp of the most recent state mutation
_state: dict[str, Any] = {
    "state": "idle",
    "restarts_at": None,
    "old_pid": None,
    "new_pid": None,
    "port": None,
    "last_error": None,
    "updated_at": None,
}


# ── PID file ────────────────────────────────────────────────────────────────

def _pid_file_path() -> Path:
    return Path(STATE_DIR) / _PID_FILE_NAME


def write_pid_file() -> Path:
    """Atomically write ``server.pid`` after the TCP port has been bound.

    Uses ``os.replace`` so a concurrent reader never sees a half-written file.
    On a fresh install this also creates ``STATE_DIR`` if it does not exist.

    Returns the path that was written. Raises on permission / disk errors so
    ``server.py`` can log a clear warning rather than silently losing track
    of its own PID.
    """
    payload = {
        "pid": os.getpid(),
        "port": int(PORT),
        "host": HOST,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "version": WEBUI_VERSION,
    }
    p = _pid_file_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".pid.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, p)
    return p


def clear_pid_file() -> None:
    """Remove ``server.pid`` if it still points at this process.

    A stale ``server.pid`` (left behind by an earlier crash) is intentionally
    preserved so ``restart.ps1`` can still detect and stop a leaked server.
    We only delete the file when its ``pid`` matches ``os.getpid()``; if a
    concurrent process has already replaced the file with a different PID,
    we leave it alone — that other process owns the file now.
    """
    p = _pid_file_path()
    if not p.exists():
        return
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # Corrupt JSON: safe to remove since we don't know whose it is.
        p.unlink(missing_ok=True)
        return
    if int(payload.get("pid") or -1) == os.getpid():
        p.unlink(missing_ok=True)


def read_pid_file() -> dict | None:
    """Return the parsed ``server.pid`` payload, or ``None`` if missing/invalid."""
    p = _pid_file_path()
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Port polling ────────────────────────────────────────────────────────────

def is_port_free(host: str, port: int, *, timeout: float = 0.25) -> bool:
    """Return True if nothing is listening on (host, port).

    A successful TCP connect means *something* answered — that's the opposite
    of "free", so we return False. Connection refused / reset / timeout all
    mean "nothing listening" and we return True. This mirrors the existing
    ``_abort_if_already_serving`` probe in ``server.py``.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return False
    except (ConnectionRefusedError, ConnectionResetError, OSError, socket.timeout):
        return True


def wait_for_port_free(
    host: str,
    port: int,
    *,
    max_seconds: float = 15.0,
    interval: float = 0.25,
) -> bool:
    """Poll until (host, port) accepts no connection, or until the deadline.

    Returns True if the port was observed free at least once. Returns False on
    timeout so callers can decide whether to abort or proceed. The 15s
    default is sized for Windows ``WSAEADDRINUSE`` cleanup, which the
    existing ``QuietHTTPServer.server_bind`` retry loop (server.py:164)
    covers on the bind side; 15s leaves comfortable headroom.
    """
    deadline = time.monotonic() + max_seconds
    while time.monotonic() < deadline:
        if is_port_free(host, port, timeout=interval):
            return True
        time.sleep(interval)
    return False


# ── State machine ───────────────────────────────────────────────────────────

def get_state() -> dict[str, Any]:
    """Return a snapshot of the current restart-state machine.

    The returned dict is a shallow copy, so callers cannot mutate the
    canonical state. Always reads under ``_LOCK`` so a concurrent
    ``set_state`` mid-call cannot tear the snapshot.
    """
    with _LOCK:
        return dict(_state)


def set_state(**fields: Any) -> None:
    """Merge ``fields`` into the state machine and stamp ``updated_at``.

    Unknown keys are allowed so forward-compatibility is trivial; readers
    must tolerate missing keys. Stamps ``updated_at`` on every call so the
    frontend can render a "last update Xs ago" hint without a second clock
    source.
    """
    with _LOCK:
        _state.update(fields)
        _state["updated_at"] = datetime.now(timezone.utc).isoformat()


def reset_state() -> None:
    """Hard-reset the state machine back to ``idle``.

    Used by tests and by the /api/server/restart/status path after a
    successful relaunch, so a subsequent restart call starts clean.
    """
    with _LOCK:
        for key in list(_state.keys()):
            if key == "state":
                _state[key] = "idle"
            else:
                _state[key] = None
        _state["updated_at"] = datetime.now(timezone.utc).isoformat()