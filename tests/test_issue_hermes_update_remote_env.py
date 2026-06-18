"""Tests for HERMES_WEBUI_UPDATE_REMOTE env var override.

The WebUI's self-updater (api/updates.py) historically hardcoded the
remote name 'origin' in 12 places. To support private-fork installs
without losing the upstream `git pull --ff-only` path, the updater now
reads the remote name from the HERMES_WEBUI_UPDATE_REMOTE env var and
falls back to 'origin' when unset or empty. These tests pin that
contract.
"""
import os
from unittest.mock import patch

import api.updates as updates


def test_default_remote_is_origin_when_env_unset():
    """Without HERMES_WEBUI_UPDATE_REMOTE, the helper returns 'origin'."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop('HERMES_WEBUI_UPDATE_REMOTE', None)
        assert updates._update_remote() == 'origin'


def test_env_override_takes_effect():
    """A non-empty HERMES_WEBUI_UPDATE_REMOTE value is returned verbatim."""
    with patch.dict(os.environ, {'HERMES_WEBUI_UPDATE_REMOTE': 'fork'}):
        assert updates._update_remote() == 'fork'


def test_empty_env_value_falls_back_to_origin():
    """An empty HERMES_WEBUI_UPDATE_REMOTE is treated as unset (defensive)."""
    with patch.dict(os.environ, {'HERMES_WEBUI_UPDATE_REMOTE': ''}):
        assert updates._update_remote() == 'origin'


def test_whitespace_env_value_falls_back_to_origin():
    """A whitespace-only HERMES_WEBUI_UPDATE_REMOTE is treated as unset."""
    with patch.dict(os.environ, {'HERMES_WEBUI_UPDATE_REMOTE': '   '}):
        assert updates._update_remote() == 'origin'


def test_helper_is_called_on_every_git_invocation(tmp_path):
    """The updater's git operations go through _update_remote(), not the literal 'origin'.

    We patch _run_git to record invocations and assert the constructed
    argv never contains a hardcoded 'origin' string — only the value the
    helper returns. With HERMES_WEBUI_UPDATE_REMOTE=fork, every command
    uses 'fork' instead.
    """
    (tmp_path / '.git').mkdir()

    invocations = []

    def fake_git(args, cwd, timeout=10):
        invocations.append(args)
        # Return enough to keep _check_repo from erroring
        if args[0] == 'symbolic-ref':
            return '', False
        if args[0] == 'rev-parse' and '--verify' in args:
            return '', False
        if args[0] == 'diff-index':
            return '', True  # clean tree
        if args[0] == 'fetch':
            return 'network unreachable', False
        if args[0] == 'tag':
            return '', True
        return '', False

    with patch.dict(os.environ, {'HERMES_WEBUI_UPDATE_REMOTE': 'fork'}):
        with patch.object(updates, '_run_git', side_effect=fake_git):
            updates._check_repo(tmp_path, 'webui')

    # Every git invocation that names a remote should name 'fork', never 'origin'.
    for args in invocations:
        if 'origin' in args:
            raise AssertionError(
                f"git invocation still hardcodes 'origin': {args!r} — "
                "all remote references must go through _update_remote()"
            )
