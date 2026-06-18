"""Coverage for the composer-footer TPS indicator.

Product decision (mod `tps-footer`):
- The TPS chip should ALSO appear in the composer footer (right side, next to the
  context indicator) so users can see live throughput without scrolling back to
  the message header.
- Behavior mirrors the header chip: only renders when show_tps is enabled AND a
  real TPS value is available; otherwise the chip stays hidden (no placeholder).
- The existing header chip is untouched — this is additive.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INDEX_HTML = (REPO / "static" / "index.html").read_text(encoding="utf-8")
UI_JS = (REPO / "static" / "ui.js").read_text(encoding="utf-8")
CSS = (REPO / "static" / "style.css").read_text(encoding="utf-8")


def _slice(text, start_marker, end_marker):
    start = text.find(start_marker)
    assert start != -1, f"missing start marker: {start_marker!r}"
    end = text.find(end_marker, start)
    if end == -1:
        end = len(text)
    return text[start:end]


def test_footer_has_dedicated_tps_chip_element():
    # The element must exist as a discrete span inside .composer-right so it
    # can be toggled without disturbing the ctx indicator or send button.
    assert 'id="composerTpsChip"' in INDEX_HTML, (
        "composer footer should expose a dedicated TPS chip element"
    )
    # The chip must live inside .composer-right (right side of the footer)
    footer_slice = _slice(INDEX_HTML, 'class="composer-right"', "</div>")
    assert 'id="composerTpsChip"' in footer_slice, (
        "TPS chip should be inside .composer-right (footer right side)"
    )


def test_footer_tps_chip_respects_display_setting_and_value():
    # _setLiveAssistantTps should now update both the header chip AND the
    # footer chip. We test the function in isolation by inspecting its body.
    body = _slice(UI_JS, "function _setLiveAssistantTps(value)", "\n}\n")
    assert "composerTpsChip" in body, (
        "_setLiveAssistantTps should also target the footer TPS chip"
    )
    assert "isTpsDisplayEnabled" in body, (
        "footer TPS chip must respect the show_tps setting (off ⇒ hidden)"
    )
    # The footer update must suppress the chip when no value is given
    # (mirroring the header chip's null-clear behavior).
    assert "null" in body and ("style.display" in body or "hidden" in body), (
        "footer chip should hide itself when value is null/empty (no placeholder)"
    )


def test_footer_tps_chip_uses_known_style_hook():
    # The footer chip should reuse the same visual language as the header chip
    # (tabular numerals, bordered pill) — defined as a dedicated CSS class so
    # the header chip stays untouched.
    assert ".composer-tps-chip" in CSS, (
        "footer TPS chip needs an explicit CSS class hook"
    )
    assert "font-variant-numeric: tabular-nums" in CSS, (
        "footer TPS chip should keep tabular numerals for stable widths"
    )


def test_header_chip_is_unchanged():
    # Sanity: the existing per-message header chip must still exist and use
    # the same hook — this mod is purely additive.
    assert "msg-tps-inline" in UI_JS
    assert "msg-tps-inline" in CSS
    assert "_assistantRoleHtml(tsTitle='', tpsText='')" in UI_JS
