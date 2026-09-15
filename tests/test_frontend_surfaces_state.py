"""State the backend sends must actually be rendered.

The dashboard is an audit surface, so a reading the backend publishes and the frontend never
looks at is not a cosmetic omission -- it is a check that exists on paper only.

That is what happened to reconciliation. `snapshot.py` has included `state.reconciliation`
for as long as the checker has existed, and `renderHealth` read `account_audit` instead and
never touched it. Measured on the real database, the checker fired 20 consecutive times with
`position_order_mismatch` and `cash_mismatch` -- an account disagreeing with its own order
ledger by up to 612 units -- and none of it reached a screen.

These tests read the shipped JavaScript and assert the contract, because the failure mode is
silence rather than an exception: nothing throws when a payload key is ignored.
"""
from pathlib import Path

from app.web import web as web_module

JS = web_module.WEB_ROOT / "js"


def read(name):
    return (JS / name).read_text(encoding="utf-8")


def test_the_health_view_reads_reconciliation():
    source = read("operations.js")
    assert "state.reconciliation" in source, (
        "renderHealth no longer reads state.reconciliation, so reconciliation findings "
        "are invisible again")


def test_reconciliation_findings_are_named_not_just_counted():
    """An operator needs to know which symbol disagrees and by how much."""
    source = read("operations.js")
    assert "findings" in source
    assert "differences" in source
    assert "difference" in source


def test_a_missing_side_is_shown_as_missing_rather_than_a_number():
    """The reconciler reports order_side_missing with a null difference; it must not render
    as a blank or a zero, which would read as agreement."""
    source = read("operations.js")
    assert "方向缺失" in source


def test_the_reconciliation_cell_distinguishes_agreement():
    source = read("operations.js")
    assert "一致" in source
    assert "尚未运行" in source


def test_a_disagreement_is_marked_as_a_fault():
    """The one reading that invalidates the rest of the page is coloured, not neutral."""
    source = read("operations.js")
    assert 'class="down"' in source


def test_an_out_of_distribution_signal_is_visibly_flagged():
    """298 of the 303 signals that passed the gate carried this warning.

    It was rendered, but concatenated into the same muted cell as "ensemble_agrees", so a
    signal the model had no basis for looked identical to one it did. The point of a
    warning is to be seen.
    """
    source = read("operations.js")
    assert "ALARM_CODES" in source
    assert "out_of_distribution_warning" in source
    # The alarm renders into a down-coloured element rather than the muted one.
    alarm_at = source.index("ALARM_CODES")
    assert '<small class="down">' in source[alarm_at:]


def test_an_unreviewed_weight_is_flagged_as_unreviewed():
    """MODEL_REQUIRE_PROMOTED is 0, so weights that failed review are traded.

    The table showed the bare word "candidate", which reads as an ordinary lifecycle state
    rather than as "this never passed review".
    """
    source = read("operations.js")
    assert "未通过审核" in source


def test_the_alarm_list_covers_the_codes_the_backend_emits():
    """A code the backend sends but the UI does not list stays invisible."""
    source = read("operations.js")
    for code in ("out_of_distribution", "degraded_features", "model_not_promoted",
                 "edge_gate_disabled"):
        assert code in source, code


def test_the_backend_still_sends_what_the_frontend_reads():
    """Guards the other direction: the payload key must keep existing."""
    snapshot = (Path(__file__).resolve().parents[1] / "app" / "backtest" / "snapshot.py")
    text = snapshot.read_text(encoding="utf-8")
    assert "'reconciliation'" in text, "the state payload stopped carrying reconciliation"


def test_every_view_module_parses_as_a_module():
    """A syntax error in a view blanks the page; there is no bundler to catch it."""
    for path in sorted(JS.rglob("*.js")):
        text = path.read_text(encoding="utf-8")
        # Modules use import/export, so a crude balance check is what is available without
        # a JS runtime in the test environment. Unbalanced braces mean a truncated file.
        assert text.count("{") == text.count("}"), "unbalanced braces in %s" % path.name
        assert text.count("(") == text.count(")"), "unbalanced parens in %s" % path.name
