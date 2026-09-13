"""Out-of-sample portfolio evidence tests."""
import json

from app.models.portfolio_oos import MIN_TRADES, decision_strategy, gate_verdict, oos_decisions, portfolio_evidence, series_from_rows
from app.models.training_job import TrainingRunner

DAY = 86_400_000


def bars(symbol, count=400, step=300_000, price=100.0, drift=0.0):
    rows = []
    for index in range(count):
        level = price * (1 + drift * index)
        rows.append({"symbol": symbol, "open_time": index * step,
                     "close_time": index * step + step - 1, "open": level,
                     "high": level * 1.02, "low": level * 0.98, "close": level,
                     "volume": 10.0, "is_closed": 1})
    return rows


def funding(symbols=("AUSDT", "BUSDT"), steps=400, rate=0.0001):
    """Published funding rates, one per eight-hour window, for the replay to charge.

    The replay used to apply fees and slippage and no funding, while the live loop charges
    funding on every tick on every open position. costs_included asserted all costs were in
    a number that omitted the only cost a perpetual position pays for simply existing, and
    the promotion gate reads exactly that flag.
    """
    window = 8 * 3600 * 1000
    span = steps * 300_000
    return {symbol: [(index * window + window - 1, rate)
                     for index in range(max(1, span // window))]
            for symbol in symbols}


def dataset_rows(symbols=("AUSDT", "BUSDT"), count=400, drift=0.0):
    """A bar series that passes the same OHLCV validation the backtest applies.

    The high and low have to bracket open and close on every bar: a fixture whose range was
    fixed while the close drifted out of it fails validation from the bar where it escapes,
    and the replay reports "invalid_ohlcv" instead of the portfolio result it was meant to
    produce.
    """
    rows = []
    for symbol in symbols:
        for index in range(count):
            opening = 100.0 * (1 + drift * index) + index * 0.01
            closing = opening + 0.005
            rows.append({"symbol": symbol, "timestamp": index * 300_000,
                         "open_time": index * 300_000,
                         "close_time": index * 300_000 + 299_999,
                         "open": opening, "high": max(opening, closing) + 0.01,
                         "low": min(opening, closing) - 0.01, "close": closing,
                         "volume": 10.0, "is_closed": 1})
    return rows


def test_decisions_use_the_round_trip_cost_as_the_threshold():
    oos = [{"timestamp": 1, "symbol": "A", "predicted_return": 0.0005},
           {"timestamp": 2, "symbol": "A", "predicted_return": 0.005},
           {"timestamp": 3, "symbol": "A", "predicted_return": -0.005},
           {"timestamp": 4, "symbol": "A", "predicted_return": -0.0005}]
    decisions = oos_decisions(oos, cost_bps=12)
    chosen = decisions["A"]
    # 12bp is 0.0012, so a 5bp prediction is not a trade and a 50bp one is.
    assert sorted(chosen) == [2, 3]
    assert chosen[2]["side"] == "LONG"
    assert chosen[3]["side"] == "SHORT"


def test_records_without_a_symbol_are_skipped():
    assert oos_decisions([{"timestamp": 1, "predicted_return": 0.5}]) == {}


def test_the_strategy_only_decides_on_recorded_timestamps():
    decisions = {"A": {100: {"side": "LONG", "predicted_return": 0.01, "timestamp": 100}}}
    strategy = decision_strategy(decisions)
    history = [{"open_time": 100, "close": 50.0}]
    plan = strategy("A", history)
    assert plan["side"] == "LONG"
    assert plan["stop"] < plan["price"] < plan["target"]
    # A bar the walk-forward never predicted produces no decision, which is what keeps the
    # replay out-of-sample rather than trading on bars inside the training window.
    assert strategy("A", [{"open_time": 200, "close": 50.0}]) is None
    assert strategy("B", history) is None
    assert strategy("A", []) is None


def test_a_short_places_its_target_below_the_price():
    decisions = {"A": {100: {"side": "SHORT", "predicted_return": -0.01, "timestamp": 100}}}
    plan = decision_strategy(decisions)("A", [{"open_time": 100, "close": 50.0}])
    assert plan["target"] < plan["price"] < plan["stop"]


def test_a_zero_price_produces_no_plan():
    decisions = {"A": {100: {"side": "LONG", "predicted_return": 0.01, "timestamp": 100}}}
    assert decision_strategy(decisions)("A", [{"open_time": 100, "close": 0.0}]) is None


def test_series_from_rows_groups_by_symbol():
    grouped = series_from_rows(dataset_rows(("AUSDT", "BUSDT"), count=10))
    assert sorted(grouped) == ["AUSDT", "BUSDT"]
    assert len(grouped["AUSDT"]) == 10


def test_one_symbol_is_not_a_portfolio():
    evidence = portfolio_evidence(bars("AUSDT"), [])
    assert evidence["status"] == "unavailable"
    assert evidence["reason"].startswith("portfolio_needs_multiple_symbols")


def test_no_decisions_is_reported_rather_than_scored_as_zero():
    evidence = portfolio_evidence(dataset_rows(), [])
    assert evidence["status"] == "unavailable"
    assert evidence["reason"].startswith("too_few_symbols_with_decisions")


def test_the_replay_runs_a_real_account():
    rows = dataset_rows(("AUSDT", "BUSDT"), count=400)
    oos = []
    for index in range(60, 380, 4):
        for symbol in ("AUSDT", "BUSDT"):
            oos.append({"timestamp": index * 300_000, "symbol": symbol,
                        "predicted_return": 0.01, "actual_return": 0.005, "fold": 0})
    evidence = portfolio_evidence(rows, oos, cost_bps=1.0, interval="5m",
                                  funding=funding())
    assert evidence["status"] in ("ok", "insufficient_trades", "no_trades")
    assert evidence["costs_included"] is True
    assert evidence["total_funding"] != 0.0, "funding must be charged, not reported as zero"
    assert "net_return" in evidence and "max_drawdown" in evidence
    assert evidence["starting_equity"] == 10000.0
    # A replay produces a number even when it produces no trades; the counts are what
    # distinguish "flat because there was nothing to do" from "flat because it lost".
    assert isinstance(evidence["trades"], int)


def test_the_gate_mirrors_the_registry_conditions():
    passing = {"status": "ok", "costs_included": True, "trades": 40,
               "net_return": 0.02, "max_drawdown": 0.05}
    assert gate_verdict(passing)["passes"] is True
    assert gate_verdict({**passing, "net_return": 0.0})["failures"] == [
        "net_return_not_positive"]
    assert gate_verdict({**passing, "costs_included": False})["passes"] is False
    assert gate_verdict({**passing, "trades": MIN_TRADES - 1})["failures"] == [
        "insufficient_trades"]
    assert gate_verdict({**passing, "max_drawdown": 0.25})["failures"] == [
        "drawdown_outside_limit"]
    assert gate_verdict({**passing, "net_return": float("nan")})["failures"] == [
        "non_finite_net_return"]


def test_a_missing_evidence_block_fails_with_its_own_reason():
    assert gate_verdict(None)["passes"] is False
    assert gate_verdict({"status": "unavailable", "reason": "no_data"})["failures"] == ["no_data"]


def test_the_evidence_reaches_the_candidate_manifest(tmp_path):
    # The gate reads metrics.portfolio_oos; the training run computes it after the manifest
    # is written, so without this the two never meet and the gate reports a missing file for
    # a model that has the evidence.
    candidate = tmp_path / "lightgbm-abc"
    candidate.mkdir()
    (candidate / "manifest.json").write_text(json.dumps({
        "status": "candidate", "metrics": {"test": {"rows": 100}},
        "feature_version": "features-v3"}), encoding="utf-8")
    evidence = {"status": "ok", "costs_included": True, "trades": 40,
                "net_return": 0.02, "max_drawdown": 0.05}
    # Lives on TrainingRunner, which owns the run stages that produce the evidence.
    TrainingRunner._attach_portfolio_evidence(str(candidate), evidence)
    manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metrics"]["portfolio_oos"] == evidence
    # The existing metrics are preserved: the gate reads both blocks.
    assert manifest["metrics"]["test"] == {"rows": 100}


def test_attaching_nothing_is_a_no_op(tmp_path):
    candidate = tmp_path / "lightgbm-abc"
    candidate.mkdir()
    (candidate / "manifest.json").write_text("{}", encoding="utf-8")
    assert TrainingRunner._attach_portfolio_evidence(str(candidate), None) is None
    assert json.loads((candidate / "manifest.json").read_text(encoding="utf-8")) == {}


def _replay(symbols=("AUSDT", "BUSDT"), rate=0.0001, cost_bps=1.0, windows=None,
            drift=0.003):
    """One replay. rate=None supplies no rates at all; a float supplies that rate.

    The bars rise, so positions reach their bracket and close. On a flat series the exit
    policy almost never fires, the replay produces two trades, and every assertion about
    the account's costs would be made on a sample too small to carry it.
    """
    rows = dataset_rows(symbols, count=400, drift=drift)
    oos = [{"timestamp": index * 300_000, "symbol": symbol,
            "predicted_return": 0.01, "actual_return": 0.005, "fold": 0}
           for index in range(60, 380, 4) for symbol in symbols]
    return portfolio_evidence(rows, oos, cost_bps=cost_bps, interval="5m",
                              funding=None if rate is None else funding(symbols, rate=rate),
                              windows=windows)


def test_a_long_pays_positive_funding_and_receives_negative_funding():
    """The sign is the whole reason the rate is published.

    The replay never charged funding at all, so the sign could not have been wrong -- and
    the evidence block claimed costs were included while omitting the one cost a perpetual
    position pays for simply existing. A crowded long pays; the other side is paid.
    Evidence that cannot tell those apart cannot rank two strategies that differ only in
    which side of the book they hold.
    """
    paying = _replay(rate=0.0001)
    receiving = _replay(rate=-0.0001)
    free = _replay(rate=0.0)
    assert paying["total_funding"] > 0
    assert receiving["total_funding"] < 0
    assert paying["net_return"] < free["net_return"] < receiving["net_return"]
    # Same trades and same prices, so the ordering is funding. Fees move by a hair because
    # funding reduces cash and cash sets the next position's size -- assert the size of the
    # effect rather than pretending it is exactly zero.
    assert paying["trades"] == receiving["trades"] == free["trades"]
    charged = free["ending_equity"] - paying["ending_equity"]
    assert charged > 0, "charging a cost must reduce the account"
    # Not an equality: funding reduces cash, cash sets the next position's size, so the two
    # runs diverge slightly after the first settlement. The assertion is that the funding is
    # the effect being measured rather than a rounding artifact riding on something else.
    assert charged >= abs(paying["total_funding"]) * 0.5
    assert abs(paying["total_fees"] - free["total_fees"]) < abs(paying["total_funding"]) * 0.2


def test_evidence_without_funding_says_the_cost_is_missing():
    """costs_included was a constant, and the gate reads exactly that flag.

    The replay applied fees and slippage -- the costs the paper account applies at fill
    time -- and no funding, while the live loop charges funding on every tick on every open
    position. So the flag asserted that every cost was in a number that omitted a real one.
    Absent rates now make it False and name what is missing, and the gate refuses.
    """
    evidence = _replay(rate=None)
    # The replay still ran: funding is a statement about the cost claim, not about whether
    # the account could be simulated.
    assert evidence["trades"] > 0
    assert evidence["net_return"] != 0.0
    assert evidence["costs_included"] is False
    assert evidence["costs_missing"] == ["funding"]
    # And its absence is visible on its own line, not only inferable from a zero.
    assert "total_funding" not in evidence
    verdict = gate_verdict(evidence)
    assert verdict["passes"] is False
    assert "costs_not_included" in verdict["failures"]


def test_the_settlement_window_is_honoured_rather_than_assumed():
    """The venue settles every eight hours on most contracts, four or one on some.

    A hardcoded window charges a four-hour contract half as often as it should and an
    eight-hour contract twice. The account keys the window per symbol and only advances it
    when the charge is booked; this asserts the replay actually reaches that logic.
    """
    eight_hour = _replay(rate=0.0001)
    four_hour = _replay(rate=0.0001,
                        windows={"AUSDT": 4 * 3600 * 1000, "BUSDT": 4 * 3600 * 1000})
    assert four_hour["total_funding"] > eight_hour["total_funding"]
