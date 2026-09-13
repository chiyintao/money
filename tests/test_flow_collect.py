import time

from app.market.flow_collect import BUCKET_MS, FlowCollector, bucket_of, liquidation_side, parse_force_order
from app.storage.storage import Store
from app.market.websocket_feed import parse_market_event


def test_bucket_boundaries_are_minute_aligned():
    assert bucket_of(0) == 0
    assert bucket_of(59_999) == 0
    assert bucket_of(60_000) == 60_000
    assert bucket_of(60_001) == 60_000
    assert bucket_of(125_000) == 120_000


def test_liquidation_side_is_the_position_not_the_order():
    # A forceOrder SELL is a long being closed out. Reading it as short-side volume
    # inverts every cascade.
    assert liquidation_side("SELL") == "long"
    assert liquidation_side("BUY") == "short"
    assert liquidation_side("") is None
    assert liquidation_side(None) is None


def test_force_order_payload_normalises():
    event = parse_force_order({"e": "forceOrder", "E": 1, "o": {
        "s": "BTCUSDT", "S": "SELL", "q": "0.014", "p": "9910", "ap": "9910.5",
        "z": "0.014", "T": 1568014460893, "X": "FILLED"}})
    assert event["type"] == "liquidation"
    assert event["symbol"] == "BTCUSDT"
    assert event["side"] == "long"
    assert event["quantity"] == 0.014
    assert event["price"] == 9910.5   # average fill price, not the limit price
    assert event["event_time"] == 1568014460893


def test_force_order_survives_a_combined_stream_envelope():
    event = parse_force_order({"stream": "!forceOrder@arr",
                               "data": {"e": "forceOrder", "o": {"s": "ETHUSDT",
                                                                 "S": "BUY", "z": "1",
                                                                 "ap": "10", "T": 5}}})
    assert event["symbol"] == "ETHUSDT"
    assert event["side"] == "short"


def test_the_socket_parser_routes_force_order_events():
    parsed = parse_market_event({"data": {"e": "forceOrder", "o": {
        "s": "SOLUSDT", "S": "SELL", "z": "3", "ap": "20", "T": 9}}})
    assert parsed["type"] == "liquidation"
    assert parsed["side"] == "long"


def test_trades_split_by_aggressor():
    collector = FlowCollector()
    collector.add_trade("BTCUSDT", 1_000, 2.0, 100.0, "BUY")
    collector.add_trade("BTCUSDT", 2_000, 1.0, 100.0, "SELL")
    row = collector.drain(now_ms=200_000)[0]
    assert row["buy_volume"] == 2.0
    assert row["sell_volume"] == 1.0
    assert row["trades"] == 2
    assert row["buy_trades"] == 1
    assert row["notional"] == 300.0
    assert row["largest_trade"] == 200.0
    assert row["delta_volume"] == 1.0
    assert abs(row["delta_ratio"] - 1 / 3) < 1e-9


def test_liquidations_are_signed_by_the_side_forced_out():
    collector = FlowCollector()
    collector.add_liquidation("BTCUSDT", 1_000, 5.0, 100.0, "long")
    collector.add_liquidation("BTCUSDT", 2_000, 2.0, 100.0, "short")
    row = collector.drain(now_ms=200_000)[0]
    assert row["liquidations"] == 2
    assert row["liquidation_long_volume"] == 5.0
    assert row["liquidation_short_volume"] == 2.0
    # Negative means longs were forced out.
    assert row["liquidation_delta"] == -3.0
    assert row["liquidation_volume"] == 7.0
    assert row["largest_liquidation"] == 5.0


def test_liquidation_share_is_relative_to_traded_volume():
    collector = FlowCollector()
    collector.add_trade("BTCUSDT", 1_000, 100.0, 10.0, "BUY")
    collector.add_liquidation("BTCUSDT", 1_000, 25.0, 10.0, "long")
    assert collector.drain(now_ms=200_000)[0]["liquidation_share"] == 0.25


def test_share_is_absent_rather_than_zero_when_nothing_traded():
    # An unmeasured ratio must be distinguishable from a measured zero: a minute with no
    # trades is not a minute with balanced flow.
    collector = FlowCollector()
    collector.add_liquidation("BTCUSDT", 1_000, 1.0, 10.0, "long")
    row = collector.drain(now_ms=200_000)[0]
    assert row["delta_ratio"] is None
    assert row["liquidation_share"] is None


def test_events_are_aggregated_per_symbol_and_per_minute():
    collector = FlowCollector()
    collector.add_trade("BTCUSDT", 1_000, 1.0, 10.0, "BUY")
    collector.add_trade("ETHUSDT", 1_000, 1.0, 10.0, "BUY")
    collector.add_trade("BTCUSDT", 61_000, 1.0, 10.0, "BUY")
    rows = collector.drain(now_ms=200_000)
    assert len(rows) == 3
    assert sorted((row["symbol"], row["event_time"]) for row in rows) == [
        ("BTCUSDT", 0), ("BTCUSDT", 60_000), ("ETHUSDT", 0)]


def test_the_current_minute_is_held_back():
    # Emitting a partial minute would persist a number the next drain then overwrites,
    # and a restart mid-minute would leave it permanently short.
    collector = FlowCollector()
    collector.add_trade("BTCUSDT", 60_000, 1.0, 10.0, "BUY")
    assert collector.drain(now_ms=60_500) == []
    assert len(collector.buckets) == 1
    assert len(collector.drain(now_ms=120_000)) == 1


def test_observe_routes_normalised_events():
    collector = FlowCollector()
    collector.observe({"type": "trade", "symbol": "BTCUSDT", "event_time": 1_000,
                       "quantity": 2.0, "price": 10.0, "side": "BUY"})
    collector.observe({"type": "liquidation", "symbol": "BTCUSDT", "event_time": 1_000,
                       "quantity": 1.0, "price": 10.0, "side": "long"})
    collector.observe({"type": "book", "symbol": "BTCUSDT", "event_time": 1_000})
    row = collector.drain(now_ms=200_000)[0]
    assert row["trades"] == 1 and row["liquidations"] == 1


def test_malformed_events_are_ignored():
    collector = FlowCollector()
    collector.add_trade(None, 1_000, 1.0, 10.0, "BUY")
    collector.add_trade("BTCUSDT", 1_000, 0, 10.0, "BUY")
    collector.add_liquidation("BTCUSDT", 1_000, 1.0, 10.0, None)
    assert collector.drain(now_ms=200_000) == []


def test_open_buckets_are_bounded():
    collector = FlowCollector(max_buckets=10)
    for index in range(50):
        collector.add_trade("S%dUSDT" % index, 1_000, 1.0, 10.0, "BUY")
    assert len(collector.buckets) == 10
    assert collector.stats["dropped"] == 40
    # The buckets that survive are the most recent ones.
    assert ("S49USDT", 0) in collector.buckets


def test_drain_due_respects_its_interval():
    collector = FlowCollector(drain_interval_ms=30_000)
    assert collector.drain_due(now_ms=1_000_000) is True
    collector.drain(now_ms=1_000_000)
    assert collector.drain_due(now_ms=1_010_000) is False
    assert collector.drain_due(now_ms=1_031_000) is True


def test_health_reports_what_was_seen():
    collector = FlowCollector()
    collector.add_trade("BTCUSDT", 1_000, 1.0, 10.0, "BUY")
    collector.add_liquidation("BTCUSDT", 1_000, 1.0, 10.0, "long")
    collector.drain(now_ms=200_000)
    health = collector.health()
    assert health["trades"] == 1 and health["liquidations"] == 1
    assert health["buckets"] == 1 and health["written"] == 0
    assert health["open_buckets"] == 0


def test_flow_rows_round_trip_through_storage(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    collector = FlowCollector()
    collector.add_trade("BTCUSDT", 60_000, 2.0, 100.0, "BUY")
    collector.add_trade("BTCUSDT", 61_000, 1.0, 100.0, "SELL")
    collector.add_liquidation("BTCUSDT", 62_000, 0.5, 100.0, "short")
    rows = collector.drain(now_ms=180_000)
    assert store.record_flow(rows) == 1
    back = store.flow("BTCUSDT", limit=10)
    assert len(back) == 1
    assert back[0]["buy_volume"] == 2.0 and back[0]["sell_volume"] == 1.0
    assert back[0]["liquidations"] == 1 and back[0]["liquidation_short_volume"] == 0.5
    store.close()


def test_flow_is_idempotent_for_a_repeated_bucket(tmp_path):
    # The drain can legitimately re-emit a bucket if the process restarted mid-minute, so
    # the write must replace rather than accumulate.
    store = Store(str(tmp_path / "test.db"))
    row = {"symbol": "BTCUSDT", "event_time": 60_000, "buy_volume": 1.0, "sell_volume": 2.0,
           "trades": 3, "buy_trades": 1, "notional": 30.0, "largest_trade": 20.0}
    store.record_flow([row])
    store.record_flow([{**row, "buy_volume": 5.0}])
    back = store.flow("BTCUSDT")
    assert len(back) == 1 and back[0]["buy_volume"] == 5.0
    store.close()


def test_flow_since_filters_by_time(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    store.record_flow([{"symbol": "BTCUSDT", "event_time": 60_000, "buy_volume": 1.0},
                       {"symbol": "BTCUSDT", "event_time": 120_000, "buy_volume": 2.0}])
    assert len(store.flow(since=100_000)) == 1
    assert store.flow(since=100_000)[0]["event_time"] == 120_000
    store.close()


def test_flow_rows_are_pruned_to_a_bound(tmp_path):
    # The socket prints thousands of trades a minute; unbounded this table would grow
    # without limit, which is exactly the failure the events table had.
    store = Store(str(tmp_path / "test.db"))
    store.record_flow([{"symbol": "BTCUSDT", "event_time": index * 60_000,
                        "buy_volume": 1.0} for index in range(50)])
    assert store.prune_flow(keep_rows=10) == 40
    remaining = store.flow(limit=100)
    assert len(remaining) == 10
    # The newest rows survive, not the oldest.
    assert remaining[-1]["event_time"] == 49 * 60_000
    store.close()


def test_non_finite_values_are_stored_as_null(tmp_path):
    store = Store(str(tmp_path / "test.db"))
    store.record_flow([{"symbol": "BTCUSDT", "event_time": 60_000,
                        "buy_volume": float("nan"), "delta_ratio": float("inf")}])
    back = store.flow("BTCUSDT")[0]
    assert back["buy_volume"] == 0.0  # NOT NULL column falls back to its default
    assert back["delta_ratio"] is None
    store.close()

def flow_row(index, **overrides):
    row = {"symbol": "BTCUSDT", "event_time": index * 60_000, "trades": 10 + index,
           "buy_trades": 5, "buy_volume": 2.0 + index, "sell_volume": 1.0,
           "notional": 300.0, "largest_trade": 100.0 * (index + 1),
           "delta_volume": 1.0, "delta_ratio": 0.01 * index, "liquidations": 0,
           "liquidation_long_volume": 0.0, "liquidation_short_volume": 0.0,
           "liquidation_delta": 0.0, "liquidation_volume": 0.0,
           "largest_liquidation": 0.0, "liquidation_share": 0.0}
    row.update(overrides)
    return row


def test_flow_features_report_unmeasured_rather_than_neutral():
    from app.features.order_flow import flow_features

    assert flow_features([])["flow_buckets"] == 0
    assert flow_features([flow_row(0)])["flow_buckets"] == 0  # one row is not a history


def test_flow_features_measure_imbalance_against_its_own_history():
    from app.features.order_flow import flow_features

    steady = [flow_row(index, delta_ratio=0.01) for index in range(60)]
    quiet = flow_features(steady)
    assert quiet["flow_delta_ratio"] == 0.01
    # A flat history has no deviation, so nothing in it can be unusual.
    assert quiet["flow_delta_z"] == 0.0
    assert quiet["flow_buckets"] == 60
    moving = [flow_row(index, delta_ratio=0.01) for index in range(59)]
    moving.append(flow_row(59, delta_ratio=0.9))
    assert flow_features(moving)["flow_delta_z"] > 5


def test_liquidation_pressure_carries_the_side_that_was_forced_out():
    from app.features.order_flow import flow_features

    base = [flow_row(index) for index in range(59)]
    squeezed = base + [flow_row(59, liquidations=1, liquidation_short_volume=50.0,
                                 liquidation_delta=50.0, liquidation_volume=50.0,
                                 liquidation_share=0.4)]
    assert flow_features(squeezed)["liquidation_pressure"] > 0   # shorts carried out
    flushed = base + [flow_row(59, liquidations=1, liquidation_long_volume=50.0,
                               liquidation_delta=-50.0, liquidation_volume=50.0,
                               liquidation_share=0.4)]
    assert flow_features(flushed)["liquidation_pressure"] < 0    # longs carried out


def test_liquidation_features_are_zero_when_nothing_was_liquidated():
    from app.features.order_flow import flow_features

    rows = [flow_row(index) for index in range(10)]
    measured = flow_features(rows)
    assert measured["liquidation_share"] == 0.0
    assert measured["liquidation_pressure"] == 0.0
    heavy = [flow_row(index) for index in range(10)]
    heavy[-1].update(liquidations=3, liquidation_volume=80.0, liquidation_share=0.75,
                     liquidation_short_volume=80.0, liquidation_delta=80.0)
    assert flow_features(heavy)["liquidation_share"] == 0.75
    assert flow_features(heavy)["liquidation_pressure"] > 0
    # A spike with no side to it is not directional, so it carries no pressure: the sign is
    # the whole content of the signal.
    sided = [flow_row(index) for index in range(10)]
    sided[-1].update(liquidations=3, liquidation_volume=80.0, liquidation_share=0.75,
                     liquidation_long_volume=40.0, liquidation_short_volume=40.0)
    assert flow_features(sided)["liquidation_pressure"] == 0.0


def test_flow_family_is_exposed_to_the_dataset_builder():
    from app.features.order_flow import FAMILIES

    assert set(FAMILIES) >= {"order_flow", "positioning", "flow"}
