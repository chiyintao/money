"""Contract specs come from the exchange's own filters, not from one shared default."""
from app.trading.broker import ContractSpec
from app.market.market import DEFAULT_MIN_NOTIONAL, DEFAULT_STEP_SIZE, DEFAULT_TICK_SIZE, build_contract_specs


def contract(symbol, tick="0.010", step="0.001", market_step="0.001", min_notional="5",
             contract_type="PERPETUAL", quote="USDT", status="TRADING"):
    return {
        "symbol": symbol, "contractType": contract_type, "quoteAsset": quote, "status": status,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": tick},
            {"filterType": "LOT_SIZE", "stepSize": step},
            {"filterType": "MARKET_LOT_SIZE", "stepSize": market_step},
            {"filterType": "MIN_NOTIONAL", "notional": min_notional},
        ],
    }


def test_specs_are_read_per_symbol():
    info = {"symbols": [contract("BTCUSDT", tick="0.10", market_step="0.001", min_notional="100"),
                        contract("SHIBUSDT", tick="0.0000010", market_step="1", min_notional="5")]}
    specs = build_contract_specs(info)
    assert specs["BTCUSDT"].tick_size == 0.10
    assert specs["BTCUSDT"].min_notional == 100.0
    # A 0.0000010 quote and a 0.10 quote cannot round quantity the same way, which is
    # exactly what one shared hardcoded spec got wrong.
    assert specs["SHIBUSDT"].tick_size == 0.000001
    assert specs["SHIBUSDT"].step_size == 1.0
    assert specs["BTCUSDT"].step_size == 0.001


def test_market_lot_size_wins_over_the_general_lot_size():
    # The service submits market orders, so the market filter is the one that decides
    # whether the rounded quantity is actually tradeable.
    specs = build_contract_specs({"symbols": [contract("XUSDT", step="0.001", market_step="0.01")]})
    assert specs["XUSDT"].step_size == 0.01


def test_non_perpetual_and_untradeable_contracts_are_skipped():
    info = {"symbols": [contract("AUSDT"), contract("BUSDT", contract_type="CURRENT_QUARTER"),
                        contract("CUSDT", status="BREAK"), contract("DUSDT", quote="BUSD")]}
    assert set(build_contract_specs(info)) == {"AUSDT"}


def test_missing_filters_fall_back_to_tradeable_defaults():
    specs = build_contract_specs({"symbols": [{"symbol": "XUSDT", "contractType": "PERPETUAL",
                                               "quoteAsset": "USDT", "status": "TRADING"}]})
    spec = specs["XUSDT"]
    assert (spec.tick_size, spec.step_size, spec.min_notional) == (
        DEFAULT_TICK_SIZE, DEFAULT_STEP_SIZE, DEFAULT_MIN_NOTIONAL)


def test_unparsable_and_zero_filter_values_fall_back():
    info = {"symbols": [contract("XUSDT", tick="abc", step="0", min_notional="")]}
    spec = build_contract_specs(info)["XUSDT"]
    assert spec.tick_size == DEFAULT_TICK_SIZE
    assert spec.step_size == DEFAULT_STEP_SIZE
    assert spec.min_notional == DEFAULT_MIN_NOTIONAL


def test_empty_or_malformed_info_yields_no_specs():
    assert build_contract_specs(None) == {}
    assert build_contract_specs({}) == {}
    assert build_contract_specs({"symbols": None}) == {}


def test_specs_round_the_way_the_exchange_does():
    spec = ContractSpec("XUSDT", tick_size=.05, step_size=1.0, min_notional=5.0)
    assert spec.round_qty(2.9) == 2.0
    assert spec.round_price(10.03) == 10.05
