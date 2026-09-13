"""Reconciliation between what the account believes and what the ledger says.

Named account_reconcile rather than reconcile because app/reconcile.py already held the
candle-series gap detector -- a different concern that happens to share the word. Going
straight to reconcile.py overwrote that file, which is what this name is here to prevent.

The paper account keeps positions, cash and equity in memory. Fills are written to the
audit trail, orders to the orders table, and the session account is snapshotted into the
runtime key-value store. Nothing ever compared those views against each other.

They can disagree for ordinary reasons. A fill accepted by the account but refused by risk
is recorded as refused and must not appear in the position. A restart reloads orders from
storage while the account is rebuilt from the session, so anything written between the two
is only in one place. A crash between a position change and its snapshot loses the change
for good. None of these raise; all of them mean the numbers on the dashboard are not the
numbers a trade would have got.

So this compares the views and reports the differences with their size. It deliberately
does not repair anything: silently correcting a discrepancy destroys the evidence that it
happened, and the size of the discrepancy is the interesting part.
"""
import time

from ..core import order_state

# Below this, a difference is float noise from summing the same quantities in a different
# order rather than a real disagreement.
NOTIONAL_TOLERANCE = 1e-6
QUANTITY_TOLERANCE = 1e-9


def _number(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number


def position_quantities(account):
    """Held quantity per symbol, whatever shape the account stores positions in.

    The account's Position stores its size as `qty`; this read `quantity` and fell back to
    a dict key of the same name, so for the real account it returned 0 for every symbol.
    The consequence was not a wrong number in the report -- it was that
    `compare_positions` compared the fill ledger against a set of zeros, so every position
    mismatch either vanished (fills also zero) or reported the entire position as the
    discrepancy, and the check could never do its job. Both spellings are accepted now,
    which is what the "whatever shape" in this docstring was always promising.
    """
    positions = getattr(account, "positions", None) or {}
    values = {}
    if isinstance(positions, dict):
        items = positions.items()
    else:
        items = ((getattr(item, "symbol", None), item) for item in positions)
    for symbol, position in items:
        if not symbol:
            continue
        quantity = None
        for name in ("qty", "quantity"):
            if isinstance(position, dict):
                quantity = position.get(name)
            else:
                quantity = getattr(position, name, None)
            if quantity is not None:
                break
        values[symbol] = _number(quantity)
    return values


def signed_fills(fills):
    """Net signed quantity per symbol from a fill history.

    A BUY adds, a SELL subtracts; the account holds the sum. This is the definition the
    position ledger is supposed to be the running total of, so it is the one to check.
    """
    totals = {}
    for fill in fills or ():
        symbol = fill.get("symbol") if isinstance(fill, dict) else getattr(fill, "symbol", None)
        if not symbol:
            continue
        side = (fill.get("side") if isinstance(fill, dict) else getattr(fill, "side", "")) or ""
        quantity = _number(fill.get("quantity") if isinstance(fill, dict)
                           else getattr(fill, "quantity", 0))
        direction = 1.0 if str(side).upper() in ("BUY", "LONG") else -1.0
        totals[symbol] = totals.get(symbol, 0.0) + direction * quantity
    return totals


def compare_positions(expected, actual, quantity_tolerance=QUANTITY_TOLERANCE):
    symbols = sorted(set(expected) | set(actual))
    differences = []
    for symbol in symbols:
        want = _number(expected.get(symbol))
        have = _number(actual.get(symbol))
        if abs(want - have) > quantity_tolerance:
            differences.append({"symbol": symbol, "ledger": want, "account": have,
                                "difference": have - want})
    return differences


def compare_cash(account, fills, fees_key="fees", tolerance=NOTIONAL_TOLERANCE):
    """Cash movement implied by the fills against the cash the account reports.

    Only meaningful when a starting balance is recorded, so it reports raw evidence rather
    than an assertion: without one there is nothing to compare against and the caller is
    told so instead of being handed a number that looks like a finding.
    """
    starting = getattr(account, "initial_cash", None)
    if starting is None:
        return {"status": "unavailable", "reason": "no_starting_cash"}
    realised = 0.0
    fees = 0.0
    funding = 0.0
    for fill in fills or ():
        realised += _number(fill.get("realized_pnl"))
        fees += _number(fill.get(fees_key))
        funding += _number(fill.get("funding"))
    expected = _number(starting) + realised - fees - funding
    actual = _number(getattr(account, "cash", 0))
    return {"status": "ok", "starting_cash": _number(starting), "realized_pnl": realised,
            "fees": fees, "funding": funding, "expected_cash": expected, "account_cash": actual,
            "difference": actual - expected,
            "within_tolerance": abs(actual - expected) <= tolerance}


def compare_open_orders(broker, persisted):
    """Orders the broker is working against the ones storage believes are working.

    A mismatch here is the operational one: an order storage thinks is open but the broker
    does not is a phantom that holds a position slot after a restart, and an order the
    broker holds but storage does not is one that will vanish from the audit trail.
    """
    live = {order.order_id: order for order in broker.open_orders()}
    stored = {}
    for row in persisted or ():
        status = row.get("status") if isinstance(row, dict) else getattr(row, "status", None)
        if not order_state.is_open(status):
            continue
        key = row.get("order_id") if isinstance(row, dict) else getattr(row, "order_id", None)
        if key:
            stored[key] = row
    only_live = sorted(set(live) - set(stored))
    only_stored = sorted(set(stored) - set(live))
    quantity_mismatch = []
    for key in sorted(set(live) & set(stored)):
        row = stored[key]
        stored_filled = _number(row.get("filled_quantity") if isinstance(row, dict)
                                else getattr(row, "filled_quantity", 0))
        if abs(live[key].filled_quantity - stored_filled) > QUANTITY_TOLERANCE:
            quantity_mismatch.append({"order_id": key,
                                      "broker_filled": live[key].filled_quantity,
                                      "stored_filled": stored_filled})
    return {"live": len(live), "stored": len(stored), "only_live": only_live,
            "only_stored": only_stored, "quantity_mismatch": quantity_mismatch,
            "consistent": not (only_live or only_stored or quantity_mismatch)}


def compare_exposure(account, positions=None, tolerance=NOTIONAL_TOLERANCE):
    """Committed margin against what the open positions imply.

    Margin is the number the liquidation check reads, so a margin figure that has drifted
    from the positions it is supposed to cover is a risk finding rather than a bookkeeping
    one.
    """
    reported = getattr(account, "margin_used", None)
    if reported is None:
        return {"status": "unavailable", "reason": "no_margin_field"}
    positions = positions if positions is not None else position_quantities(account)
    held = getattr(account, "positions", None) or {}
    computed = 0.0
    covered = 0
    for symbol, quantity in positions.items():
        position = held.get(symbol) if isinstance(held, dict) else None
        price = _number(getattr(position, "mark_price", None))
        leverage = _number(getattr(position, "leverage", None), 1.0)
        if price <= 0 or leverage <= 0:
            continue
        computed += abs(quantity) * price / leverage
        covered += 1
    return {"status": "ok", "positions_covered": covered, "positions_held": len(positions),
            "reported_margin": _number(reported), "computed_margin": computed,
            "difference": _number(reported) - computed,
            "within_tolerance": abs(_number(reported) - computed) <= max(tolerance,
                                                                        computed * 1e-6)}


def order_position_mismatches(account, orders, tolerance=QUANTITY_TOLERANCE):
    """Per symbol: quantity the orders say filled against quantity the position holds.

    The account and the order ledger are two independent records of the same event. An
    order that reached PARTIALLY_FILLED and then EXPIRED leaves a position whose size no
    order ever completed -- `is_open(PARTIALLY_FILLED)` is true, so the order stops being
    worked, the position stays, and nothing anywhere states that the entry never finished.
    Risk sizes from the position while the order side believes it filled less.

    Only orders that are working or finished-with-a-fill are counted, and only for symbols
    a position exists on: an order for a symbol with no position is the entry-not-yet-
    accepted case, which `compare_open_orders` already covers.
    """
    held = position_quantities(account)
    fills_by_symbol = {}
    for order in orders or ():
        symbol = getattr(order, 'symbol', None)
        if not symbol or symbol not in held:
            continue
        filled = _number(getattr(order, 'filled_quantity', 0))
        if filled <= 0:
            continue
        side = (getattr(order, 'side', '') or '').upper()
        direction = 1.0 if side in ('BUY', 'LONG') else -1.0
        fills_by_symbol[symbol] = fills_by_symbol.get(symbol, 0.0) + direction * filled
    mismatches = []
    for symbol in sorted(fills_by_symbol):
        want = fills_by_symbol[symbol]
        have = held.get(symbol, 0.0)
        if abs(abs(want) - abs(have)) > tolerance:
            mismatches.append({'symbol': symbol, 'from_orders': want, 'held': have,
                               'difference': have - want})
    return mismatches


def reconcile(runtime):
    """Compare every view of the account and report what disagrees.

    Never raises: this runs on a timer inside the decision loop, and a monitoring step that
    can take trading down is worse than the discrepancy it was watching for.
    """
    report = {"checked_at": int(time.time() * 1000), "findings": []}
    try:
        account = runtime.account
        broker = runtime.broker
        trades = getattr(account, "trades", None) or []
        # `trades` holds CLOSED round trips, not fills: a symbol appears in it only after
        # its position is gone, and then with a net of zero. Comparing it against held
        # positions therefore said nothing about an open position -- it reported a clean
        # match for a position nobody had checked. The order ledger is the record that
        # actually describes how a position was built, so that is what is compared.
        actual = position_quantities(account)
        orders = list(getattr(getattr(runtime, 'broker', None), 'orders', {}).values())
        differences = order_position_mismatches(account, orders)
        report["positions"] = {"held": actual,
                               "from_orders": {m['symbol']: m['from_orders'] for m in differences},
                               "differences": differences,
                               "closed_round_trips": len(trades)}
        if differences:
            report["findings"].append("position_order_mismatch")
        cash = compare_cash(account, trades)
        report["cash"] = cash
        if cash.get("status") == "ok" and not cash.get("within_tolerance"):
            report["findings"].append("cash_mismatch")
        try:
            persisted = runtime.store.orders(active_only=True)
        except Exception as exc:
            persisted = []
            report["orders_error"] = repr(exc)
        orders = compare_open_orders(broker, persisted)
        report["orders"] = orders
        if not orders["consistent"]:
            report["findings"].append("open_order_mismatch")
        exposure = compare_exposure(account, positions=actual)
        report["exposure"] = exposure
        if exposure.get("status") == "ok" and not exposure.get("within_tolerance"):
            report["findings"].append("margin_mismatch")
        report["order_outcomes"] = order_state.summarise(broker.orders.values())
        # A status change the lifecycle refused is a book event the venue produced and the
        # account did not record. It is exactly the kind of divergence this report exists
        # to surface, and it used to be neither refused nor counted.
        refusals = dict(getattr(broker, "transition_refusals", None) or {})
        report["order_transition_refusals"] = refusals
        if refusals:
            report["findings"].append("order_transition_refused")
    except Exception as exc:
        report["error"] = "%s: %s" % (type(exc).__name__, exc)
        report["findings"].append("reconcile_failed")
    report["consistent"] = not report["findings"]
    return report
