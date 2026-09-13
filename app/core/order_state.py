"""Order lifecycle: the states, the legal transitions, and what each one means.

The same five strings were hardcoded in six modules -- the broker, the execution layer,
the storage queries, the runtime restore, the dashboard and the snapshot -- and each of
them decided independently which subset was "open". Two consequences were already
visible.

Storage asked for status IN ('OPEN','PARTIALLY_FILLED') and execution kept its own copy of
the same tuple, so adding a state meant finding every copy or silently leaving one behind.
More seriously, a deadline expiry and an explicit cancel both ended as CANCELED. Those are
different events -- one means the plan went stale and nobody acted, the other means
somebody decided to stop -- and an audit trail that cannot tell them apart cannot answer
why an order disappeared, which is the question it exists for.

This module is the single definition. The transition table is data rather than control
flow so it can be asserted against, and so an illegal transition is a returned reason
instead of a state the rest of the system has to cope with.
"""

import dataclasses

PENDING = "PENDING"
OPEN = "OPEN"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
CANCELED = "CANCELED"
EXPIRED = "EXPIRED"
REJECTED = "REJECTED"

# Where an order can still be acted on: it holds a position slot, a risk reservation and
# a place in the book.
OPEN_STATUSES = (PENDING, OPEN, PARTIALLY_FILLED)
# Where it is finished. Nothing after this changes the order.
TERMINAL_STATUSES = (FILLED, CANCELED, EXPIRED, REJECTED)
# Finished with a fill, versus finished without one. The distinction is what position and
# exposure accounting needs.
FILLED_STATUSES = (FILLED,)

ALL_STATUSES = OPEN_STATUSES + TERMINAL_STATUSES

# What each state may become. This table is enforced by the broker, so it has to admit
# every move the venue can actually produce and refuse the rest.
#
# PENDING used to allow only (OPEN, REJECTED, CANCELED). That is wrong in the one case
# that matters most: PENDING is in OPEN_STATUSES, so the broker treats such an order as
# actable and process() will fill it -- and status_for_fill returns FILLED or
# PARTIALLY_FILLED, neither of which the old table allowed. A marketable order fills
# immediately, so the single most common transition in the system was illegal and the
# table said so. Wiring the table in without this fix would have converted the first fill
# of every pending order into a refusal.
TRANSITIONS = {
    # Acknowledged, filled, or dead. An order can go straight from submitted to filled
    # because the venue does not promise an acknowledgement ahead of an execution.
    PENDING: (OPEN, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, REJECTED),
    OPEN: (PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, REJECTED),
    # PARTIALLY_FILLED -> PARTIALLY_FILLED is how a further partial fill is recorded.
    PARTIALLY_FILLED: (PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED),
    FILLED: (),
    CANCELED: (),
    EXPIRED: (),
    REJECTED: (),
}


def is_open(status):
    return status in OPEN_STATUSES


def filled(status):
    """Whether the order finished with quantity traded."""
    return status in FILLED_STATUSES or status == PARTIALLY_FILLED


def can_transition(source, target):
    return target in TRANSITIONS.get(source, ())


def transition(order, status, changes=None):
    """Move an order to a state, or refuse. Returns (order, reason).

    Refusing rather than raising: the caller is usually a socket handler reacting to a
    book event that arrived after the order was already resolved, which is a race and not
    a bug in the caller. The reason is returned so it can be counted.

    ``changes`` carries the other fields the move sets -- the filled quantity, the
    timestamp. A same-status move WITH changes is legal and is how a further partial fill
    is recorded; without changes it is the no-op a late book event produces.

    PaperOrder is a frozen dataclass, so the old ``order.status = status`` raised
    FrozenInstanceError on the only type this module exists for. The function was never
    called, which is why nobody found out: a lifecycle table that could not be applied to
    a single order in the system. Dataclass instances are rebuilt instead of mutated.
    """
    current = getattr(order, "status", None)
    updates = dict(changes or {})
    if current == status and not updates:
        return order, "already_" + str(status).lower()
    if current not in TRANSITIONS:
        return order, "unknown_status:" + str(current)
    if not can_transition(current, status):
        return order, "illegal_transition:%s->%s" % (current, status)
    updates["status"] = status
    return _rebuilt(order, updates), "ok"


def _rebuilt(order, updates):
    """The same order with new field values, mutating only what can be mutated."""
    if dataclasses.is_dataclass(order) and not isinstance(order, type):
        return dataclasses.replace(order, **updates)
    for name, value in updates.items():
        setattr(order, name, value)
    return order


def status_for_fill(quantity, filled_quantity, tolerance=1e-12):
    """The status a fill leaves an order in."""
    if float(filled_quantity) >= float(quantity) - tolerance:
        return FILLED
    return PARTIALLY_FILLED


def outcome(status):
    """What a terminal status means, in one word.

    cancel and expiry are both "the order did not fill", but only one of them was a
    decision, and reporting them as one number hides which is happening.
    """
    return {FILLED: "filled", PARTIALLY_FILLED: "partial", CANCELED: "canceled",
            EXPIRED: "expired", REJECTED: "rejected", PENDING: "pending",
            OPEN: "working"}.get(status, "unknown")


def summarise(orders):
    """Counts by status, plus the two groupings nothing used to compute."""
    counts = {status: 0 for status in ALL_STATUSES}
    for order in orders or ():
        status = getattr(order, "status", None)
        if status in counts:
            counts[status] += 1
    working = sum(counts[status] for status in OPEN_STATUSES)
    return {**counts, "working": working,
            "terminal": sum(counts[status] for status in TERMINAL_STATUSES),
            "filled_or_partial": counts[FILLED] + counts[PARTIALLY_FILLED],
            # Filled share of everything that finished. A low number here with a healthy
            # trade count means the strategies are resting orders that never get hit.
            "fill_rate": ((counts[FILLED] + counts[PARTIALLY_FILLED])
                          / max(1, sum(counts[status] for status in TERMINAL_STATUSES)))}
