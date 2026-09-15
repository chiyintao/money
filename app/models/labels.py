"""Triple-barrier labels, and how much each row is actually worth.

The dataset used to label every row with the same thing: the return from the next bar's
open to the close `horizon` bars later. That label describes a strategy nobody runs. The
live system does not hold for exactly twelve bars and exit at the close -- it has a stop, a
target, a trailing stop and a time stop, and it leaves when one of them is touched. A model
trained to predict a fixed-horizon return is therefore answering a question the execution
layer never asks, and the audit measured the consequence: the label distribution and the
realised trade distribution are not the same distribution.

Lopez de Prado's triple barrier labels the outcome the strategy actually experiences. From
each bar, three barriers are set -- an upper (profit target), a lower (stop) and a vertical
(time limit) -- and the label is whichever is touched first, with the return realised at that
touch. This is the labelling scheme the exit policy implies, written down.

Two further pieces are needed for the labels to be usable:

* **Uniqueness weights.** Consecutive rows share most of their outcome window. With a
  twelve-bar horizon, row i and row i+1 share eleven of them, so they are nearly the same
  observation counted twice. The audit measured label lag-1 autocorrelation at 0.557;
  treating those as independent samples understates the standard error, and naive t=14.0
  collapses to roughly t=5.15 once overlap is accounted for. Each row gets a weight that
  falls as its window overlaps its neighbours', so the fit stops counting one event many
  times.
* **A same-distribution guarantee.** The barriers come from the same exit policy function the
  live path calls, not from a copy of its constants. If the policy moves, the labels move.
"""

LABEL_SCHEME = "triple-barrier-v1"

# Which barrier ended the trade. Stored so a report can say what the model is actually
# predicting, and so a row whose outcome was ambiguous is visible rather than averaged in.
UPPER = "upper"
LOWER = "lower"
VERTICAL = "vertical"

# How to label a bar that touches BOTH barriers. OHLC cannot order the two touches, so the
# choice is a modelling decision, and it has to agree with the rule the account trades on.
AMBIGUOUS_STOP = "stop"      # matches execution_rules.evaluate_bar_exit; the default
AMBIGUOUS_TARGET = "target"  # the optimistic reading, for measuring the bias only


def triple_barrier(bars, index, stop_distance, target_distance, horizon, entry_price=None,
                   ambiguous_policy=AMBIGUOUS_STOP):
    """Resolve one row's label by walking forward until a barrier is touched.

    Returns a dict with the realised return, which barrier fired, and the bar it fired on.
    The walk is over the same bars the fixed-horizon label used, but it stops at the first
    touch instead of always running to `horizon`.

    **The tie-break matches the live exit rule, and that is the point.** OHLC data cannot
    say whether the high or the low of a bar came first, so a bar that spans both barriers
    is genuinely ambiguous. This function used to resolve it optimistically -- the upper
    barrier was tested first -- while `execution_rules.evaluate_bar_exit`, the rule the
    paper account actually trades on, resolves it conservatively by taking the stop. The
    model was therefore trained on one outcome and traded on its opposite: every bar that
    touched both levels was a winner in the training set and a loser in the account.

    The size of that is not small. The median row's ATR is 0.00318 against a 0.004 barrier,
    so one bar routinely spans both, and the labelled returns are asymmetric by construction
    -- mean |upper| 0.00676 against mean |lower| 0.00653 on 300k rows, which is 5.8% of the
    barrier of free optimism the model was fitted to.

    `ambiguous_policy` exists for research that wants to measure the optimistic case. The
    default is the one that agrees with execution, because a label that flatters the
    strategy is worse than no label: it teaches the model to expect a fill the broker will
    never give it.
    """
    if horizon < 1:
        raise ValueError("invalid_horizon")
    if stop_distance <= 0 or target_distance <= 0:
        raise ValueError("invalid_barrier")
    entry = float(entry_price if entry_price is not None else bars[index]["close"])
    if entry <= 0:
        raise ValueError("invalid_entry")
    upper = entry + target_distance
    lower = entry - stop_distance
    last = min(index + horizon, len(bars) - 1)
    optimistic = ambiguous_policy == AMBIGUOUS_TARGET
    for step in range(index + 1, last + 1):
        bar = bars[step]
        high = float(bar["high"]); low = float(bar["low"])
        touched_upper = high >= upper
        touched_lower = low <= lower
        stamp = int(bar.get("close_time") or bar.get("open_time") or 0)
        if touched_upper and touched_lower and not optimistic:
            return {"future_return": (lower - entry) / entry, "barrier": LOWER,
                    "barrier_time": stamp, "bars_held": step - index, "ambiguous": True}
        if touched_upper:
            return {"future_return": (upper - entry) / entry, "barrier": UPPER,
                    "barrier_time": stamp, "bars_held": step - index,
                    "ambiguous": bool(touched_lower)}
        if touched_lower:
            return {"future_return": (lower - entry) / entry, "barrier": LOWER,
                    "barrier_time": stamp, "bars_held": step - index, "ambiguous": False}
    exit_price = float(bars[last]["close"])
    return {"future_return": (exit_price - entry) / entry, "barrier": VERTICAL,
            "barrier_time": int(bars[last].get("close_time") or 0),
            "bars_held": last - index}


def uniqueness_weights(intervals, total):
    """Average uniqueness of each label interval, in (0, 1].

    `intervals` is a list of (start, end) bar indices, one per row, and `total` is the
    number of bars they live in. A row that shares its span with many others gets a small
    weight; a row that stands alone gets 1.

    This is the average-over-time form from Lopez de Prado, *Advances in Financial Machine
    Learning*, ch. 4: for each bar a label covers, count how many labels cover that bar, and
    take the mean of 1/count over the label's own span. Computed with a difference array, so
    it is O(rows + bars) rather than O(rows x horizon) -- at 1.2M rows the naive version is
    tens of billions of operations.
    """
    count = [0] * (total + 2)
    for start, end in intervals:
        if end < start:
            continue
        count[max(0, start)] += 1
        count[min(total, end) + 1] -= 1
    running = 0
    concurrency = [0] * (total + 1)
    for index in range(total + 1):
        running += count[index]
        concurrency[index] = running
    weights = []
    for start, end in intervals:
        if end < start:
            weights.append(0.0)
            continue
        lo = max(0, start); hi = min(total, end)
        span = hi - lo + 1
        if span <= 0:
            weights.append(0.0)
            continue
        total_weight = 0.0
        for index in range(lo, hi + 1):
            active = concurrency[index]
            total_weight += 1.0 / active if active > 0 else 0.0
        weights.append(total_weight / span)
    return weights


def sample_weights(rows, bars_held_key="bars_held", decay=1.0):
    """Per-row fit weights from label overlap, normalised to a mean of 1.

    The intervals are converted to BAR ORDINALS before the concurrency sweep, and that is
    not a detail. The recorded spans are exchange timestamps in milliseconds, so feeding
    them to `uniqueness_weights` directly makes it allocate one slot per millisecond: 587
    rows spanning 176 million ms took 134 seconds and most of a gigabyte, and a real
    1.2M-row dataset would not have finished at all. Weighting is about how many labels
    are alive at the same time, which is measured in bars, so bars is the unit used.

    The row's own position in timestamp order supplies the start, and `bars_held` -- which
    the labeller writes and which is exactly the number of bars the trade lived -- supplies
    the length. Reading the end timestamp instead would reintroduce the same unit problem
    and would be less precise, since a barrier rarely lands on an exact bar boundary.

    Normalising to a mean of 1 keeps the effective learning rate comparable to an
    unweighted fit, so a model trained on weights is not also silently trained at a
    different scale.
    """
    if not rows:
        return []
    # Dense rank of the distinct timestamps this row set actually contains: ordinal
    # position in time, counted in bars rather than milliseconds.
    stamps = sorted({int(row.get("timestamp") or 0) for row in rows})
    ordinal = {stamp: index for index, stamp in enumerate(stamps)}
    intervals = []
    for row in rows:
        start = ordinal.get(int(row.get("timestamp") or 0), 0)
        held = int(row.get(bars_held_key) or 0)
        intervals.append((start, start + max(0, held)))
    if not intervals:
        return []
    span = max(end for _, end in intervals) + 1
    weights = uniqueness_weights(intervals, max(1, span))
    if decay and decay != 1.0:
        weights = [weight ** float(decay) for weight in weights]
    mean = sum(weights) / len(weights) if weights else 0.0
    if mean <= 0:
        return [1.0] * len(weights)
    return [weight / mean for weight in weights]
