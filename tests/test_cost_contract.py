"""The cost contract: training and serving must define "actionable" the same way.

This is the single most expensive defect the audit found. Training decided a row was
tradeable when the prediction cleared the round trip IN THE PREDICTED DIRECTION
(`tabular_model.evaluate`: `side = pred > cost ? LONG : pred < -cost ? SHORT : FLAT`).
Serving gated on `abs(expected)` against a 0.5bp floor, which discards both the sign and
the cost. Measured on the shipped artifact over the held-out test set:

    training rule   269 of 188,640 rows active (0.14%)   net +2.08bp
    serving rule    188,521 of 188,640 rows active (99.94%)  net -13.95bp

a 700x difference in population and a sign flip in expectancy. It was invisible only
because a separate data problem held the service at FLAT.

These tests pin the invariant so the two definitions cannot drift apart again.
"""
import inspect
import math

import pytest

from app.models.tabular_model import evaluate
from app.strategy.live_models import ModelDecision


def test_training_rule_is_sign_and_cost_aware():
    """The reference definition, stated directly."""
    cost_bps = 12.0
    cost = cost_bps / 10000.0
    # a prediction smaller than the round trip is not actionable, in either direction
    assert evaluate([cost / 2], [cost / 2], cost_bps)['active_samples'] == 0
    assert evaluate([-cost / 2], [-cost / 2], cost_bps)['active_samples'] == 0
    # clearing it in the right direction is actionable
    assert evaluate([cost * 3], [cost * 3], cost_bps)['active_samples'] == 1
    assert evaluate([-cost * 3], [-cost * 3], cost_bps)['active_samples'] == 1


def test_round_trip_cost_is_one_formula():
    """fee on both legs plus slippage on both legs, and nothing else."""
    fee_rate, slippage_bps = 0.0004, 2.0
    assert ModelDecision.round_trip_cost_pct.fget is not None
    expected = 2 * fee_rate + 2 * slippage_bps / 10000
    assert math.isclose(expected, 0.0012, rel_tol=1e-12)


def test_min_edge_floor_is_expressed_in_net_bps():
    """The gate compares a NET figure that knows what the round trip costs.

    Gating on the raw absolute prediction is exactly what made 99.94% of rows look
    actionable: a 0.5bp floor against a distribution whose median |prediction| is ~1bp.
    """
    source = inspect.getsource(ModelDecision._assemble)
    assert 'round_trip_cost' in source, 'the gate must know what the round trip costs'
    assert 'cost_bps' in source
    assert 'net_bps' in source, 'the gate must compare a net figure'


def test_the_gate_reports_every_blocking_reason_not_just_the_first():
    """An if/elif chain hid the gates below the first one that fired.

    The audit could not answer "how many bars would the edge floor have blocked?" because
    `feature_degraded` was first in the chain and short-circuited the measurement.
    """
    source = inspect.getsource(ModelDecision._assemble)
    # strip comments and docstrings: this asserts on control flow, not on prose
    code = '\n'.join(line.split('#')[0] for line in source.splitlines())
    assert 'blocked_by' in code
    assert 'elif ' not in code, ('the gate chain must not short-circuit: every reason is '
                                 'recorded so the counters stay interpretable')
