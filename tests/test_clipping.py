"""Tests for the clipping construction, which had a bug that flattered every candidate.

The first version of the sweep clipped the DAILY SPREAD rather than each trade's forward
return. That is a much weaker test -- a single extreme coin can then move its day by at
most 10% of the spread, so large coins on ordinary days survive -- and it reported 82%
retention for a signal whose true figure is 50%. A bug that makes signals look more robust
than they are is the most expensive kind here, so the construction is pinned by tests.
"""
import math
import sys
import unittest

sys.path.insert(0, ".")

from scripts import panel_expectation as pe
from scripts.feature_sweep import spread


class TestClipConstruction(unittest.TestCase):
    def test_clipping_each_trade_is_stricter_than_clipping_the_spread(self):
        # The measured values on the shipped panel. If the construction regresses to
        # clipping the spread, the retained share jumps back to ~82% and this fails.
        raw = pe.stats(spread("ret_7d", sign=1))
        per_trade = pe.stats(spread("ret_7d", sign=1, clip=0.10))
        retained = per_trade["mean"] / raw["mean"]
        self.assertLess(retained, 0.60, "retention is too high: is the clip per trade?")
        self.assertGreater(retained, 0.40)

    def test_clipping_never_increases_the_spread(self):
        # Clipping can only pull extreme outcomes toward zero, so the mean magnitude of a
        # clipped spread cannot exceed the raw one. A violation means the clip is not
        # being applied to the outcome at all.
        for field in ("ret_7d", "close_pos", "ret_14d"):
            raw = pe.stats(spread(field, sign=1))
            clipped = pe.stats(spread(field, sign=1, clip=0.10))
            self.assertLessEqual(abs(clipped["mean"]), abs(raw["mean"]) + 1e-9, field)

    def test_tighter_clip_removes_more(self):
        # A tighter bound must never leave more of the spread than a looser one.
        loose = pe.stats(spread("ret_7d", sign=1, clip=0.15))["mean"]
        tight = pe.stats(spread("ret_7d", sign=1, clip=0.03))["mean"]
        self.assertLess(abs(tight), abs(loose))

    def test_exclude_removes_the_coin_from_the_universe(self):
        # ZEC is the largest contributor to the momentum spread, so dropping it must
        # reduce the raw figure. This also checks that exclude is wired through at all --
        # an ignored argument would leave the two identical.
        with_zec = pe.stats(spread("ret_7d", sign=1))["mean"]
        without = pe.stats(spread("ret_7d", sign=1, exclude=("ZECUSDT",)))["mean"]
        self.assertLess(without, with_zec)

    def test_a_constant_column_is_skipped_not_scored(self):
        # A column with no cross-sectional variation cannot rank anything, so the sweep
        # returns None rather than reporting a meaningless zero. mkt_ret_1d is the same
        # value for every coin on a given day, which is exactly that case -- using a column
        # that merely happens to repeat (like "index", a per-day counter) would not test
        # the guard at all.
        self.assertIsNone(spread("mkt_ret_1d", sign=1))

    def test_a_column_that_varies_is_not_skipped(self):
        # The other side of the guard: a real feature must not be dropped.
        self.assertIsNotNone(spread("close_pos", sign=1))


class TestTailRetentionMonotonicity(unittest.TestCase):
    def test_retention_decreases_as_the_clip_tightens(self):
        # The whole diagnostic rests on this shape: if the signal's profit lives in the
        # tails, closing the tails must remove progressively more of it.
        raw = pe.stats(spread("ret_7d", sign=1))["mean"]
        previous = 1.0
        for clip in (0.15, 0.10, 0.07, 0.05, 0.03):
            share = pe.stats(spread("ret_7d", sign=1, clip=clip))["mean"] / raw
            self.assertLess(share, previous + 1e-9, "clip %.2f" % clip)
            previous = share


if __name__ == "__main__":
    unittest.main(verbosity=2)
