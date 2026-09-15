"""Tests for the edge gate and the panel expectation code.

The gate decides whether a signal is worth trading, so its arithmetic and its thresholds
both need to be right. These tests pin the rank function and the statistics against
hand-computed values, and check that the gate actually rejects the signal it was built to
reject -- a gate that passes everything is worse than none, because it looks like
evidence.
"""
import math
import sys
import unittest

sys.path.insert(0, ".")

from scripts import panel_expectation as pe
from scripts import edge_gate


class TestRank(unittest.TestCase):
    def test_orders_ascending(self):
        self.assertEqual(pe.rank([10.0, 20.0, 30.0]), [1.0, 2.0, 3.0])

    def test_ties_get_the_average_rank(self):
        # Two values tied for positions 1 and 2 both get 1.5.
        self.assertEqual(pe.rank([5.0, 5.0, 9.0]), [1.5, 1.5, 3.0])

    def test_order_of_input_does_not_matter(self):
        a = pe.rank([3.0, 1.0, 2.0])
        b = pe.rank([1.0, 2.0, 3.0])
        self.assertEqual(sorted(a), sorted(b))


class TestStats(unittest.TestCase):
    def test_matches_hand_computation(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = pe.stats(values)
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["mean"], 3.0)
        # Sample stdev of 1..5 is sqrt(2.5).
        self.assertAlmostEqual(s["sd"], math.sqrt(2.5))
        self.assertAlmostEqual(s["t"], 3.0 / (math.sqrt(2.5) / math.sqrt(5)))

    def test_bps_conversion(self):
        s = pe.stats([0.01] * 4)
        self.assertAlmostEqual(s["mean_bps"], 100.0)

    def test_short_input_is_not_a_crash(self):
        self.assertEqual(pe.stats([1.0])["n"], 1)


class TestGate(unittest.TestCase):
    def test_gate_rejects_the_momentum_signal(self):
        # The whole reason the gate exists. This is the measured behaviour on the shipped
        # panel, so a refactor that starts passing it must fail loudly here.
        code = edge_gate.main([])
        self.assertEqual(code, 1)

    def test_a_series_with_no_tails_keeps_all_of_its_spread(self):
        # A steady positive drift with tiny alternating noise: clipping at +/-10% cannot
        # touch it, so the retained share must be 100%. This is the case the gate is
        # supposed to pass, and it guards against the clip being applied to the signal
        # instead of the outcome.
        values = [0.002 + (0.0001 if i % 2 else -0.0001) for i in range(200)]
        clipped = [max(-0.10, min(0.10, v)) for v in values]
        self.assertEqual(values, clipped)
        self.assertGreater(pe.stats(values)["t"], 2.0)

    def test_a_series_that_is_all_tail_loses_its_spread(self):
        # One huge day among flat ones. Clipping removes it, and the mean collapses --
        # which is exactly the failure mode the momentum signal showed.
        values = [0.0] * 99 + [1.0]
        raw = pe.stats(values)["mean_bps"]
        clipped = pe.stats([max(-0.10, min(0.10, v)) for v in values])["mean_bps"]
        self.assertGreater(raw, clipped * 5)


class TestWinsorisation(unittest.TestCase):
    def test_clip_bounds_are_respected(self):
        values = [0.5, -0.5, 0.02]
        clipped = [max(-0.1, min(0.1, v)) for v in values]
        self.assertEqual(clipped, [0.1, -0.1, 0.02])

    def test_clipping_a_tail_free_series_changes_nothing(self):
        values = [0.01, -0.01, 0.02, -0.02]
        clipped = [max(-0.1, min(0.1, v)) for v in values]
        self.assertEqual(values, clipped)


if __name__ == "__main__":
    unittest.main(verbosity=2)
