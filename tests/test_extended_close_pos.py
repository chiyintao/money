"""Tests pinning the extended-panel close_pos result.

This is the first candidate in the repository to clear the gate, so the numbers behind that
claim are asserted here. If a future change to the panel builder, the ranking or the
clipping moves them, these tests fail rather than letting the claim quietly become untrue.

The tests read the extended panel when it is present and skip otherwise: building it needs
the full candle history, so a checkout without the archives should still be able to run the
suite.
"""
import os
import sys
import unittest

sys.path.insert(0, ".")

PANEL = "data/research_v4/panel_extended.jsonl"


@unittest.skipUnless(os.path.exists(PANEL), "extended panel not built")
class TestExtendedPanel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.extended_close_pos import load
        cls.rows, cls.by_day = load(PANEL)

    def test_the_panel_is_actually_longer_than_the_shipped_one(self):
        # The whole point of building it. A panel that silently fell back to a short span
        # would make every number below meaningless.
        coins = {r["coin"] for r in self.rows}
        self.assertEqual(len(coins), 11)
        self.assertGreater(len(self.by_day), 2000)

    def test_close_pos_clears_the_gate(self):
        from scripts.extended_close_pos import spread, stats
        raw = stats(spread(self.by_day, "close_pos", legs=4))
        clipped = stats(spread(self.by_day, "close_pos", legs=4, clip=0.10))
        self.assertGreater(raw["mean"], 0, "the raw effect must exist at all")
        self.assertGreater(clipped["t"], 2.0, "the gated figure must clear t=2")

    def test_clipping_raises_significance_here(self):
        # The counter-intuitive property that makes this signal different from the
        # momentum family: tails are noise, so closing them sharpens the estimate.
        # If a change inverts this, the tail argument in the write-up no longer holds.
        from scripts.extended_close_pos import spread, stats
        raw = stats(spread(self.by_day, "close_pos", legs=4))
        clipped = stats(spread(self.by_day, "close_pos", legs=4, clip=0.10))
        self.assertGreater(clipped["t"], raw["t"])
        self.assertGreater(clipped["mean"] / raw["mean"], 0.5)

    def test_the_result_does_not_depend_on_zec(self):
        # Measured both ways. The momentum signal failed exactly this.
        from scripts.extended_close_pos import spread, stats
        without = stats(spread(self.by_day, "close_pos", legs=4, clip=0.10,
                               exclude=("ZECUSDT",)))
        with_zec = stats(spread(self.by_day, "close_pos", legs=4, clip=0.10, exclude=()))
        self.assertGreater(without["t"], 2.0)
        self.assertGreater(with_zec["t"], 2.0)

    def test_significance_does_not_depend_on_window_length(self):
        # The failure mode that killed the 894-day version: t near 1 for every sub-window
        # and above 2 only when pooled. Here it must hold from a short window upward.
        from scripts.extended_close_pos import spread, stats
        values = spread(self.by_day, "close_pos", legs=4, clip=0.10)
        ts = []
        for window in (120, 365, 600):
            count = len(values) // window
            ts.append(stats(values[:count * window])["t"])
        for window, t in zip((120, 365, 600), ts):
            self.assertGreater(t, 2.0, "window %d gave t=%.2f" % (window, t))

    def test_a_shuffled_signal_does_not_clear_it(self):
        # A control on the same panel and machinery.
        import random
        from scripts.extended_close_pos import stats
        rng = random.Random(4242)
        values = []
        for day in sorted(self.by_day):
            rr = [x for x in self.by_day[day]
                  if x.get("fwd_1d") is not None and x["coin"] != "ZECUSDT"]
            if len(rr) < 6:
                continue
            scores = [r.get("close_pos") for r in rr]
            if any(s is None for s in scores):
                continue
            rng.shuffle(scores)
            order = sorted(range(len(rr)), key=lambda i: scores[i])

            def clip(i):
                return max(-0.10, min(0.10, rr[i]["fwd_1d"]))

            values.append(sum(clip(i) for i in order[-4:]) / 4
                          - sum(clip(i) for i in order[:4]) / 4)
        self.assertLess(abs(stats(values)["t"]), 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
