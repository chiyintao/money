"""Before hunting for a signal that passes the gate, prove the gate is passable.

A screen that rejects everything is not evidence of anything. If no configuration of the
known features clears the three bars, then either the features genuinely carry nothing
(possible) or the bars are miscalibrated (also possible), and the difference matters
before anyone spends a week on the next idea.

This runs a null and a plant through the same machinery. The null is a shuffled forward
return, which must fail. The plant is a deliberately constructed predictor, which must
pass. If the plant fails, the gate is broken and every rejection it has produced so far --
including the momentum one -- is uninterpretable.
"""
import math
import random
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe

rows, by_day = pe.load()
days = sorted(by_day)


def spread_with(field, clip=None, top=2, bottom=2, transform=None):
    out = []
    for day in days:
        rr = [x for x in by_day[day]
              if x.get("fwd_1d") is not None and math.isfinite(x["fwd_1d"])]
        if len(rr) < 8:
            continue
        values = [r.get(field) for r in rr]
        if any(v is None or not math.isfinite(v) for v in values):
            continue
        if transform:
            values = transform(values)
        scores = pe.rank(values)
        order = sorted(range(len(rr)), key=lambda i: scores[i])

        def value(i):
            v = rr[i]["fwd_1d"]
            return max(-clip, min(clip, v)) if clip else v

        out.append(sum(value(i) for i in order[-top:]) / top
                   - sum(value(i) for i in order[:bottom]) / bottom)
    return out


def gate(values, label):
    raw = pe.stats(values)
    clipped = pe.stats([max(-0.10, min(0.10, v)) for v in values])
    retained = clipped["mean"] / raw["mean"] if raw["mean"] else 0.0
    verdict = (raw["t"] >= 2.0 and clipped["t"] >= 2.0 and retained >= 0.5)
    print("  %-28s raw %+8.2f (t=%5.2f)  clip %+8.2f (t=%5.2f)  ret %4.0f%%  %s"
          % (label, raw["mean_bps"], raw["t"], clipped["mean_bps"], clipped["t"],
             100 * retained, "PASS" if verdict else "fail"))
    return verdict


def main():
    print("IS THE GATE PASSABLE?")
    print("=" * 88)

    # The null: a random score must fail. Run several seeds so one lucky draw cannot be
    # mistaken for a working gate.
    rng = random.Random(20260914)
    passes = 0
    for trial in range(5):
        seed = rng.randrange(10 ** 6)
        def shuffled(values, seed=seed):
            local = random.Random(seed)
            copy = list(values)
            local.shuffle(copy)
            return copy
        values = spread_with("ret_7d", transform=shuffled)
        if gate(values, "null (shuffled rank #%d)" % trial):
            passes += 1
    print("  null passed %d of 5 -- must be 0" % passes)
    print()

    # The plant: rank on the NEXT day's return, one day stale. This is a real if
    # impossible predictor, because it uses information the live system would not have, so
    # it must sail through. If it does not, the gate rejects everything and is useless.
    planted = spread_with("fwd_1d")
    gate(planted, "plant (1-day stale oracle)")

    print()
    print("  A gate that fails the plant is broken; one that passes the null is broken.")
    print("  Only a gate that does both correctly can be used to reject a real signal.")


if __name__ == "__main__":
    main()
