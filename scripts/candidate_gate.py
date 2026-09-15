"""The three features that survive without ZECUSDT, put through every bar at once.

Ten of sixteen apparent findings on this panel were carried by one coin. That makes the
survivors worth a proper look rather than a dismissal, because they are the only candidates
that were not merely correlated with ZEC's rise.

`close_pos` is the interesting one: it has never been tested for tail retention, and it is
the only non-momentum feature to survive. It is the close's position within the day's
range -- a mean-reversion flavoured measure rather than a trend one, so it is not simply
the same signal wearing a different name.
"""
import json
import sys

sys.path.insert(0, ".")
from scripts import panel_expectation as pe
from scripts.feature_sweep import spread


def full_gate(field, sign, top=2, bottom=2):
    raw = pe.stats(spread(field, sign=sign, top=top, bottom=bottom))
    clip = pe.stats(spread(field, sign=sign, top=top, bottom=bottom, clip=0.10))
    no_zec = pe.stats(spread(field, sign=sign, top=top, bottom=bottom,
                             exclude=("ZECUSDT",)))
    no_zec_clip = pe.stats(spread(field, sign=sign, top=top, bottom=bottom, clip=0.10,
                                  exclude=("ZECUSDT",)))
    h = len(spread(field, sign=sign, top=top, bottom=bottom))
    halves = spread(field, sign=sign, top=top, bottom=bottom)
    cut = len(halves) // 2
    first = pe.stats(halves[:cut])
    second = pe.stats(halves[cut:])
    retained = clip["mean"] / raw["mean"] if raw["mean"] else 0.0
    checks = {
        "raw t>=2": raw["t"] >= 2.0,
        "clipped t>=2": clip["t"] >= 2.0,
        "retained>=50%": retained >= 0.50,
        "no-ZEC t>=2": no_zec["t"] >= 2.0,
        "no-ZEC clipped t>=2": no_zec_clip["t"] >= 2.0,
        "both halves >0": first["mean"] > 0 and second["mean"] > 0,
    }
    return {
        "field": field, "sign": sign,
        "raw_bps": raw["mean_bps"], "raw_t": raw["t"],
        "clip_bps": clip["mean_bps"], "clip_t": clip["t"], "retained": retained,
        "no_zec_bps": no_zec["mean_bps"], "no_zec_t": no_zec["t"],
        "no_zec_clip_bps": no_zec_clip["mean_bps"], "no_zec_clip_t": no_zec_clip["t"],
        "first_bps": first["mean_bps"], "second_bps": second["mean_bps"],
        "checks": checks, "passed": all(checks.values()),
    }


def main():
    print("FULL GATE ON THE THREE CANDIDATES THAT SURVIVE WITHOUT ZEC")
    print("=" * 92)
    out = []
    for field, sign in (("ret_1d", 1), ("ret_14d", 1), ("close_pos", 1)):
        r = full_gate(field, sign)
        out.append(r)
        print()
        print("  %s (direction %s)" % (field, "+" if sign > 0 else "-"))
        print("    raw                 %+8.2f bps  t=%5.2f" % (r["raw_bps"], r["raw_t"]))
        print("    clipped +/-10%%      %+8.2f bps  t=%5.2f   retained %.0f%%"
              % (r["clip_bps"], r["clip_t"], 100 * r["retained"]))
        print("    without ZEC         %+8.2f bps  t=%5.2f" % (r["no_zec_bps"], r["no_zec_t"]))
        print("    without ZEC clipped %+8.2f bps  t=%5.2f"
              % (r["no_zec_clip_bps"], r["no_zec_clip_t"]))
        print("    halves              %+8.2f / %+.2f bps" % (r["first_bps"], r["second_bps"]))
        for name, ok in r["checks"].items():
            print("      [%s] %s" % ("PASS" if ok else "FAIL", name))
        print("    VERDICT: %s" % ("PASSES" if r["passed"] else "rejected"))
    print()
    winners = [r for r in out if r["passed"]]
    print("  %d of %d pass every bar." % (len(winners), len(out)))
    with open("data/research_v4/candidate_gate.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
