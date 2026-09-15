"""Is ZECUSDT distorting the panel, or is it simply the coin that moved most?

Every candidate in the sweep -- five momentum lookbacks and a flow feature -- named ZECUSDT
as its largest contributor. Six unrelated signals pointing at the same coin is not six
findings. It is one fact about the panel: ZEC went up 68x and everything correlated with
"went up" looks predictive.

The test is whether the panel's conclusions change when that one series is excluded
everywhere. If the answer is yes for every feature, then any future signal discovered on
this panel should be re-run without ZEC before it is believed.
"""
import sys
sys.path.insert(0, ".")
from scripts import panel_expectation as pe
from scripts.feature_sweep import spread, CANDIDATES

rows, by_day = pe.load()


def main():
    print("EVERY CANDIDATE, WITH AND WITHOUT ZECUSDT")
    print("=" * 92)
    print("  %-16s %-3s %12s %8s %12s %8s %s" % (
        "field", "dir", "with ZEC", "t", "without ZEC", "t", ""))
    strong_with, strong_without = [], []
    for field in CANDIDATES:
        for sign, label in ((1, "+"), (-1, "-")):
            a = spread(field, sign=sign)
            if not a:
                continue
            b = spread(field, sign=sign, exclude=("ZECUSDT",))
            sa, sb = pe.stats(a), pe.stats(b)
            if abs(sa["t"]) >= 2.0:
                strong_with.append((field, label, sa["t"]))
            if abs(sb["t"]) >= 2.0:
                strong_without.append((field, label, sb["t"]))
            flag = ""
            if abs(sa["t"]) >= 2.0 and abs(sb["t"]) < 2.0:
                flag = "  <-- collapses"
            print("  %-16s %-3s %12.2f %8.2f %12.2f %8.2f%s" % (
                field, label, sa["mean_bps"], sa["t"], sb["mean_bps"], sb["t"], flag))
    print()
    print("  candidates with |t| >= 2 including ZEC    : %d" % len(strong_with))
    print("  candidates with |t| >= 2 excluding ZEC    : %d" % len(strong_without))
    print()
    print("  the ones that survive without ZEC:")
    for field, label, t in strong_without:
        print("     %-16s %s  t=%.2f" % (field, label, t))
    print()
    if len(strong_without) == 0:
        print("  No feature reaches significance once one coin is removed. On this panel")
        print("  there is no evidence of a cross-sectional effect at all -- the appearance")
        print("  of one came from a single series that trended 68x.")


if __name__ == "__main__":
    main()
