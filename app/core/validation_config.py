"""Refuse to let a demonstration configuration produce a performance claim.

The audit of 2026-09-14 found a simulation whose every trade carried the reason code
\`edge_gate_disabled\`. The gate that would have blocked all six selected symbols -- each
with a negative realised net edge, averaging -11.13 bps -- was switched off on purpose, so
that the simulator would exercise the whole path instead of sitting idle. The .env file
said so, in a comment, next to the setting.

Nothing in the output said so. The run produced an equity curve and a set of trades that
look exactly like a real result, and it took reading the configuration to learn they were
a demo. That is the failure this module prevents: a small, cheap check that makes the
configuration travel with the number.
"""
import os

# Each entry is (setting, demo value, why it makes the result uninterpretable). The demo
# values are the ones the shipped .env actually uses, not hypothetical ones.
DEMO_FLAGS = (
    ("MODEL_MIN_EDGE_BPS", lambda v: float(v) < 0.5,
     "the model's edge floor is below the value that covers the round trip"),
    ("MODEL_MIN_AGREEMENT", lambda v: float(v) < 0.6,
     "model agreement is allowed near one third, which is close to chance"),
    ("MIN_EDGE_MULTIPLE", lambda v: float(v) < 2.5,
     "the target must barely clear cost, so coin-flip signals pass"),
    ("SYMBOL_EDGE_MIN_NET_BPS", lambda v: float(v) < 0.0,
     "a symbol is allowed to trade with a negative realised net edge"),
)


class DemoConfiguration(RuntimeError):
    """Raised when a performance claim is requested from a demo configuration."""


def demo_settings(environ=None):
    """The demo switches that are currently set, as (name, value, reason)."""
    environ = os.environ if environ is None else environ
    found = []
    for name, is_demo, reason in DEMO_FLAGS:
        raw = environ.get(name)
        if raw is None:
            continue
        try:
            if is_demo(raw):
                found.append((name, raw, reason))
        except (TypeError, ValueError):
            found.append((name, raw, "value is not numeric and cannot be checked"))
    # Booleans are checked separately because their demo value is a string, not a bound.
    for name, reason in (("SYMBOL_EDGE_ENABLED",
                          "the per-symbol evidence gate is switched off, so a symbol "
                          "with no positive record still trades"),
                         ("MODEL_OOD_POLICY", None)):
        raw = environ.get(name)
        if raw is None:
            continue
        if name == "SYMBOL_EDGE_ENABLED" and str(raw).strip().lower() in ("0", "false", "no"):
            found.append((name, raw, reason))
        if name == "MODEL_OOD_POLICY" and str(raw).strip().lower() == "warn":
            found.append((name, raw, "out-of-distribution inputs warn instead of blocking"))
    return found


def require_validation_config(environ=None):
    """Raise if any demo switch is on. Call this before reporting a performance number."""
    found = demo_settings(environ)
    if not found:
        return
    lines = ["this configuration is for demonstration, not for measuring performance:"]
    for name, value, reason in found:
        lines.append("  %s=%s  -- %s" % (name, value, reason))
    lines.append("")
    lines.append("Use .env.validation instead, or clear these settings. A net result "
                 "produced with them on says nothing about whether the strategy makes "
                 "money.")
    raise DemoConfiguration("\n".join(lines))


def main():
    import sys
    found = demo_settings()
    if not found:
        print("configuration is a validation configuration: no demo switches are set")
        return 0
    print("DEMO CONFIGURATION DETECTED")
    print("=" * 64)
    for name, value, reason in found:
        print("  %-26s = %-8s %s" % (name, value, reason))
    print()
    print("  A net result from this configuration cannot be used to judge whether the")
    print("  strategy makes money. Use .env.validation.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
