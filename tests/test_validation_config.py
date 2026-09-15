"""Tests for the demo-configuration guard.

The guard's job is to stop a number produced under a demonstration configuration from
being read as evidence about the strategy. So the tests are about both directions: it must
fire on the values the shipped .env actually uses, and it must stay quiet on a real
validation configuration. A guard that fires on everything gets deleted.
"""
import sys
import unittest

sys.path.insert(0, ".")

from app.core import validation_config as vc


class TestDemoDetection(unittest.TestCase):
    def test_the_shipped_demo_values_are_all_detected(self):
        # Taken from the .env the repository actually ships.
        environ = {
            "MODEL_MIN_EDGE_BPS": "-1.5",
            "MODEL_MIN_AGREEMENT": "0.3",
            "MIN_EDGE_MULTIPLE": "0.9",
            "SYMBOL_EDGE_MIN_NET_BPS": "-999",
            "SYMBOL_EDGE_ENABLED": "0",
            "MODEL_OOD_POLICY": "warn",
        }
        found = {name for name, _, _ in vc.demo_settings(environ)}
        self.assertEqual(found, set(environ))

    def test_a_validation_configuration_is_silent(self):
        environ = {
            "MODEL_MIN_EDGE_BPS": "0.5",
            "MODEL_MIN_AGREEMENT": "0.6",
            "MIN_EDGE_MULTIPLE": "2.5",
            "SYMBOL_EDGE_MIN_NET_BPS": "0",
            "SYMBOL_EDGE_ENABLED": "1",
            "MODEL_OOD_POLICY": "block",
        }
        self.assertEqual(vc.demo_settings(environ), [])

    def test_empty_environment_is_silent(self):
        # Absent settings fall back to the documented defaults in config.py, which are
        # strict. Reporting them as demo would be wrong.
        self.assertEqual(vc.demo_settings({}), [])

    def test_symbol_gate_disabled_is_flagged(self):
        found = vc.demo_settings({"SYMBOL_EDGE_ENABLED": "0"})
        self.assertEqual(len(found), 1)
        self.assertIn("evidence gate", found[0][2])

    def test_symbol_gate_enabled_is_not_flagged(self):
        self.assertEqual(vc.demo_settings({"SYMBOL_EDGE_ENABLED": "1"}), [])

    def test_negative_net_bps_floor_is_flagged(self):
        # Any negative floor means a losing symbol can still trade. -999 is the shipped
        # value but the check must not depend on that specific number.
        for value in ("-1", "-999", "-0.5"):
            self.assertTrue(vc.demo_settings({"SYMBOL_EDGE_MIN_NET_BPS": value}), value)

    def test_non_numeric_value_is_flagged_not_ignored(self):
        # An unparseable setting cannot be shown to be safe, and silently passing it
        # would be the wrong direction to fail in.
        found = vc.demo_settings({"MODEL_MIN_EDGE_BPS": "abc"})
        self.assertEqual(len(found), 1)


class TestRequirement(unittest.TestCase):
    def test_raises_and_names_every_offending_setting(self):
        environ = {"SYMBOL_EDGE_ENABLED": "0", "MODEL_OOD_POLICY": "warn"}
        with self.assertRaises(vc.DemoConfiguration) as caught:
            vc.require_validation_config(environ)
        message = str(caught.exception)
        self.assertIn("SYMBOL_EDGE_ENABLED", message)
        self.assertIn("MODEL_OOD_POLICY", message)
        self.assertIn(".env.validation", message)

    def test_passes_silently_on_a_clean_configuration(self):
        vc.require_validation_config({"SYMBOL_EDGE_ENABLED": "1"})

    def test_the_exception_is_a_runtime_error(self):
        # Callers that already guard broad failure modes should still catch this.
        self.assertTrue(issubclass(vc.DemoConfiguration, RuntimeError))


if __name__ == "__main__":
    unittest.main(verbosity=2)
