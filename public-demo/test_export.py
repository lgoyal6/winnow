"""Protect the demo's substantive benchmark regressions and scope."""
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("demo_export", Path(__file__).with_name("export.py"))
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


class RecordingTest(unittest.TestCase):
    def setUp(self):
        self.data = exporter.build()

    def test_committed_export_is_source_exact(self):
        self.assertEqual(self.data, json.loads(Path(__file__).with_name("site").joinpath("data.json").read_text()))

    def test_six_bit_proxy_success_does_not_hide_scored_failure(self):
        rows = {row["arm"]: row for row in self.data["quality"]
                if row["bucket"] == "all" and row["task"] == "all"}
        self.assertEqual(120, rows["tq6"]["n"])
        self.assertAlmostEqual(-25.02, rows["tq6"]["score"] - rows["fp16"]["score"], places=2)
        six = next(row for row in self.data["bitwidth"] if row["bits"] == 6)
        self.assertEqual(1, six["greedy_token_match_vs_fp16"])
        self.assertAlmostEqual(-0.60, rows["tq8"]["score"] - rows["fp16"]["score"], places=2)

    def test_resident_and_persistent_bytes_are_not_substituted(self):
        rows = {row["arm"]: row for row in self.data["memory"] if row["ctx"] == 16384}
        self.assertGreater(rows["tq4-current"]["resident_kv_mb"], rows["fp16"]["resident_kv_mb"])
        self.assertGreater(rows["tq4-packed"]["resident_kv_mb"], rows["tq4-packed"]["persistent_kv_mb"])
        self.assertLess(rows["tq4-packed"]["decode_tok_s"], rows["fp16"]["decode_tok_s"])

    def test_decode_keeps_every_losing_arm_and_model_scope(self):
        for rows in self.data["decode"].values():
            for arm in ("triton_tf32", "triton_fp32", "native_cuda_fp32"):
                self.assertGreater(rows[arm]["ms_per_token"], rows["fp16_baseline"]["ms_per_token"])
        self.assertGreater(self.data["decode"]["16384"]["native_cuda_fp32"]["ms_per_token"],
                           self.data["decode"]["16384"]["triton_tf32"]["ms_per_token"])
        self.assertNotIn("pred", json.dumps(self.data))


if __name__ == "__main__":
    unittest.main()
