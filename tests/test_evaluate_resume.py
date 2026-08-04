import json
import shutil
import tempfile
import unittest
from pathlib import Path

import evaluate


class EvaluateResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.output = self.tmp_dir / "results.jsonl"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_rows(self, rows):
        with self.output.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_load_resume_state_requeues_empty_transient_failures(self):
        self._write_rows(
            [
                {"filename": "ok.jpg", "parse_failed": False, "prediction_text": "(1, 2)"},
                {"filename": "retry.jpg", "parse_failed": True, "prediction_text": ""},
            ]
        )

        existing, order, already_done, retryable_count, needs_compaction = evaluate.load_resume_state(str(self.output))

        self.assertEqual(order, ["ok.jpg", "retry.jpg"])
        self.assertEqual(set(existing), {"ok.jpg"})
        self.assertEqual(already_done, {"ok.jpg"})
        self.assertEqual(retryable_count, 1)
        self.assertTrue(needs_compaction)

    def test_load_resume_state_keeps_nonempty_parse_failures_final(self):
        self._write_rows(
            [
                {"filename": "bad.jpg", "parse_failed": True, "prediction_text": "no coordinates"},
            ]
        )

        existing, _, already_done, retryable_count, needs_compaction = evaluate.load_resume_state(str(self.output))

        self.assertEqual(set(existing), {"bad.jpg"})
        self.assertEqual(already_done, {"bad.jpg"})
        self.assertEqual(retryable_count, 0)
        self.assertFalse(needs_compaction)


if __name__ == "__main__":
    unittest.main()
