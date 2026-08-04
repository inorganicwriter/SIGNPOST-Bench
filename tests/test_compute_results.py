import json
import shutil
import tempfile
import unittest
from pathlib import Path

from analysis import compute_results


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class ComputeResultsTests(unittest.TestCase):
    def test_analyze_results_recomputes_tbs_from_blank_baseline(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            results_dir = tmp_dir / "results"
            model_dir = results_dir / "model-a"
            _write_jsonl(
                model_dir / "results_Blank_model-a.jsonl",
                [
                    {"filename": "img1_Blank.jpg", "error_km": 100.0},
                    {"filename": "img2_Blank.jpg", "error_km": 200.0},
                ],
            )
            _write_jsonl(
                model_dir / "results_Random_model-a.jsonl",
                [
                    {"filename": "img1_random_1.jpg", "error_km": 130.0, "tbs": 9999.0},
                    {"filename": "img2_random_1.jpg", "error_km": 260.0, "tbs": 9999.0},
                ],
            )

            results = compute_results.analyze_results(results_dir, dataset_name="tmp")

            self.assertEqual(results["model-a"]["Random"]["TBS"], 45.0)
            self.assertEqual(results["model-a"]["Random"]["TBSPairs"], 2)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
