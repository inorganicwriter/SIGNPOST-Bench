import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evaluation.mcrs import compute_dataset_level_metrics


class MCRSDatasetMetricsTests(unittest.TestCase):
    """Regression test: compute_dataset_level_metrics must run without ID exclusion."""

    def _write(self, model_dir, attack, rows):
        path = model_dir / f"results_{attack}_model-a.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_compute_dataset_level_metrics_no_exclusion_path(self):
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            base_dir = tmp_dir / "data"
            model_dir = base_dir / "im2gps3k" / "results" / "model-a"

            rows = {
                "Original": [{"filename": f"img{i}_orig.jpg", "error_km": float(i * 10)} for i in range(1, 6)],
                "Blank": [{"filename": f"img{i}_Blank.jpg", "error_km": float(i * 10)} for i in range(1, 6)],
                "Similar": [{"filename": f"img{i}_similar_x.jpg", "error_km": float(i * 10 + 5)} for i in range(1, 6)],
                "Random": [{"filename": f"img{i}_random_x.jpg", "error_km": float(i * 10 + 8)} for i in range(1, 6)],
                "Adversarial": [
                    {
                        "filename": f"img{i}_adversarial_x.jpg",
                        "error_km": float(i * 10 + 12),
                        "pred_lat": 40.0,
                        "pred_lon": -70.0,
                    }
                    for i in range(1, 6)
                ],
            }
            for attack, attack_rows in rows.items():
                self._write(model_dir, attack, attack_rows)

            analyzed_model = {
                attack: {"WLA": 50.0, "MedErr": 100.0, "TBS": 20.0, "TBSPairs": 5}
                for attack in ("Original", "Blank", "Similar", "Random", "Adversarial")
            }
            geocode_cache = {"some target": {"lat": 40.0, "lon": -70.0}}
            taxonomy_targets = {
                f"img{i}": "some target" for i in range(1, 6)
            }

            metrics = compute_dataset_level_metrics(
                "im2gps3k",
                "model-a",
                base_dir,
                geocode_cache=geocode_cache,
                taxonomy_targets=taxonomy_targets,
                analyzed_model=analyzed_model,
            )

            self.assertIsNotNone(metrics)
            self.assertAlmostEqual(metrics["wla_original"], 0.5)
            self.assertIn("tfr_adv", metrics)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
