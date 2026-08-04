import unittest

from analysis.mcrs_validation import (
    BASE_R_WEIGHTS,
    compute_scores,
    leave_one_component_out,
    rank_stability,
)
from evaluation.mcrs import compute_mcrs


class MCRSValidationTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "model-a": {
                "wla_original": 0.8,
                "wla_blank": 0.7,
                "wla_similar": 0.7,
                "wla_random": 0.5,
                "wla_adversarial": 0.4,
                "tbs": 500.0,
                "tfr_adv": 0.1,
                "adv_near_tier": 0.3,
            },
            "model-b": {
                "wla_original": 0.6,
                "wla_blank": 0.5,
                "wla_similar": 0.5,
                "wla_random": 0.3,
                "wla_adversarial": 0.2,
                "tbs": 1500.0,
                "tfr_adv": 0.3,
                "adv_near_tier": 0.1,
            },
        }

    def test_compute_scores_matches_new_capability_robustness_formula(self):
        scores = compute_scores(self.metrics)

        self.assertGreater(scores["model-a"]["mcrs"], scores["model-b"]["mcrs"])
        self.assertAlmostEqual(scores["model-a"]["C"], 0.75)
        self.assertAlmostEqual(scores["model-a"]["R"], 0.6777380952, places=4)
        self.assertAlmostEqual(
            scores["model-a"]["mcrs"],
            100.0 * 0.75**0.4 * 0.6777380952**0.6,
            places=2,
        )
        self.assertAlmostEqual(sum(BASE_R_WEIGHTS.values()), 1.0)
        self.assertEqual(
            BASE_R_WEIGHTS,
            {
                "random_retention": 0.22,
                "adversarial_retention": 0.44,
                "tbs_quality": 0.17,
                "tfr_quality": 0.17,
            },
        )

    def test_production_mcrs_matches_validation_formula(self):
        production = compute_mcrs(self.metrics)
        validation = compute_scores(self.metrics)

        self.assertAlmostEqual(
            production["model-a"]["mcrs"],
            validation["model-a"]["mcrs"],
            places=2,
        )
        self.assertAlmostEqual(production["model-a"]["R"], validation["model-a"]["R"])

    def test_leave_one_out_renormalizes_additive_weights(self):
        variants = leave_one_component_out(self.metrics)
        no_random = variants["random_retention"]

        self.assertNotIn("random_retention", no_random["weights"])
        self.assertAlmostEqual(sum(no_random["weights"].values()), 1.0)
        self.assertEqual(set(no_random["scores"]), {"model-a", "model-b"})

    def test_rank_stability_is_one_for_identical_rankings(self):
        baseline = compute_scores(self.metrics)
        stability = rank_stability(baseline, baseline)

        self.assertAlmostEqual(stability["kendall_tau"], 1.0)
        self.assertAlmostEqual(stability["spearman_rho"], 1.0)


if __name__ == "__main__":
    unittest.main()
