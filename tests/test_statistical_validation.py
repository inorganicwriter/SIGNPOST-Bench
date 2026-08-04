import unittest

from analysis.statistical_validation import (
    holm_adjust,
    paired_wilcoxon,
    rank_biserial_from_differences,
)


class StatisticalValidationTests(unittest.TestCase):
    def test_holm_adjust_controls_ordered_family(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])

        self.assertEqual(adjusted, [0.03, 0.06, 0.06])

    def test_rank_biserial_is_positive_when_original_scores_are_higher(self):
        effect = rank_biserial_from_differences([0.4, 0.2, -0.1])

        self.assertAlmostEqual(effect, 2.0 / 3.0)

    def test_paired_wilcoxon_reports_one_sided_decrease(self):
        result = paired_wilcoxon(
            original_scores=[0.9, 0.8, 0.7, 0.6],
            adversarial_scores=[0.4, 0.3, 0.2, 0.1],
        )

        self.assertEqual(result["n"], 4)
        self.assertGreater(result["rank_biserial"], 0.99)
        self.assertLess(result["p_value"], 0.1)


if __name__ == "__main__":
    unittest.main()
