import sys
import unittest
from unittest.mock import patch

import evaluate


class EvaluateCliTests(unittest.TestCase):
    def test_main_preserves_explicit_path_arguments(self):
        argv = [
            "evaluate.py",
            "--img-dir",
            "images/Blank",
            "--metadata-file",
            "metadata/gt.tsv",
            "--output",
            "results/out.jsonl",
            "--model",
            "gemini-2.5-flash",
        ]

        with patch.object(sys, "argv", argv), patch.object(evaluate, "run_single") as run_single:
            evaluate.main()

        run_single.assert_called_once()
        args = run_single.call_args.args[0]
        self.assertIsNone(args.dataset)
        self.assertEqual(args.img_dir, "images/Blank")
        self.assertEqual(args.metadata_file, "metadata/gt.tsv")
        self.assertEqual(args.output, "results/out.jsonl")

    def test_docstring_does_not_reference_removed_vertex_aliases(self):
        self.assertNotIn("-vertex", evaluate.__doc__)


if __name__ == "__main__":
    unittest.main()
