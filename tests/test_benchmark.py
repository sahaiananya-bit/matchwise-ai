"""Check evaluation accounting without requiring model downloads in normal CI."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.compare_methods import evaluate, summarize


class BenchmarkTests(unittest.TestCase):
    def test_missing_semantic_model_stops_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        with patch("scripts.compare_methods.ResumeJDMatcher._load_semantic_model", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "No comparison was published"):
                evaluate(root / "benchmarks/ranking_cases.json", root / "models/missing")

    def test_ties_do_not_count_as_correct(self):
        rows = []
        for preferred, alternative in [(70, 20), (30, 30), (10, 80)]:
            rows.append({side: {"content_similarity": value, "overall_match_percentage": value}
                         for side, value in [("preferred", preferred), ("alternative", alternative)]})
        result = summarize(rows)["content_similarity"]
        self.assertEqual(result, {"correct": 1, "ties": 1, "total": 3, "strict_pairwise_accuracy": 0.3333})

    def test_fixture_has_distinct_pairs_and_label_rationales(self):
        root = Path(__file__).resolve().parents[1]
        cases = json.loads((root / "benchmarks/ranking_cases.json").read_text())["cases"]
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        for case in cases:
            self.assertNotEqual(case["preferred"], case["alternative"])
            self.assertTrue(case["reason"])
            self.assertTrue(case["job"])
