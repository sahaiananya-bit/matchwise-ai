"""Regression coverage for evidence, bounded scoring and truly local fallback."""

import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

from config import MAX_TEXT_CHARS
from src.matcher import ResumeJDMatcher
from src.skills import extract_skill_evidence


class MatcherTests(unittest.TestCase):
    def setUp(self):
        self.matcher = ResumeJDMatcher()

    def test_aliases_and_symbol_boundaries(self):
        skills = self.matcher.extract_skills(
            "Skills: C++, C#, sklearn, k8s, PostgreSQL, Amazon Web Services, "
            "natural language processing, retrieval-augmented generation, large language models."
        )
        self.assertTrue({"C++", "C#", "scikit-learn", "Kubernetes", "PostgreSQL", "AWS", "NLP", "RAG", "LLMs"} <= skills)

    def test_substrings_are_not_skills(self):
        self.assertEqual(self.matcher.extract_skills("JavaScript and rustproofing; pythons provoke a reaction."), {"JavaScript"})
        self.assertNotIn("C++", self.matcher.extract_skills("AC++ syntax and C+++ are unrelated tokens."))

    def test_ambiguous_r_and_go(self):
        self.assertFalse({"Go", "R"} & self.matcher.extract_skills("Go to the office. Work in R&D. Every r matters."))
        self.assertTrue({"Go", "R"} <= self.matcher.extract_skills("Programming languages: Go, R and Python."))
        self.assertIn("Go", self.matcher.extract_skills("Golang backend development."))

    def test_negated_and_future_claims_do_not_become_skills(self):
        result = self.matcher.analyze(
            "No experience with Docker and Kubernetes. I plan to learn Python. "
            "Currently learning TensorFlow. I built APIs in FastAPI.",
            "Required: Docker, Kubernetes, Python, TensorFlow, FastAPI."
        )
        self.assertEqual(result["overlapping_skills"], ["FastAPI"])
        self.assertTrue({"Docker", "Kubernetes", "Python", "TensorFlow"} <= set(result["missing_skills"]))

    def test_negation_does_not_cross_sentences_or_but(self):
        self.assertEqual(self.matcher.extract_skills("No Docker, but Python is my main language. SQL projects."), {"Python", "SQL"})

    def test_machine_learning_does_not_negate_implementation_skills(self):
        self.assertTrue({"Machine learning", "Python", "PyTorch", "PostgreSQL"} <= self.matcher.extract_skills(
            "Built machine learning models with Python and PyTorch; stored outcomes in PostgreSQL."
        ))
        self.assertIn("TensorFlow", self.matcher.extract_skills("Deep learning using TensorFlow."))

    def test_jd_explicit_non_requirements_are_excluded(self):
        evidence = extract_skill_evidence("No Docker experience required. Python is required. SQL not required.", True)
        self.assertEqual(set(evidence), {"Python"})

    def test_negation_suffix_can_apply_to_entire_skill_list(self):
        self.assertEqual(extract_skill_evidence("Python, SQL and Docker are not required.", True), {})
        self.assertEqual(self.matcher.extract_skills("Python and Docker are not my skills."), set())
        self.assertEqual(set(extract_skill_evidence("Python required, Docker not required.", True)), {"Python"})

    def test_affirmative_not_only_and_no_difficulty(self):
        self.assertEqual(self.matcher.extract_skills("Not only Python and SQL."), {"Python", "SQL"})
        self.assertEqual(self.matcher.extract_skills("I have no difficulty with Python and SQL."), {"Python", "SQL"})
        self.assertEqual(self.matcher.extract_skills("I don't know Docker or Kubernetes."), set())

    def test_generic_model_evaluation_does_not_claim_llm_experience(self):
        skills = self.matcher.extract_skills("Worked on machine learning model evaluation.")
        self.assertIn("Model evaluation", skills)
        self.assertNotIn("LLM evaluation", skills)

    def test_required_and_preferred_headings_and_weighting(self):
        result = self.matcher.analyze("Python projects.", "Required skills:\n- Python\nNice to have:\n- Docker")
        items = {item["skill"]: item for item in result["skill_evidence"]}
        self.assertEqual(items["Python"]["priority"], "required")
        self.assertEqual(items["Docker"]["priority"], "preferred")
        self.assertAlmostEqual(result["skill_match_percentage"], 66.67, places=2)
        self.assertEqual(result["required_skill_count"], 1)
        self.assertEqual(result["preferred_skill_count"], 1)

    def test_inline_required_preferred_and_required_wins_repetition(self):
        items = extract_skill_evidence("Python required, Docker preferred.\nRequired: Docker.", True)
        self.assertEqual(items["Python"]["priority"], "required")
        self.assertEqual(items["Docker"]["priority"], "required")

    def test_evidence_contains_skill_and_original_wording(self):
        result = self.matcher.analyze("Project: Deployed models with PyTorch.", "Need PyTorch and Docker.")
        items = {item["skill"]: item for item in result["skill_evidence"]}
        self.assertEqual(items["PyTorch"]["resume_evidence"], "Project: Deployed models with PyTorch.")
        self.assertIn("Docker", items["Docker"]["jd_evidence"])
        self.assertEqual(items["Docker"]["resume_evidence"], "")

    def test_self_reported_skills_never_inflate_score(self):
        original = self.matcher.analyze("Python project.", "Python and Docker required.")
        with_profile = self.matcher.analyze("Python project.", "Python and Docker required.", extra_skills="Docker")
        self.assertEqual(original["overall_match_percentage"], with_profile["overall_match_percentage"])
        self.assertEqual(original["overlapping_skills"], with_profile["overlapping_skills"])
        self.assertEqual(with_profile["self_reported_skills"], ["Docker"])
        self.assertIn("Document the skills you reported", [item["title"] for item in with_profile["recommendations"]])

    def test_unsupported_jd_has_no_fabricated_skill_score(self):
        result = self.matcher.analyze("Python projects.", "Friendly professional wanted for an unusual opportunity.")
        self.assertEqual(result["jd_skills"], [])
        self.assertIsNone(result["skill_match_percentage"])
        self.assertEqual(result["scoring"]["content_weight"], 1.0)
        self.assertEqual(result["scoring"]["skill_weight"], 0.0)
        self.assertEqual(result["overall_match_percentage"], result["content_similarity"])

    def test_blank_punctuation_and_non_text_inputs_rejected(self):
        for invalid in ["", " \n ", "... ----", None, 123]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.matcher.analyze(invalid, "Python")
                with self.assertRaises(ValueError):
                    self.matcher.analyze("Python", invalid)

    def test_both_documents_have_character_limits(self):
        oversized = "x" * (MAX_TEXT_CHARS + 1)
        with self.assertRaisesRegex(ValueError, "50,000"):
            self.matcher.analyze(oversized, "Python")
        with self.assertRaisesRegex(ValueError, "50,000"):
            self.matcher.analyze("Python", oversized)

    def test_profile_input_validation(self):
        for invalid in [123, {"Python": True}, [123], "x" * 5001]:
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(ValueError):
                    self.matcher.analyze("Python", "Python", extra_skills=invalid)

    def test_identical_text_and_json_serialization(self):
        text = "Built Python services. Deployed Docker containers."
        result = self.matcher.analyze(text, text)
        self.assertEqual(result["overall_match_percentage"], 100.0)
        self.assertEqual(result["content_similarity"], 100.0)
        self.assertEqual(result["skill_match_percentage"], 100.0)
        self.assertIsNone(result["semantic_similarity"])
        self.assertEqual(result["similarity_method"], "lexical_tfidf")
        json.dumps(result, allow_nan=False)

    def test_best_sentence_alignment_is_not_all_pairs_average(self):
        result = self.matcher.analyze("Built Python services. Designed garden sculptures.", "Built Python services.")
        self.assertEqual(result["content_similarity"], 100.0)

    def test_no_overlap_and_stopword_only_scores_are_finite(self):
        for resume, jd in [("Python", "Docker"), ("the and a", "an and the")]:
            result = self.matcher.analyze(resume, jd)
            self.assertEqual(result["content_similarity"], 0.0)
            self.assertTrue(0 <= result["overall_match_percentage"] <= 100)
            json.dumps(result, allow_nan=False)

    def test_default_does_not_attempt_model_import(self):
        with patch.object(self.matcher, "_load_semantic_model", side_effect=AssertionError("Must stay local")):
            self.matcher.analyze("Python", "Python")

    def test_semantic_unavailable_falls_back_without_download(self):
        factory = Mock(side_effect=OSError("Model not cached"))
        fake_module = types.SimpleNamespace(SentenceTransformer=factory)
        matcher = ResumeJDMatcher(use_semantic=True)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            result = matcher.analyze("Python", "Python")
            matcher.analyze("Python", "Python")
        factory.assert_called_once_with("all-MiniLM-L6-v2", local_files_only=True, trust_remote_code=False, device="cpu")
        self.assertEqual(result["similarity_method"], "lexical_tfidf")
        self.assertIsNone(result["semantic_similarity"])
        self.assertTrue(any("unavailable locally" in warning for warning in result["warnings"]))

    def test_optional_semantic_uses_best_sentence_match(self):
        model = Mock()
        model.encode.return_value = [[1, 0], [0, 1], [1, 0]]
        matcher = ResumeJDMatcher(use_semantic=True)
        with patch.object(matcher, "_load_semantic_model", return_value=model):
            result = matcher.analyze("Python services. Garden sculpture.", "Python services.")
        self.assertEqual(result["semantic_similarity"], 100.0)
        self.assertEqual(result["similarity_method"], "semantic")

    def test_invalid_embeddings_use_transparent_fallback(self):
        model = Mock()
        model.encode.return_value = [[float("nan")], [1]]
        matcher = ResumeJDMatcher(use_semantic=True)
        with patch.object(matcher, "_load_semantic_model", return_value=model):
            result = matcher.analyze("Python", "Python")
        self.assertEqual(result["similarity_method"], "lexical_tfidf")
        self.assertTrue(any("could not finish" in warning for warning in result["warnings"]))

    def test_legacy_pdf_entrypoint_delegates_extractor(self):
        with patch("src.documents.extract_text", return_value="Python projects.") as extract:
            result = self.matcher.match_resume_to_jd("sample.pdf", "Python required.")
        extract.assert_called_once_with("sample.pdf")
        self.assertEqual(result["overlapping_skills"], ["Python"])


if __name__ == "__main__":
    unittest.main()
