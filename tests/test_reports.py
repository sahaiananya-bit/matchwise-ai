"""Report privacy, interpretation, formatting, and spreadsheet safety."""

import csv
import io
import json
import unittest

from src.reports import build_comparison_csv, build_json_report, build_markdown_report


def example_result():
    return {
        "schema_version": "2.0", "overall_match_percentage": 72.5,
        "semantic_similarity": None, "content_similarity": 60.0,
        "similarity_method": "TF-IDF", "skill_match_percentage": 80.0,
        "overlapping_skills": ["Python"], "missing_skills": ["MLflow"],
        "resume_skills": ["Python"], "jd_skills": ["Python", "MLflow"],
        "total_required_skills": 2, "matched_skills": 1,
        "skill_evidence": [{"skill": "Python", "status": "matched", "priority": "required",
                            "resume_evidence": "Built a Python API", "jd_evidence": "Python required"}],
        "recommendations": [{"priority": "high", "title": "Add deployment evidence", "detail": "Describe a deployed project if you have one."}],
        "warnings": ["Lexical fallback enabled"],
        "scoring": {"content_weight": 0.4, "skill_weight": 0.6},
        "resume_checks": [{"label": "Project outcomes", "status": "review", "detail": "Add measured outcomes when available."}],
        "resume_text": "PRIVATE FULL RESUME", "jd_text": "PRIVATE FULL JOB DESCRIPTION",
    }


class ReportTests(unittest.TestCase):
    def test_json_omits_raw_inputs_keeps_evidence_and_marks_privacy(self):
        report = json.loads(build_json_report(example_result(), job_title="AI Platform Engineer"))
        self.assertEqual(report["role"], "AI Engineer")
        self.assertEqual(report["job_title"], "AI Platform Engineer")
        self.assertEqual(report["analysis"]["skill_evidence"][0]["resume_evidence"], "Built a Python API")
        self.assertNotIn("resume_text", report["analysis"])
        self.assertNotIn("jd_text", report["analysis"])
        self.assertIn("excerpts", report["privacy_note"])
        self.assertIn("not hiring probability", report["interpretation_note"])
        self.assertNotIn("PRIVATE FULL", json.dumps(report))

    def test_json_drops_unknown_fields_and_nested_raw_inputs(self):
        result = example_result()
        result["unplanned_private_blob"] = "private"
        result["scoring"]["resume_text"] = "PRIVATE FULL RESUME"
        report = json.loads(build_json_report(result))
        self.assertNotIn("unplanned_private_blob", report["analysis"])
        self.assertNotIn("resume_text", report["analysis"]["scoring"])

    def test_json_preserves_self_reported_profile_skills(self):
        result = example_result()
        result["self_reported_skills"] = ["MLflow", "Docker"]
        report = json.loads(build_json_report(result))
        self.assertEqual(report["analysis"]["self_reported_skills"], ["MLflow", "Docker"])

    def test_report_does_not_mutate_result(self):
        result = example_result()
        before = json.dumps(result)
        build_json_report(result)
        build_markdown_report(result)
        self.assertEqual(json.dumps(result), before)

    def test_markdown_covers_explanation_and_action_without_full_resume(self):
        report = build_markdown_report(example_result(), job_title="AI Platform Engineer")
        for phrase in ("72.5%", "80.0%", "TF-IDF", "Built a Python API", "Add deployment evidence", "Project outcomes", "Lexical fallback", "excerpts", "AI Platform Engineer"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, report)
        self.assertNotIn("PRIVATE FULL RESUME", report)
        self.assertNotIn("PRIVATE FULL JOB DESCRIPTION", report)

    def test_markdown_escapes_table_breaks_links_and_html(self):
        result = example_result()
        result["skill_evidence"][0]["resume_evidence"] = "<script>alert(1)</script>|[click](https://evil.example)\nNext"
        report = build_markdown_report(result, role="# Pretend heading")
        self.assertNotIn("<script>", report)
        self.assertNotIn("[click]", report)
        self.assertIn("\\|", report)
        self.assertIn("\\# Pretend heading", report)
        self.assertIn("Next |", report)

    def test_missing_skills_not_assessed_is_not_reported_as_zero(self):
        report = build_markdown_report({"skill_match_percentage": None})
        self.assertIn("| Skill coverage | Not assessed |", report)
        self.assertNotIn("0%", report)
        self.assertIn("No recognized job skills", report)

    def test_csv_round_trips_unicode_commas_newlines_and_sparse_columns(self):
        rows = [{"job": "AI, engineer — café", "score": 85.5, "skills": ["Python", "SQL"]}, {"job": "Second\nline", "method": "TF-IDF"}]
        restored = list(csv.DictReader(io.StringIO(build_comparison_csv(rows))))
        self.assertEqual(restored[0]["job"], rows[0]["job"])
        self.assertEqual(restored[0]["score"], "85.5")
        self.assertEqual(restored[0]["skills"], "Python, SQL")
        self.assertEqual(restored[1]["job"], "Second\nline")
        self.assertEqual(restored[1]["score"], "")
        self.assertEqual(restored[1]["method"], "TF-IDF")

    def test_csv_prevents_formula_injection_after_spaces_and_controls(self):
        dangerous = ["=1+1", "+SUM(A1)", "-1+1", "@SUM(A1)", " \t=1+1", "\t=1+1", "\r=1+1", "\n=1+1", "\ufeff=1+1", "\x00=1+1", "\u2003=1+1"]
        text = build_comparison_csv([{"job": value} for value in dangerous])
        restored = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([row["job"] for row in restored], ["'" + value for value in dangerous])

    def test_csv_headers_also_escape_formulas(self):
        data = list(csv.reader(io.StringIO(build_comparison_csv([{ "=HEADER()": "safe"}]))))
        self.assertEqual(data[0], ["'=HEADER()"])

    def test_empty_comparison_returns_empty_text(self):
        self.assertEqual(build_comparison_csv([]), "")


if __name__ == "__main__":
    unittest.main()
