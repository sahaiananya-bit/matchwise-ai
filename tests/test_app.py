"""User-flow checks for state, validation, comparison, and report availability."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def button(app, label):
    return next(widget for widget in app.button if widget.label == label)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def assert_healthy(self):
        self.assertEqual(len(self.app.exception), 0, [item.message for item in self.app.exception])

    def analyze_demo(self):
        button(self.app, "Try the AI engineer demo").click().run()
        button(self.app, "Analyze my fit  →").click().run()
        self.assert_healthy()
        self.assertEqual(len(self.app.metric), 4)

    def test_blank_inputs_are_actionable(self):
        self.assert_healthy()
        button(self.app, "Analyze my fit  →").click().run()
        self.assert_healthy()
        self.assertTrue(any("both your resume" in item.value for item in self.app.error))
        self.assertEqual(len(self.app.metric), 0)

    def test_evaluation_page_preserves_resume_draft(self):
        self.analyze_demo()
        resume = self.app.session_state["resume_text"]
        self.app.radio(key="workspace").set_value("Evaluation").run()
        self.assert_healthy()
        self.assertTrue(any("measured comparison" in item.value for item in self.app.markdown))
        self.app.radio(key="workspace").set_value("Resume analysis").run()
        self.assert_healthy()
        self.assertEqual(self.app.session_state["resume_text"], resume)
        self.assertEqual(len(self.app.metric), 4)

    def test_editing_input_hides_old_results_and_downloads(self):
        self.analyze_demo()
        self.assertGreaterEqual(len(self.app.get("download_button")), 2)
        self.app.text_area(key="jd_text").set_value("Required skills: Rust, Java and Kubernetes.").run()
        self.assert_healthy()
        self.assertEqual(len(self.app.metric), 0)
        self.assertEqual(len(self.app.get("download_button")), 0)
        self.assertTrue(any("inputs changed" in item.value for item in self.app.info))
        button(self.app, "Analyze my fit  →").click().run()
        self.assert_healthy()
        self.assertEqual(len(self.app.metric), 4)

    def test_save_compare_download_remove(self):
        self.analyze_demo()
        button(self.app, "Save to comparison").click().run()
        self.assertEqual(len(self.app.session_state["saved"]), 1)
        button(self.app, "Save to comparison").click().run()
        self.assertEqual(len(self.app.session_state["saved"]), 1)
        self.app.radio(key="workspace").set_value("Compare roles").run()
        self.assert_healthy()
        self.assertEqual(len(self.app.dataframe), 1)
        self.assertEqual(len(self.app.get("download_button")), 2)
        button(self.app, "Remove from comparison").click().run()
        self.assert_healthy()
        self.assertEqual(len(self.app.session_state["saved"]), 0)

    def test_clear_removes_inputs_results_and_comparisons(self):
        self.analyze_demo()
        button(self.app, "Save to comparison").click().run()
        button(self.app, "Clear session data").click().run()
        self.assert_healthy()
        self.assertEqual(self.app.text_area(key="resume_text").value, "")
        self.assertEqual(self.app.text_area(key="jd_text").value, "")
        self.assertEqual(self.app.session_state["saved"], [])
        self.assertIsNone(self.app.session_state["result"])

    def test_profile_skill_does_not_increase_score(self):
        self.analyze_demo()
        before = self.app.session_state["result"]["overall_match_percentage"]
        self.app.text_area(key="profile_skills").set_value("Kubernetes, MLflow, AWS").run()
        button(self.app, "Analyze my fit  →").click().run()
        self.assert_healthy()
        self.assertEqual(self.app.session_state["result"]["overall_match_percentage"], before)

    def test_no_detected_job_skills_is_not_zero_skill_coverage(self):
        self.app.text_area(key="resume_text").set_value("Built Python APIs and deployed a Docker application.").run()
        self.app.text_area(key="jd_text").set_value("We need someone who enjoys helping people succeed.").run()
        button(self.app, "Analyze my fit  →").click().run()
        self.assert_healthy()
        self.assertEqual(self.app.metric[1].value, "N/A")

    def test_navigation_preserves_draft_and_result(self):
        self.analyze_demo()
        resume = self.app.text_area(key="resume_text").value
        jd = self.app.text_area(key="jd_text").value
        self.app.radio(key="workspace").set_value("Compare roles").run()
        self.app.radio(key="workspace").set_value("Resume analysis").run()
        self.assert_healthy()
        self.assertEqual(self.app.text_area(key="resume_text").value, resume)
        self.assertEqual(self.app.text_area(key="jd_text").value, jd)
        self.assertEqual(len(self.app.metric), 4)

    def test_switching_input_source_preserves_pasted_resume(self):
        self.app.text_area(key="resume_text").set_value("My Python and Docker project.").run()
        self.app.radio(key="source").set_value("Upload a file").run()
        self.app.radio(key="source").set_value("Paste text").run()
        self.assert_healthy()
        self.assertEqual(self.app.text_area(key="resume_text").value, "My Python and Docker project.")

    def test_upload_draft_survives_navigation_and_can_be_removed(self):
        # AppTest does not set file uploads; simulate a successful parser result.
        self.app.radio(key="source").set_value("Upload a file").run()
        self.app.session_state["extracted_upload_text"] = "Built Python and Docker applications."
        self.app.session_state["extracted_upload_name"] = "resume.txt"
        self.app.text_area(key="jd_text").set_value("Required: Python and Docker.").run()
        button(self.app, "Analyze my fit  →").click().run()
        self.assertEqual(len(self.app.metric), 4)
        self.app.radio(key="workspace").set_value("Compare roles").run()
        self.app.radio(key="workspace").set_value("Resume analysis").run()
        self.assert_healthy()
        self.assertEqual(len(self.app.metric), 4)
        self.assertEqual(self.app.radio(key="source").value, "Upload a file")
        button(self.app, "Remove uploaded resume").click().run()
        self.assert_healthy()
        self.assertEqual(len(self.app.metric), 0)

    def test_compare_warns_when_scoring_weights_differ(self):
        self.analyze_demo()
        button(self.app, "Save to comparison").click().run()
        self.app.text_area(key="jd_text").set_value("We need someone who enjoys helping people succeed.").run()
        button(self.app, "Analyze my fit  →").click().run()
        button(self.app, "Save to comparison").click().run()
        self.app.radio(key="workspace").set_value("Compare roles").run()
        self.assert_healthy()
        self.assertTrue(any("different scoring weights" in item.value for item in self.app.warning))


if __name__ == "__main__":
    unittest.main()
