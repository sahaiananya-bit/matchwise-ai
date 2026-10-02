"""Matchwise: an explainable resume workspace for AI engineering candidates."""

from copy import deepcopy
from hashlib import sha256
from html import escape
import json
import logging
from pathlib import Path

import streamlit as st

from config import MAX_SAVED_COMPARISONS, MAX_TEXT_CHARS, SUPPORTED_FILE_TYPES
from src.documents import InputError, extract_text
from src.matcher import ResumeJDMatcher
from src.reports import build_comparison_csv, build_json_report, build_markdown_report

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Matchwise · AI career workspace", page_icon="◈", layout="wide")

st.markdown("""
<style>
.block-container {max-width:1280px; padding-top:2rem; padding-bottom:3rem;}
h1,h2,h3 {letter-spacing:-.04em;}
h1 {font-weight:750 !important;}
[data-testid="stSidebar"] {border-right:1px solid #E3E8ED;}
[data-testid="stMetric"] {background:white; border:1px solid #E3E8ED;
  border-radius:14px; padding:18px 20px; min-height:120px;}
[data-testid="stMetricLabel"] {color:#657286;}
[data-testid="stMetricValue"] {font-size:2rem; letter-spacing:-.05em;}
.brand {font-size:23px; font-weight:800; color:#182438; letter-spacing:-1px; margin-bottom:4px;}
.brand span {color:#147D72;}
.eyebrow {text-transform:uppercase; font-size:11px; letter-spacing:2px; font-weight:700; color:#147D72;}
.hero {background:#15273B; padding:30px 34px; border-radius:18px; margin:10px 0 25px; color:#FFF;}
.hero h1 {color:#FFF; font-size:2.5rem; padding:4px 0 12px; line-height:1.15;}
.hero p {color:#CAD5E2; max-width:720px; margin-bottom:5px; font-size:16px;}
.hero .eyebrow {color:#88D8C7;}
.hero-foot {margin-top:24px; font-size:12px; color:#B8CCD7; letter-spacing:.4px;}
.pill {display:inline-block; font-size:12px; border-radius:7px; padding:5px 10px; margin:0 5px 7px 0;
  background:#E7F4EE; color:#17634F; border:1px solid #C6E4D7;}
.pill.gap {background:#FFF2E6; color:#925018; border-color:#F1D7BD;}
.empty {padding:32px; border:1px dashed #CBD5E1; border-radius:14px; text-align:center; color:#657286;}
.step {font-size:11px; font-weight:700; color:#147D72; letter-spacing:1.5px; margin-top:8px;}
.footer {font-size:12px; color:#778397; border-top:1px solid #E3E8ED; padding-top:18px; margin-top:30px;}
button {border-radius:9px !important;}
@media(max-width:700px) {.hero{padding:22px;} .hero h1{font-size:2rem;}}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {"resume_text": "", "jd_text": "", "job_title": "", "profile_skills": "",
                "source": "Paste text", "saved": [], "result": None, "analysis_signature": "",
                "upload_generation": 0, "demo_loaded": False}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = st.session_state.get(f"draft_{key}", value)


def remember_draft(key):
    # Conditional widgets are removed by Streamlit after navigation. Keep the
    # user's draft in independent session keys, not in a shared application cache.
    st.session_state[f"draft_{key}"] = st.session_state[key]


def clear_extracted_upload():
    for key in ("extracted_upload_id", "extracted_upload_text", "extracted_upload_name"):
        st.session_state.pop(key, None)


def remove_upload():
    clear_extracted_upload()
    st.session_state.upload_generation += 1


def load_demo():
    st.session_state.resume_text = (ROOT / "data/sample_resume.txt").read_text(encoding="utf-8")
    st.session_state.jd_text = (ROOT / "data/sample_job.txt").read_text(encoding="utf-8")
    st.session_state.job_title = "AI Engineer · Document Intelligence"
    st.session_state.source = "Paste text"
    st.session_state.workspace = "Resume analysis"
    st.session_state.profile_skills = ""
    st.session_state.result = None
    st.session_state.demo_loaded = True
    for key in ("resume_text", "jd_text", "job_title", "source"):
        remember_draft(key)


def clear_session():
    generation = st.session_state.get("upload_generation", 0) + 1
    for key in list(st.session_state):
        del st.session_state[key]
    st.session_state.upload_generation = generation


@st.cache_resource(show_spinner=False)
def get_matcher(use_semantic):
    """Cache the current engine's model resources, never document results."""
    return ResumeJDMatcher(use_semantic=use_semantic)


def digest(*values):
    return sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def pills(skills, missing=False):
    if not skills:
        st.caption("No skills in this group.")
        return
    css = "pill gap" if missing else "pill"
    st.markdown("".join(f'<span class="{css}">{escape(str(skill))}</span>' for skill in skills),
                unsafe_allow_html=True)


def pct(value):
    return "N/A" if value is None else f"{value:.0f}%"


def method_label(value):
    return "Local semantic model" if value == "semantic" else "Text similarity (TF-IDF)"


init_state()

with st.sidebar:
    st.markdown('<div class="brand"><span>◈</span> matchwise</div>', unsafe_allow_html=True)
    st.caption("YOUR NEXT ROLE, WITH EVIDENCE")
    st.divider()
    page = st.radio("Workspace", ["Resume analysis", "Compare roles"], key="workspace", label_visibility="collapsed")
    st.divider()
    st.markdown("**Career focus**")
    role = st.selectbox("Target role", ["AI Engineer", "Machine Learning Engineer", "Data Scientist", "Software Engineer"])
    st.text_area("Skills you can demonstrate", key="profile_skills", height=100,
                 placeholder="e.g. Python, PyTorch, RAG, FastAPI", max_chars=1500,
                 help="Optional, comma-separated. These guide recommendations; they do not boost your resume score.")
    use_semantic = st.toggle("Use local embedding model", value=False,
                             help="Uses a model already installed on this machine. If unavailable, the app clearly labels its lexical fallback.")
    st.caption("Target role labels your report. The job description determines what is assessed.")
    st.divider()
    st.button("Try the AI engineer demo", on_click=load_demo, width="stretch")
    st.button("Clear session data", on_click=clear_session, width="stretch")
    st.caption("Session only · No external AI API calls\n\nUploaded files are processed in memory. Downloads contain resume excerpts.")

st.markdown('<div class="eyebrow">CAREER INTELLIGENCE / AI ENGINEERING</div>', unsafe_allow_html=True)

if page == "Compare roles":
    st.title("Choose where to focus.")
    st.write("Compare saved analyses from this session. Save a result after analyzing a job description.")
    saved = st.session_state.saved
    if not saved:
        st.info("Your shortlist is empty. Open Resume analysis, analyze a role, then select Save to comparison.")
    else:
        if len({item["resume_id"] for item in saved}) > 1:
            st.warning("These results use different resume versions. Compare the same resume across roles for a fairer view.")
        if len({item["result"]["similarity_method"] for item in saved}) > 1:
            st.warning("These analyses use different content methods. Their scores are not directly comparable.")
        if len({(item["result"]["scoring"]["content_weight"], item["result"]["scoring"]["skill_weight"]) for item in saved}) > 1:
            st.warning("These analyses use different scoring weights because some jobs have no recognized skills. Compare their evidence rather than their overall scores.")
        rows = [{"job_title": item["job_title"], "target_role": item["role"],
                 "overall_match_percentage": item["result"]["overall_match_percentage"],
                 "skill_match_percentage": item["result"]["skill_match_percentage"],
                 "content_similarity": item["result"]["content_similarity"],
                 "missing_skills": ", ".join(item["result"]["missing_skills"]),
                 "similarity_method": method_label(item["result"]["similarity_method"]),
                 "resume_version": item["resume_id"][:8]} for item in saved]
        rows.sort(key=lambda row: row["overall_match_percentage"], reverse=True)
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"job_title": "Role", "target_role": "Career focus",
                                    "overall_match_percentage": st.column_config.ProgressColumn("Match", min_value=0, max_value=100, format="%.1f%%"),
                                    "skill_match_percentage": "Skill coverage (%)", "content_similarity": "Content (%)",
                                    "missing_skills": "Not evidenced", "similarity_method": "Method", "resume_version": "Resume version"})
        st.download_button("Download comparison CSV", build_comparison_csv(rows), "matchwise-comparison.csv", "text/csv")
        for index, item in enumerate(saved):
            with st.expander(item["job_title"]):
                st.write(f"{item['role']} · {pct(item['result']['overall_match_percentage'])} match")
                pills(item["result"]["missing_skills"], missing=True)
                st.download_button("Download saved report", build_markdown_report(item["result"], item["role"], item["job_title"]),
                                   "matchwise-report.md", "text/markdown", key=f"saved_report_{index}")
                if st.button("Remove from comparison", key=f"remove_{index}"):
                    st.session_state.saved.pop(index)
                    st.rerun()
        st.caption("Scores are document-alignment indicators, not predictions of interviews or hiring outcomes.")
else:
    st.markdown("""<div class="hero"><div class="eyebrow">MAKE YOUR EXPERIENCE VISIBLE</div>
      <h1>Your next role.<br>A clearer path to it.</h1>
      <p>Find where your resume fits, see the evidence, and turn skill gaps into a focused plan.</p>
      <div class="hero-foot">01 &nbsp; ADD YOUR RESUME &nbsp;&nbsp; / &nbsp;&nbsp; 02 &nbsp; CHOOSE A ROLE &nbsp;&nbsp; / &nbsp;&nbsp; 03 &nbsp; FIND YOUR FOCUS</div>
      </div>""", unsafe_allow_html=True)
    if st.session_state.demo_loaded:
        st.info("Demo content is fictional. Replace it with your own resume and a real job description to personalize the analysis.")

    left, right = st.columns(2, gap="large")
    resume_text = ""
    input_error = ""
    with left:
        st.markdown('<div class="step">01 / YOUR EXPERIENCE</div>', unsafe_allow_html=True)
        st.subheader("Start with your resume")
        source = st.radio("Resume source", ["Paste text", "Upload a file"], key="source", horizontal=True,
                          on_change=remember_draft, args=("source",),
                          label_visibility="collapsed")
        if source == "Paste text":
            resume_text = st.text_area("Resume text", key="resume_text", height=290, max_chars=MAX_TEXT_CHARS,
                                       on_change=remember_draft, args=("resume_text",),
                                       placeholder="Paste your resume, including skills, projects, experience and education…")
        else:
            upload = st.file_uploader("PDF, DOCX or TXT · up to 5 MB", type=SUPPORTED_FILE_TYPES,
                                      key=f"upload_{st.session_state.upload_generation}", on_change=clear_extracted_upload)
            if upload is not None:
                try:
                    upload_bytes = upload.getvalue()
                    upload_id = sha256(upload.name.encode("utf-8") + upload_bytes).hexdigest()
                    if st.session_state.get("extracted_upload_id") != upload_id:
                        extracted = extract_text(upload_bytes, filename=upload.name)
                        st.session_state.extracted_upload_id = upload_id
                        st.session_state.extracted_upload_text = extracted
                        st.session_state.extracted_upload_name = upload.name
                    resume_text = st.session_state.extracted_upload_text
                except InputError as exc:
                    input_error = str(exc)
                    st.error(input_error)
            elif st.session_state.get("extracted_upload_text"):
                resume_text = st.session_state.extracted_upload_text
            if resume_text:
                st.success(f"Resume ready · {len(resume_text.split()):,} words extracted")
                st.caption(st.session_state.get("extracted_upload_name", "Uploaded resume"))
                with st.expander("Review extracted text"):
                    st.text(resume_text)
                st.button("Remove uploaded resume", on_click=remove_upload)
            st.caption("Text-based PDFs supported. For a scanned resume, paste its text. PDF limit: 30 pages.")
        st.caption("Only claims supported by your resume count toward the match score.")
    with right:
        st.markdown('<div class="step">02 / YOUR OPPORTUNITY</div>', unsafe_allow_html=True)
        st.subheader("Add the role you want")
        title = st.text_input("Job title / company", key="job_title", max_chars=120,
                              on_change=remember_draft, args=("job_title",),
                              placeholder="AI Engineer · Your target company")
        jd_text = st.text_area("Job description", key="jd_text", height=290, max_chars=MAX_TEXT_CHARS,
                               on_change=remember_draft, args=("jd_text",),
                               placeholder="Paste the full job description, including required and preferred skills…")

    profile = [skill.strip() for skill in st.session_state.profile_skills.split(",") if skill.strip()]
    signature = digest(resume_text, jd_text, title, profile, role, use_semantic)
    if st.button("Analyze my fit  →", type="primary", width="stretch"):
        st.session_state.result = None
        if input_error:
            st.error("Please fix the resume upload before analyzing.")
        elif not resume_text.strip() or not jd_text.strip():
            st.error("Add both your resume and the job description to continue.")
        else:
            try:
                with st.spinner("Finding skills, checking evidence and building your plan…"):
                    st.session_state.result = get_matcher(use_semantic).analyze(resume_text, jd_text, extra_skills=profile)
                st.session_state.analysis_signature = signature
            except ValueError as exc:
                st.error(str(exc))
            except Exception:
                # Never log document text or exception payloads, which may contain private input.
                logging.getLogger(__name__).error("Analysis failed unexpectedly; document content was not logged.")
                st.error("The analysis could not finish. Try the standard mode or a shorter, text-only resume.")

    result = st.session_state.result
    if result is not None and st.session_state.analysis_signature != signature:
        st.info("Your inputs changed. Analyze again to update the results and downloads.")
        result = None
    if result is None:
        st.markdown('<div class="empty">Your evidence, skill coverage and next steps will appear here.</div>', unsafe_allow_html=True)
    else:
        st.divider()
        st.markdown('<div class="step">03 / YOUR MATCH REPORT</div>', unsafe_allow_html=True)
        st.subheader(title or "Your opportunity, explained")
        st.caption(f"Career focus: {role} · Content method: {method_label(result['similarity_method'])}")
        cols = st.columns(4)
        cols[0].metric("Match score", pct(result["overall_match_percentage"]))
        cols[1].metric("Skill coverage", pct(result["skill_match_percentage"]))
        cols[2].metric("Content alignment", pct(result["content_similarity"]))
        cols[3].metric("Skills not evidenced", len(result["missing_skills"]))
        st.caption("A guide to document alignment, not an ATS score or hiring probability. A mention is evidence in the text, not proof of proficiency.")
        for warning in result.get("warnings", []):
            if not warning.startswith("Scores are heuristic"):
                st.warning(warning)

        overview, evidence, action, quality = st.tabs(["Skill map", "Evidence", "Action plan", "Resume checks"])
        with overview:
            a, b = st.columns(2, gap="large")
            with a:
                st.markdown(f"#### Evidenced in your resume · {len(result['overlapping_skills'])}")
                pills(result["overlapping_skills"])
            with b:
                st.markdown(f"#### Not yet evidenced · {len(result['missing_skills'])}")
                pills(result["missing_skills"], missing=True)
                st.caption("A gap means the skill was not detected in this resume; it does not mean you cannot do it.")
            with st.expander("All skills detected in your resume"):
                pills(result["resume_skills"])
            with st.expander("How the score works"):
                weights = result["scoring"]
                st.write(f"Content alignment contributes {weights['content_weight']:.0%}; skill coverage contributes {weights['skill_weight']:.0%}.")
                st.write("Skill coverage uses recognized job-description skills and their priority. Content alignment compares each job passage with the most relevant resume passage. Preferred requirements carry less weight.")
                st.write("The skill vocabulary and priority detection are rule-based and may miss context. Review the evidence before acting on a recommendation.")
        with evidence:
            st.write("Trace each recognized job skill to the text that supports it.")
            if result["skill_evidence"]:
                evidence_filter = st.radio("Show evidence", ["All skills", "Not evidenced", "Evidenced"], horizontal=True)
                for entry in result["skill_evidence"]:
                    if evidence_filter == "Not evidenced" and entry["status"] != "missing":
                        continue
                    if evidence_filter == "Evidenced" and entry["status"] != "matched":
                        continue
                    status = "Evidenced" if entry["status"] == "matched" else "Not evidenced"
                    with st.expander(f"{entry['skill']} · {entry['priority'].title()} · {status}"):
                        st.caption("JOB DESCRIPTION")
                        st.text(entry["jd_evidence"] or "No excerpt available.")
                        st.caption("YOUR RESUME")
                        st.text(entry["resume_evidence"] or "No supporting mention found.")
            else:
                st.info("No recognized job skills found. Use a more detailed job description to get a skill map.")
        with action:
            st.write("Prioritize relevant, truthful improvements to your application.")
            for index, recommendation in enumerate(result.get("recommendations", []), 1):
                with st.container(border=True):
                    st.markdown(f"**{index}. {escape(recommendation['title'])}**")
                    st.caption(str(recommendation["priority"]).upper())
                    st.write(recommendation["detail"])
            st.info("For AI engineering projects, explain the problem, your model or retrieval choices, how you evaluated them, and how you served the system. Include measured results only when you have them.")
        with quality:
            st.write("Simple checks on extracted text. These do not evaluate visual layout or ATS compatibility.")
            for check in result.get("resume_checks", []):
                with st.container(border=True):
                    st.markdown(f"**{check['label']}** · {check['status']}")
                    st.write(check["detail"])

        st.divider()
        st.markdown("**Take your next step**")
        a, b, c = st.columns(3)
        a.download_button("Download report (.md)", build_markdown_report(result, role, title),
                          "matchwise-report.md", "text/markdown", width="stretch")
        b.download_button("Download analysis (.json)", build_json_report(result, role, title),
                          "matchwise-analysis.json", "application/json", width="stretch")
        if c.button("Save to comparison", width="stretch"):
            if any(item["signature"] == signature for item in st.session_state.saved):
                st.info("This analysis is already in your comparison.")
            elif len(st.session_state.saved) >= MAX_SAVED_COMPARISONS:
                st.warning("Your comparison holds five roles. Remove one in Compare roles to add another.")
            else:
                st.session_state.saved.append({"signature": signature, "resume_id": digest(resume_text),
                                               "job_title": title or f"Untitled role {len(st.session_state.saved) + 1}",
                                               "role": role, "result": deepcopy(result)})
                st.success("Saved. Open Compare roles in the sidebar to see your shortlist.")
        st.caption("Reports include evidence excerpts and may contain personal information. Session data is lost when the session ends; download anything you want to keep.")

st.markdown('<div class="footer">◈ &nbsp; MATCHWISE &nbsp; / &nbsp; Built for thoughtful applications. Evidence first. Progress next.</div>', unsafe_allow_html=True)
