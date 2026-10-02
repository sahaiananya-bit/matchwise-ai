"""Local evidence-based alignment, not candidate ability or hiring prediction.

Pure-Python TF-IDF requires no network or ML download. Optional embeddings use
an already cached SentenceTransformer only.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Set

from config import MAX_TEXT_CHARS, SEMANTIC_WEIGHT, SKILL_WEIGHT
from src.skills import extract_skill_evidence, parse_self_reported_skills


_STOP_WORDS = frozenset("""
a an and are as at be been being but by can could did do does for from had has
have he her his i if in into is it its may more most must of on or our she should
so such than that the their them then there these they this those to up us was
we were what when where which who will with would you your required requirements
preferred skills skill experience experienced using use used ability knowledge
""".split())


class ResumeJDMatcher:
    """Compare documented evidence with a JD using a transparent scoring model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_semantic: bool = False):
        self.model_name = model_name
        self.use_semantic = use_semantic
        self.model = None
        self._model_attempted = False
        self._model_warning = None

    @staticmethod
    def _validate_text(text: str, label: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label} must contain readable text.")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"{label} must be at most {MAX_TEXT_CHARS:,} characters.")
        if not re.search(r"[^\W_]", text, re.UNICODE):
            raise ValueError(f"{label} must contain words, not only punctuation.")
        return text.strip()

    @staticmethod
    def preprocess_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def extract_skills(text: str) -> Set[str]:
        return set(extract_skill_evidence(text))

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        parts = [part.strip(" \t-*•") for part in re.split(r"[\r\n]+|(?<=[.!?;])\s+", text)]
        parts = [part for part in parts if part]
        chunks = []
        for part in parts:
            words = part.split()
            chunks.extend(" ".join(words[index:index + 90]) for index in range(0, len(words), 90))
        # Bound comparison work without discarding any document content.
        if len(chunks) > 180:
            group_size = math.ceil(len(chunks) / 180)
            chunks = [" ".join(chunks[index:index + group_size]) for index in range(0, len(chunks), group_size)]
        return chunks

    @staticmethod
    def _tokens(text: str) -> List[str]:
        tokens = re.findall(r"c\+\+|c#|[\w]+(?:[.-][\w]+)*", text.lower(), re.UNICODE)
        return [token for token in tokens if token not in _STOP_WORDS and not token.isdigit()]

    @classmethod
    def _lexical_similarity(cls, resume_segments: List[str], jd_segments: List[str]) -> float:
        """Average each JD segment's best resume match, with TF-IDF cosine."""
        documents = resume_segments + jd_segments
        counts = []
        frequency = Counter()
        for document in documents:
            tokens = cls._tokens(document)
            terms = tokens + [left + " " + right for left, right in zip(tokens, tokens[1:])]
            term_counts = Counter(terms)
            counts.append(term_counts)
            frequency.update(term_counts.keys())
        vectors = []
        for term_counts in counts:
            vector = {
                term: (1 + math.log(count)) * (1 + math.log((1 + len(documents)) / (1 + frequency[term])))
                for term, count in term_counts.items()
            }
            norm = math.sqrt(sum(value * value for value in vector.values()))
            vectors.append({term: value / norm for term, value in vector.items()} if norm else {})
        resume_vectors = vectors[:len(resume_segments)]
        jd_vectors = vectors[len(resume_segments):]
        best_matches = []
        for jd_vector in jd_vectors:
            best_matches.append(max((sum(value * resume_vector.get(term, 0.0) for term, value in jd_vector.items())
                                     for resume_vector in resume_vectors), default=0.0))
        return cls._bounded(sum(best_matches) / len(best_matches) if best_matches else 0.0)

    @staticmethod
    def _bounded(value: float) -> float:
        return min(1.0, max(0.0, float(value))) if math.isfinite(float(value)) else 0.0

    def _load_semantic_model(self):
        if self._model_attempted:
            return self.model
        self._model_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, local_files_only=True, trust_remote_code=False)
        except Exception:
            self._model_warning = (
                "The optional semantic model is unavailable locally. Used local TF-IDF wording comparison; "
                "no model was downloaded. Install optional dependencies and cache the model explicitly to enable semantics."
            )
        return self.model

    def _content_similarity(self, resume_text: str, jd_text: str):
        resume_segments = self._split_into_sentences(resume_text)
        jd_segments = self._split_into_sentences(jd_text)
        warnings = []
        if self.use_semantic:
            model = self._load_semantic_model()
            if model is not None:
                try:
                    embeddings = model.encode(resume_segments + jd_segments, convert_to_numpy=True, show_progress_bar=False)
                    vectors = []
                    for vector in embeddings:
                        values = [float(value) for value in vector]
                        if not values or not all(math.isfinite(value) for value in values):
                            raise ValueError("Invalid embedding")
                        norm = math.sqrt(sum(value * value for value in values))
                        if not norm:
                            raise ValueError("Empty embedding")
                        vectors.append([value / norm for value in values])
                    if len(vectors) != len(resume_segments) + len(jd_segments):
                        raise ValueError("Incomplete embeddings")
                    resume_vectors = vectors[:len(resume_segments)]
                    best = [max(sum(left * right for left, right in zip(jd_vector, resume_vector))
                                for resume_vector in resume_vectors)
                            for jd_vector in vectors[len(resume_segments):]]
                    return self._bounded(sum(self._bounded(value) for value in best) / len(best)), "semantic", warnings
                except Exception:
                    warnings.append("Semantic comparison could not finish. Used local TF-IDF wording comparison instead.")
            elif self._model_warning:
                warnings.append(self._model_warning)
        return self._lexical_similarity(resume_segments, jd_segments), "lexical_tfidf", warnings

    def calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Compatibility API: configured content alignment in [0, 1].

        Inspect analyze()['similarity_method'] to distinguish semantic comparison
        from the lexical default/fallback.
        """
        resume_text = self._validate_text(resume_text, "Resume")
        jd_text = self._validate_text(jd_text, "Job description")
        return self._content_similarity(resume_text, jd_text)[0]

    @staticmethod
    def _resume_checks(resume_text: str) -> List[dict]:
        tests = [
            ("Contact email", bool(re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", resume_text)),
             "An email address was found.", "Add a professional email address to the resume you submit."),
            ("Project or work evidence", bool(re.search(r"\b(projects?|experience|internship|employment)\b", resume_text, re.I)),
             "A project or experience section is mentioned.", "Add project or work examples that support your skills."),
            ("Measurable outcomes", bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|users?\b|requests?\b|records?\b|ms\b|seconds?\b|hours?\b|samples?\b|documents?\b)|\b(?:reduced|improved|increased|saved)\b[^.\n]{0,45}\b\d+", resume_text, re.I)),
             "A possible measurable outcome was found; check that it is accurate and relevant.",
             "Where you have measured results, add a metric such as evaluation accuracy, latency, or dataset size."),
            ("Portfolio link", bool(re.search(r"(?:https?://|github\.com/|linkedin\.com/|huggingface\.co/)", resume_text, re.I)),
             "A web or portfolio link was found.", "Consider linking a working repository or project demo."),
        ]
        return [{"label": label, "status": "pass" if passed else "review", "detail": good if passed else improve}
                for label, passed, good, improve in tests]

    @staticmethod
    def _recommendations(evidence: List[dict], self_reported: List[str], checks: List[dict]) -> List[dict]:
        recommendations = []
        missing = [item for item in evidence if item["status"] == "missing"]
        reported_gaps = [item["skill"] for item in missing if item["skill"] in self_reported]
        if reported_gaps:
            recommendations.append({
                "priority": "high", "title": "Document the skills you reported",
                "detail": "You reported " + ", ".join(reported_gaps[:8]) + ". These were not found in your resume. "
                          "If you have used them, add a specific project, your contribution, and an outcome. "
                          "They have not been added to your match score."
            })
        for priority, title, label in [("required", "Address required skill gaps", "high"), ("preferred", "Consider preferred skills", "medium")]:
            gaps = [item["skill"] for item in missing if item["priority"] == priority and item["skill"] not in reported_gaps]
            if gaps:
                recommendations.append({
                    "priority": label, "title": title,
                    "detail": "No affirmative resume evidence was found for " + ", ".join(gaps[:8]) +
                              (f" (and {len(gaps) - 8} more)" if len(gaps) > 8 else "") + ". "
                              "Add truthful evidence if you have this experience; otherwise use these as learning priorities."
                })
        matched = [item["skill"] for item in evidence if item["status"] == "matched"]
        if matched:
            recommendations.append({
                "priority": "medium", "title": "Lead with relevant project evidence",
                "detail": "The JD and resume both mention " + ", ".join(matched[:6]) +
                          ". Make your contribution easy to verify: problem, implementation, evaluation, and measured result."
            })
        for check in checks:
            if check["status"] == "review" and check["label"] in {"Measurable outcomes", "Project or work evidence", "Portfolio link"}:
                recommendations.append({"priority": "medium", "title": check["label"], "detail": check["detail"]})
        if not evidence:
            recommendations.insert(0, {
                "priority": "high", "title": "Use a more specific job description",
                "detail": "No skills from the supported vocabulary were identified. Include the role's actual technical requirements "
                          "and review unsupported terminology manually; the score currently measures wording only."
            })
        return recommendations

    def analyze(self, resume_text: str, jd_text: str, extra_skills=None) -> Dict:
        resume_text = self._validate_text(resume_text, "Resume")
        jd_text = self._validate_text(jd_text, "Job description")
        resume_evidence = extract_skill_evidence(resume_text)
        jd_evidence = extract_skill_evidence(jd_text, is_job_description=True)
        self_reported = parse_self_reported_skills(extra_skills)
        resume_skills = set(resume_evidence)
        jd_skills = set(jd_evidence)
        overlapping = resume_skills & jd_skills
        content, method, warnings = self._content_similarity(resume_text, jd_text)
        evidence = [{
            "skill": skill, "status": "matched" if skill in overlapping else "missing",
            "priority": jd_evidence[skill]["priority"],
            "resume_evidence": resume_evidence[skill]["evidence"] if skill in overlapping else "",
            "jd_evidence": jd_evidence[skill]["evidence"],
        } for skill in sorted(jd_skills, key=str.casefold)]
        weights = {item["skill"]: (2 if item["priority"] == "required" else 1) for item in evidence}
        coverage = (sum(weights[skill] for skill in overlapping) / sum(weights.values())) if weights else None
        content_weight, skill_weight = (SEMANTIC_WEIGHT, SKILL_WEIGHT) if jd_skills else (1.0, 0.0)
        overall = self._bounded(content * content_weight + (coverage or 0.0) * skill_weight)
        if not jd_skills:
            warnings.append("No supported skills were recognized in the job description. Skill coverage is unavailable; the overall score uses content similarity only.")
        if len(resume_text.split()) < 50:
            warnings.append("The resume is short. Use the full resume for more useful evidence and recommendations.")
        if len(jd_text.split()) < 30:
            warnings.append("The job description is short. Paste the full requirements for a more representative comparison.")
        if self_reported:
            warnings.append("Additional skills are self-reported and do not affect document-based match scores.")
        warnings.append("Scores are heuristic document alignment, not a validated ATS score or a prediction of hiring success. Skill mentions do not establish proficiency.")
        checks = self._resume_checks(resume_text)
        return {
            "schema_version": "2.0",
            "overall_match_percentage": round(overall * 100, 2),
            "semantic_similarity": round(content * 100, 2) if method == "semantic" else None,
            "content_similarity": round(content * 100, 2),
            "similarity_method": method,
            "skill_match_percentage": round(coverage * 100, 2) if coverage is not None else None,
            "overlapping_skills": sorted(overlapping, key=str.casefold),
            "missing_skills": sorted(jd_skills - resume_skills, key=str.casefold),
            "resume_skills": sorted(resume_skills, key=str.casefold),
            "jd_skills": sorted(jd_skills, key=str.casefold),
            "total_required_skills": len(jd_skills),
            "required_skill_count": sum(item["priority"] == "required" for item in evidence),
            "preferred_skill_count": sum(item["priority"] == "preferred" for item in evidence),
            "matched_skills": len(overlapping),
            "self_reported_skills": self_reported,
            "skill_evidence": evidence,
            "recommendations": self._recommendations(evidence, self_reported, checks),
            "warnings": warnings,
            "scoring": {"content_weight": content_weight, "skill_weight": skill_weight,
                        "required_skill_weight": 2, "preferred_skill_weight": 1,
                        "description": "40% content alignment + 60% weighted skill coverage. Required skills count twice; unmarked skills default to required. With no recognized JD skills, only content is scored."},
            "resume_checks": checks,
        }

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        from src.documents import extract_text
        return extract_text(pdf_path)

    def match_resume_to_jd(self, resume_path: str, jd_text: str) -> Dict:
        from src.documents import extract_text
        return self.analyze(extract_text(resume_path), jd_text)
