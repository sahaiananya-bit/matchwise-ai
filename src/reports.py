"""Portable match reports with bounded disclosure and safe spreadsheet exports."""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping

_REPORT_FIELDS = (
    "schema_version", "overall_match_percentage", "semantic_similarity",
    "content_similarity", "similarity_method", "skill_match_percentage",
    "overlapping_skills", "missing_skills", "resume_skills", "jd_skills",
    "total_required_skills", "matched_skills", "skill_evidence",
    "recommendations", "warnings", "scoring", "resume_checks", "self_reported_skills",
)
_PRIVACY_NOTE = (
    "This report excludes the full resume and job description, but includes evidence "
    "excerpts that may contain personal information. Review it before sharing."
)
_LIMITATION_NOTE = (
    "Scores describe document alignment, not hiring probability or an ATS score. "
    "Missing evidence does not prove a missing skill. Human review is required."
)


def _safe_nested(value):
    """Never export raw input fields accidentally attached to nested results."""
    if isinstance(value, Mapping):
        return {
            str(key): _safe_nested(item) for key, item in value.items()
            if str(key).lower() not in {
                "resume_text", "raw_resume", "raw_resume_text", "jd_text",
                "job_description", "raw_job_description", "resume", "document_text",
            }
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_nested(item) for item in value]
    return value


def build_json_report(result: dict, role: str = "AI Engineer", job_title: str = "") -> str:
    """Export documented result fields, retaining evidence but omitting raw inputs."""
    report = {
        "report_type": "resume_job_alignment",
        "role": role,
        "job_title": job_title,
        "privacy_note": _PRIVACY_NOTE,
        "interpretation_note": _LIMITATION_NOTE,
        "analysis": {key: _safe_nested(result[key]) for key in _REPORT_FIELDS if key in result},
    }
    return json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)


def _md(value) -> str:
    text = str(value if value is not None else "Not assessed")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for symbol in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "|", "!"):
        text = text.replace(symbol, "\\" + symbol)
    return text.replace("\r", " ").replace("\n", " ")


def _percentage(value) -> str:
    return "Not assessed" if value is None else f"{value}%"


def build_markdown_report(result: dict, role: str = "AI Engineer", job_title: str = "") -> str:
    """Create a readable, shareable report using only observed analysis data."""
    lines = [
        "# Resume and job alignment report", "",
        f"**Target role:** {_md(role)}",
    ]
    if job_title:
        lines.append(f"**Job:** {_md(job_title)}")
    lines += [
        "", _LIMITATION_NOTE, "", "## Match overview", "",
        "| Measure | Result |", "| --- | --- |",
        f"| Overall alignment | {_md(_percentage(result.get('overall_match_percentage')))} |",
        f"| Skill coverage | {_md(_percentage(result.get('skill_match_percentage')))} |",
        f"| Content similarity | {_md(_percentage(result.get('content_similarity')))} |",
        f"| Similarity method | {_md(result.get('similarity_method', 'Not assessed'))} |",
        "", "## Skill evidence", "",
    ]
    evidence = result.get("skill_evidence", [])
    if evidence:
        lines += ["| Skill | Status | Priority | Resume evidence | Job evidence |", "| --- | --- | --- | --- | --- |"]
        for item in evidence:
            lines.append("| " + " | ".join(_md(item.get(key, "")) for key in (
                "skill", "status", "priority", "resume_evidence", "jd_evidence"
            )) + " |")
    else:
        lines.append("No recognized job skills were available for an evidence comparison.")
    lines += ["", "## Recommended next steps", ""]
    recommendations = result.get("recommendations", [])
    if recommendations:
        for item in recommendations:
            lines.append(f"- **{_md(item.get('title', 'Next step'))}** ({_md(item.get('priority', ''))}): {_md(item.get('detail', ''))}")
    else:
        lines.append("Review the job requirements and ensure each relevant claim has truthful project or work evidence.")
    lines += ["", "## Resume checks", ""]
    checks = result.get("resume_checks", [])
    if checks:
        for item in checks:
            lines.append(f"- **{_md(item.get('label', 'Check'))}** — {_md(item.get('status', ''))}: {_md(item.get('detail', ''))}")
    else:
        lines.append("No resume checks were provided.")
    if result.get("warnings"):
        lines += ["", "## Analysis notes", ""]
        lines.extend(f"- {_md(warning)}" for warning in result["warnings"])
    if result.get("scoring"):
        lines += ["", "## Scoring details", ""]
        for key, value in _safe_nested(result["scoring"]).items():
            lines.append(f"- **{_md(str(key).replace('_', ' '))}:** {_md(value)}")
    lines += ["", "## Sharing and privacy", "", _PRIVACY_NOTE, ""]
    return "\n".join(lines)


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        text = ", ".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = json.dumps(_safe_nested(value), ensure_ascii=False)
    else:
        text = str(value)
    # Spreadsheet apps can interpret formulas after leading spaces/control bytes.
    stripped = re.sub(r"^[\s\x00-\x1f\ufeff]+", "", text)
    if (stripped.startswith(("=", "+", "-", "@"))
            or (text and (ord(text[0]) < 32 or text[0] == "\ufeff"))):
        return "'" + text
    return text


def build_comparison_csv(rows: list[dict]) -> str:
    """Export ordered comparison columns, escaping formulas in headers and cells."""
    if not rows:
        return ""
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow([_csv_cell(field) for field in fields])
    for row in rows:
        writer.writerow([_csv_cell(row.get(field)) for field in fields])
    return stream.getvalue()
