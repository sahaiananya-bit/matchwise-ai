"""Run hand-authored matching regressions; this is not a hiring-quality benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matcher import ResumeJDMatcher  # noqa: E402


def evaluate(fixture_path: Path) -> dict[str, Any]:
    """Check semantic expectations without pinning implementation-specific floats."""
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    matcher = ResumeJDMatcher()
    results: dict[str, dict[str, Any]] = {}
    reports: list[dict[str, Any]] = []
    seen: set[str] = set()

    for case in fixture["cases"]:
        case_id = case["id"]
        if case_id in seen:
            raise ValueError(f"Duplicate fixture id: {case_id}")
        seen.add(case_id)
        failures: list[str] = []
        try:
            result = matcher.analyze(
                case["resume_text"],
                case["jd_text"],
                extra_skills=case.get("extra_skills"),
            )
            results[case_id] = result
            # Display capitalization is not part of the matching contract.
            matched = {skill.casefold() for skill in result["overlapping_skills"]}
            missing = {skill.casefold() for skill in result["missing_skills"]}
            for expected_key, actual in (
                ("expected_matched", matched),
                ("expected_missing", missing),
            ):
                absent = {skill.casefold() for skill in case.get(expected_key, [])} - actual
                if absent:
                    failures.append(f"{expected_key} not found: {', '.join(sorted(absent))}")
            unexpected = {
                skill.casefold() for skill in case.get("expected_not_matched", [])
            } & matched
            if unexpected:
                failures.append(f"False skill matches: {', '.join(sorted(unexpected))}")
            if matched & missing:
                failures.append("A skill appeared in both matched and missing sets")
            for key in ("overall_match_percentage", "semantic_similarity", "skill_match_percentage"):
                # An inactive scoring component is explicitly unavailable, not zero.
                if result[key] is not None and not 0 <= result[key] <= 100:
                    failures.append(f"{key} must be in the range 0–100")
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
        reports.append({"id": case_id, "passed": not failures, "failures": failures})

    ordering_reports: list[dict[str, Any]] = []
    for rule in fixture.get("ordering_constraints", []):
        higher = rule["higher"]
        lower = rule["lower"]
        field = rule.get("field", "overall_match_percentage")
        if higher not in seen or lower not in seen:
            raise ValueError(f"Ordering rule references an unknown case: {higher}, {lower}")
        passed = (
            higher in results
            and lower in results
            and results[higher][field] > results[lower][field]
        )
        ordering_reports.append(
            {"higher": higher, "lower": lower, "field": field, "passed": passed}
        )

    return {
        "description": fixture.get("description", "Hand-authored regression fixtures"),
        "fixture": str(fixture_path),
        "cases_passed": sum(report["passed"] for report in reports),
        "cases_total": len(reports),
        "ordering_passed": sum(report["passed"] for report in ordering_reports),
        "ordering_total": len(ordering_reports),
        "passed": bool(reports)
        and all(report["passed"] for report in reports)
        and all(report["passed"] for report in ordering_reports),
        "cases": reports,
        "ordering_constraints": ordering_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "matching_cases.json",
        help="Path to a matching fixture JSON file",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    args = parser.parse_args()
    try:
        report = evaluate(args.cases)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Unable to evaluate fixtures: {exc}\n")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print("Matchwise hand-authored regression evaluation")
        print("These fixtures check expected behavior, not real-world hiring accuracy.\n")
        for case in report["cases"]:
            print(f"{'PASS' if case['passed'] else 'FAIL'} {case['id']}")
            for failure in case["failures"]:
                print(f"  {failure}")
        for rule in report["ordering_constraints"]:
            print(
                f"{'PASS' if rule['passed'] else 'FAIL'} "
                f"{rule['higher']} > {rule['lower']} ({rule['field']})"
            )
        print(
            f"\nCases: {report['cases_passed']}/{report['cases_total']} passed; "
            f"ordering checks: {report['ordering_passed']}/{report['ordering_total']} passed."
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
