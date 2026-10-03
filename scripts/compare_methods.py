"""Reproducible paired relevance evaluation; semantic fallback is a hard error."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.matcher import ResumeJDMatcher


def summarize(rows):
    output = {}
    for field in ("content_similarity", "overall_match_percentage"):
        wins = sum(row["preferred"][field] > row["alternative"][field] for row in rows)
        ties = sum(row["preferred"][field] == row["alternative"][field] for row in rows)
        output[field] = {"correct": wins, "ties": ties, "total": len(rows),
                         "strict_pairwise_accuracy": round(wins / len(rows), 4)}
    return output


def evaluate(cases_path, model_path, repeats=3):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    raw = cases_path.read_bytes()
    fixture = json.loads(raw)
    cases = fixture["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    engines = {"lexical_tfidf": ResumeJDMatcher(),
               "semantic": ResumeJDMatcher(model_name=str(model_path), use_semantic=True)}
    start = time.perf_counter()
    if engines["semantic"]._load_semantic_model() is None:
        raise RuntimeError("Semantic model unavailable. Run scripts/cache_model.py first. No comparison was published.")
    load_seconds = time.perf_counter() - start
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "fixture_sha256": hashlib.sha256(raw).hexdigest(),
              "provenance": fixture["provenance"], "label_rule": fixture["label_rule"],
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {name: version(name) for name in
                                           ("sentence-transformers", "torch", "transformers", "numpy")}},
              "model": json.loads((model_path / "matchwise-model.json").read_text()),
              "model_load_seconds": round(load_seconds, 3),
              "timing": "Median complete analyze() time per pair, warmed model, sequential CPU execution. Includes skill extraction; excludes load and download. Not a serving SLA.",
              "repeats": repeats, "methods": {}}
    for method, engine in engines.items():
        # Warm up inference outside measured runs.
        warm = engine.analyze(cases[0]["preferred"], cases[0]["job"])
        if warm["similarity_method"] != method:
            raise RuntimeError(f"Expected {method}, got fallback")
        rows, durations = [], []
        for case in cases:
            row = {key: case[key] for key in ("id", "category", "reason")}
            for side in ("preferred", "alternative"):
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    result = engine.analyze(case[side], case["job"])
                    samples.append((time.perf_counter() - start) * 1000)
                    if result["similarity_method"] != method:
                        raise RuntimeError(f"Semantic fallback on {case['id']}; refusing to publish")
                row[side] = {key: result[key] for key in
                             ("content_similarity", "overall_match_percentage", "skill_match_percentage")}
                row[side]["median_ms"] = round(statistics.median(samples), 3)
                durations.extend(samples)
            rows.append(row)
        groups = defaultdict(list)
        for row in rows:
            groups[row["category"]].append(row)
        report["methods"][method] = {"summary": summarize(rows),
                                    "by_category": {key: summarize(value) for key, value in groups.items()},
                                    "median_ms": round(statistics.median(durations), 3), "cases": rows}
    return report


def markdown(report):
    lines = ["# TF-IDF versus MiniLM: measured comparison", "",
             "This is a small **AI-authored synthetic diagnostic**, not an independent hiring benchmark. "
             "Labels were fixed before inference. No scoring weights or cases were tuned after seeing results.", "",
             "Twenty requirement / preferred-evidence / alternative-evidence triples test direct matches, "
             "paraphrases, negation, partial coverage, and keyword traps (four each). "
             "A correct pair requires the preferred text to score strictly higher; ties count as incorrect. "
             "Both engines use the same passages, skill rules and weights. Content scores isolate the method; "
             "overall scores include the shared skill heuristic. This measures text relevance, not applicant suitability.", "",
             "| Method | Content ordering | Overall ordering | Median analysis latency |",
             "| --- | ---: | ---: | ---: |"]
    for method, data in report["methods"].items():
        content = data["summary"]["content_similarity"]
        overall = data["summary"]["overall_match_percentage"]
        lines.append(f"| {method} | {content['correct']}/{content['total']} | {overall['correct']}/{overall['total']} | {data['median_ms']:.2f} ms |")
    lines += ["", "## By category", "", "| Category | TF-IDF content / overall | MiniLM content / overall |", "| --- | ---: | ---: |"]
    for category in report["methods"]["lexical_tfidf"]["by_category"]:
        cells = []
        for data in report["methods"].values():
            group = data["by_category"][category]
            cells.append(f"{group['content_similarity']['correct']}/4 / {group['overall_match_percentage']['correct']}/4")
        lines.append(f"| {category} | {' | '.join(cells)} |")
    lines += ["", "## Failures to inspect", ""]
    for method, data in report["methods"].items():
        failed = [row["id"] for row in data["cases"] if row["preferred"]["overall_match_percentage"] <= row["alternative"]["overall_match_percentage"]]
        lines.append(f"- **{method}**, overall score: {', '.join(failed) or 'none on this fixture'}.")
    lines += ["", "High word overlap and embedding similarity do not establish that someone performed the work. "
              "Read the evidence; copied requirements, definitions and negated claims can mislead these methods. "
              "The examples are short and authored with knowledge of the task, and there is no independent annotation, "
              "full-resume validation, fairness analysis or uncertainty estimate. Do not generalize the percentages to hiring accuracy.", "",
              "## Reproduction", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m pip install -r requirements-semantic.txt",
              ".\\.venv\\Scripts\\python.exe scripts/cache_model.py", ".\\.venv\\Scripts\\python.exe scripts/compare_methods.py", "```", "",
              f"Model: `{report['model']['model_id']}` at revision `{report['model']['revision']}`. CPU only.",
              f"Fixture SHA-256: `{report['fixture_sha256']}`.",
              f"Run: {report['created_at_utc']}. Python {report['environment']['python']}.",
              f"Packages: `{json.dumps(report['environment']['packages'], sort_keys=True)}`.",
              f"Model load: {report['model_load_seconds']} s. {report['repeats']} timed repeats per input pair.",
              report["timing"], "", "[Raw results](results.json) · [Cases and labels](ranking_cases.json)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "benchmarks/ranking_cases.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models/all-MiniLM-L6-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    report = evaluate(args.cases, args.model, args.repeats)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "RESULTS.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
