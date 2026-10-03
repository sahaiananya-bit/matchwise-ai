# TF-IDF versus MiniLM: measured comparison

This is a small **AI-authored synthetic diagnostic**, not an independent hiring benchmark. Labels were fixed before inference. No scoring weights or cases were tuned after seeing results.

Twenty requirement / preferred-evidence / alternative-evidence triples test direct matches, paraphrases, negation, partial coverage, and keyword traps (four each). A correct pair requires the preferred text to score strictly higher; ties count as incorrect. Both engines use the same passages, skill rules and weights. Content scores isolate the method; overall scores include the shared skill heuristic. This measures text relevance, not applicant suitability.

| Method | Content ordering | Overall ordering | Median analysis latency |
| --- | ---: | ---: | ---: |
| lexical_tfidf | 8/20 | 11/20 | 0.45 ms |
| semantic | 11/20 | 13/20 | 12.72 ms |

## By category

| Category | TF-IDF content / overall | MiniLM content / overall |
| --- | ---: | ---: |
| direct | 4/4 / 4/4 | 4/4 / 4/4 |
| paraphrase | 0/4 / 0/4 | 1/4 / 1/4 |
| negation | 0/4 / 3/4 | 2/4 / 4/4 |
| partial | 4/4 / 4/4 | 4/4 / 4/4 |
| keyword_trap | 0/4 / 0/4 | 0/4 / 0/4 |

## Failures to inspect

- **lexical_tfidf**, overall score: retrieval_paraphrase, latency_paraphrase, speech_paraphrase, observability_paraphrase, future_rag, keyword_glossary, keyword_job_copy, keyword_definition, keyword_course.
- **semantic**, overall score: retrieval_paraphrase, latency_paraphrase, observability_paraphrase, keyword_glossary, keyword_job_copy, keyword_definition, keyword_course.

High word overlap and embedding similarity do not establish that someone performed the work. Read the evidence; copied requirements, definitions and negated claims can mislead these methods. The examples are short and authored with knowledge of the task, and there is no independent annotation, full-resume validation, fairness analysis or uncertainty estimate. Do not generalize the percentages to hiring accuracy.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
.\.venv\Scripts\python.exe scripts/cache_model.py
.\.venv\Scripts\python.exe scripts/compare_methods.py
```

Model: `sentence-transformers/all-MiniLM-L6-v2` at revision `7dbbc90392e2f80f3d3c277d6e90027e55de9125`. CPU only.
Fixture SHA-256: `ae1d857a471afe539e821c4052e2d82c979d49d6bdbe4f9f6cae36d2fe5f56c4`.
Run: 2026-10-03T13:50:49.582609+00:00. Python 3.12.14.
Packages: `{"numpy": "2.5.3", "sentence-transformers": "5.2.2", "torch": "2.14.1", "transformers": "5.18.0"}`.
Model load: 9.751 s. 3 timed repeats per input pair.
Median complete analyze() time per pair, warmed model, sequential CPU execution. Includes skill extraction; excludes load and download. Not a serving SLA.

[Raw results](results.json) · [Cases and labels](ranking_cases.json)
