"""Auditable skill vocabulary: aliases and affirmative evidence, not proficiency.

Arbitrary noun phrases, companies and job titles never become skills. Add new
canonical names and aliases here as the supported vocabulary grows.
"""

import re
from typing import Dict, List


SKILL_ALIASES = {
    "Python": ("python",),
    "Java": ("java",),
    "JavaScript": ("javascript", "java script"),
    "TypeScript": ("typescript",),
    "C++": ("c++", "cpp"),
    "C#": ("c#", "c sharp"),
    "Go": ("golang", "go"),
    "R": ("r", "r language"),
    "Rust": ("rust",),
    "SQL": ("sql", "structured query language"),
    "Bash": ("bash", "shell scripting"),
    "MATLAB": ("matlab",),
    "HTML": ("html", "html5"),
    "CSS": ("css", "css3"),
    "React": ("react", "reactjs", "react.js"),
    "Next.js": ("next.js", "nextjs"),
    "Node.js": ("node.js", "nodejs"),
    "Django": ("django",),
    "Flask": ("flask",),
    "FastAPI": ("fastapi", "fast api"),
    "Streamlit": ("streamlit",),
    "REST APIs": ("rest api", "rest apis", "restful api", "restful apis"),
    "GraphQL": ("graphql",),
    "Machine learning": ("machine learning", "ml"),
    "Deep learning": ("deep learning",),
    "Artificial intelligence": ("artificial intelligence", "ai"),
    "Generative AI": ("generative ai", "genai", "gen ai"),
    "NLP": ("nlp", "natural language processing"),
    "Computer vision": ("computer vision",),
    "LLMs": ("llm", "llms", "large language model", "large language models"),
    "RAG": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "Embeddings": ("embeddings", "embedding models", "text embeddings", "vector embeddings"),
    "Vector databases": ("vector database", "vector databases", "vector store", "vector stores"),
    "LangChain": ("langchain", "lang chain"),
    "LlamaIndex": ("llamaindex", "llama index"),
    "Hugging Face": ("hugging face", "huggingface"),
    "Transformers": ("transformers", "transformer models", "transformer architecture"),
    "Sentence transformers": ("sentence-transformers", "sentence transformers", "sentence transformer"),
    "Prompt engineering": ("prompt engineering", "prompt design"),
    "Fine-tuning": ("fine-tuning", "fine tuning", "finetuning", "fine tune", "fine-tune"),
    "LoRA": ("lora", "low-rank adaptation", "low rank adaptation"),
    "OpenAI API": ("openai api", "openai apis", "openai sdk"),
    "LLM evaluation": ("llm evaluation", "llm evaluations", "rag evaluation", "evals"),
    "Model evaluation": ("model evaluation", "model validation"),
    "AI agents": ("ai agents", "ai agent", "agentic ai", "agentic workflows"),
    "PyTorch": ("pytorch", "py torch"),
    "TensorFlow": ("tensorflow", "tensor flow"),
    "Keras": ("keras",),
    "scikit-learn": ("scikit-learn", "scikit learn", "sklearn"),
    "pandas": ("pandas",),
    "NumPy": ("numpy",),
    "Matplotlib": ("matplotlib",),
    "Seaborn": ("seaborn",),
    "Plotly": ("plotly",),
    "Statistics": ("statistics", "statistical analysis", "statistical modeling", "statistical modelling"),
    "Data analysis": ("data analysis", "data analytics"),
    "Data engineering": ("data engineering",),
    "Feature engineering": ("feature engineering",),
    "ETL": ("etl", "extract transform load"),
    "Apache Spark": ("apache spark", "pyspark", "spark"),
    "Airflow": ("apache airflow", "airflow"),
    "Kafka": ("apache kafka", "kafka"),
    "PostgreSQL": ("postgresql", "postgres"),
    "MySQL": ("mysql",),
    "SQLite": ("sqlite",),
    "MongoDB": ("mongodb", "mongo db"),
    "Redis": ("redis",),
    "Elasticsearch": ("elasticsearch", "elastic search"),
    "FAISS": ("faiss",),
    "Pinecone": ("pinecone",),
    "Chroma": ("chromadb", "chroma db", "chroma"),
    "Qdrant": ("qdrant",),
    "Weaviate": ("weaviate",),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "Google Cloud": ("gcp", "google cloud", "google cloud platform"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Terraform": ("terraform",),
    "CI/CD": ("ci/cd", "ci cd", "continuous integration", "continuous delivery", "continuous deployment"),
    "Git": ("git",),
    "GitHub Actions": ("github actions",),
    "Linux": ("linux",),
    "MLOps": ("mlops", "ml ops"),
    "MLflow": ("mlflow", "ml flow"),
    "Weights & Biases": ("weights & biases", "weights and biases", "wandb"),
    "DVC": ("dvc", "data version control"),
    "SageMaker": ("sagemaker", "sage maker"),
    "Vertex AI": ("vertex ai",),
    "ONNX": ("onnx",),
    "pytest": ("pytest",),
    "Unit testing": ("unit testing", "unit tests"),
    "A/B testing": ("a/b testing", "ab testing", "a-b testing"),
    "Agile": ("agile",),
    "Scrum": ("scrum",),
    "Communication": ("communication", "communication skills"),
    "Teamwork": ("teamwork", "team work"),
    "Leadership": ("leadership",),
    "Project management": ("project management",),
}


# Custom lookarounds preserve C++/C# and avoid Java inside JavaScript.
_PATTERNS = {
    skill: re.compile(
        r"(?<![\w+#])(?:" + "|".join(
            re.escape(alias).replace(r"\ ", r"\s+")
            for alias in sorted(aliases, key=len, reverse=True)
        ) + r")(?![\w+#])", re.IGNORECASE
    )
    for skill, aliases in SKILL_ALIASES.items()
}
_PREFERRED = re.compile(r"\b(preferred|desirable|bonus|optional|nice[- ]to[- ]have|a plus|advantageous)\b", re.I)
_REQUIRED = re.compile(r"\b(required|requirements|must[- ]have|mandatory|essential|minimum qualifications)\b", re.I)
_NEGATIVE_PREFIX = re.compile(
    r"\b(?:no|not|never|without|lack(?:s|ing)?|unfamiliar(?:ity)?(?: with)?|"
    r"don['’]t|doesn['’]t|didn['’]t|haven['’]t|hasn['’]t|cannot|can['’]t|"
    r"no experience(?: with| in)?|not experienced(?: with| in)?|"
    r"(?:want|plan|hope|aim)(?:ing|s)? to (?:learn|study)|"
    r"interested in learning)\b", re.I
)


def _ambiguous_match(skill: str, match: re.Match, sentence: str) -> bool:
    """Require programming context for one-letter R and the English verb go."""
    token = match.group()
    if skill == "Go" and token.lower() == "go":
        after = sentence[match.end():].lower()
        if re.match(r"\s+(?:to|for|ahead|beyond|through|back|live|with|on|into|out|home)\b", after):
            return True
        context = sentence[max(0, match.start() - 40):match.end() + 45]
        return not (
            re.search(r"\b(?:programming|language|developer|engineer|backend|microservices|skills|proficient|familiar|experience)\b", context, re.I)
            or re.search(r"[,/|]", context)
            or re.fullmatch(r"\s*[-*•]?\s*Go\s*[.;:]?\s*", sentence)
        )
    if skill == "R" and token.lower() == "r":
        context = sentence[max(0, match.start() - 40):match.end() + 45]
        if re.match(r"\s*&", sentence[match.end():]):
            return True
        return not (
            token == "R" and (
                re.search(r"\b(?:programming|language|statistics|statistical|analysis|skills|python)\b", context, re.I)
                or re.search(r"[,/|]", context)
                or re.fullmatch(r"\s*[-*•]?\s*R\s*[.;:]?\s*", sentence)
            )
        )
    return False


def _is_negated(sentence: str, start: int, end: int) -> bool:
    # Limit a negation to its clause. Lists joined by 'and' retain negation.
    prefix = re.split(r"[;.!?]|\b(?:but|however|although)\b", sentence[:start], flags=re.I)[-1][-150:]
    prefix = re.sub(r"\bnot (?:only|just|merely)\b|\bno (?:difficulty|issues?|problems?) with\b", "", prefix, flags=re.I)
    if re.search(r"(?:^|\bI am\b|\bcurrently\b)\s*learning\b", prefix, re.I):
        return True
    if _NEGATIVE_PREFIX.search(prefix):
        return True
    suffix = sentence[end:]
    negative = re.search(
        r"\b(?:not\s+(?:(?:my|our|a)\s+)?(?:required|needed|known|used|skills?|expertise)|"
        r"no\s+(?:experience|knowledge|proficiency))\b", suffix, re.I
    )
    if negative:
        # A suffix can negate a whole list ('Python and SQL are not required').
        # Do not carry it across a positive clause ('Python required, SQL not required').
        between = suffix[:negative.start()]
        for pattern in _PATTERNS.values():
            between = pattern.sub("", between)
        between = re.sub(r"\b(?:and|or|nor|is|are|experience|knowledge|skills?|programming|in|with)\b", "", between, flags=re.I)
        if not re.search(r"\w", between):
            return True
    return False


def extract_skill_evidence(text: str, is_job_description: bool = False) -> Dict[str, dict]:
    """Find canonical skills with original excerpts and required/preferred status.

    Preferred headings carry over to following bullets. Unmarked JD skills
    default to required; explicit skill-local priority overrides the heading.
    """
    found: Dict[str, dict] = {}
    heading_priority = "required"
    for line in text.splitlines() or [text]:
        line = line.strip()
        if not line:
            continue
        if len(line) < 90 and (line.endswith(":") or len(line.split()) <= 4):
            if _PREFERRED.search(line):
                heading_priority = "preferred"
            elif _REQUIRED.search(line) or re.search(r"\b(responsibilities|qualifications|experience|skills)\b", line, re.I):
                heading_priority = "required"
        for sentence in re.split(r"(?<=[.!?])\s+|;", line):
            if not sentence.strip():
                continue
            for skill, pattern in _PATTERNS.items():
                for match in pattern.finditer(sentence):
                    if _ambiguous_match(skill, match, sentence) or _is_negated(sentence, *match.span()):
                        continue
                    priority = heading_priority
                    if is_job_description:
                        markers = [(m, "preferred") for m in _PREFERRED.finditer(sentence)]
                        markers += [(m, "required") for m in _REQUIRED.finditer(sentence)]
                        if markers:
                            _, priority = min(markers, key=lambda pair: min(
                                abs(pair[0].start() - match.end()), abs(pair[0].end() - match.start())
                            ))
                    excerpt_start = max(0, match.start() - 130)
                    excerpt_end = min(len(sentence), match.end() + 170)
                    evidence = ("…" if excerpt_start else "") + sentence[excerpt_start:excerpt_end].strip() + ("…" if excerpt_end < len(sentence) else "")
                    existing = found.get(skill)
                    if existing is None or (existing["priority"] == "preferred" and priority == "required"):
                        found[skill] = {"evidence": evidence, "priority": priority}
                    break
    return found


def parse_self_reported_skills(skills) -> List[str]:
    """Normalize optional user claims, keeping them separate from resume evidence."""
    if skills is None:
        return []
    if isinstance(skills, str):
        source = skills
    elif isinstance(skills, (list, tuple, set)) and all(isinstance(item, str) for item in skills):
        source = ", ".join(skills)
    else:
        raise ValueError("Additional skills must be text or a list of skill names.")
    if len(source) > 5000:
        raise ValueError("Additional skills must be at most 5,000 characters.")
    return sorted(extract_skill_evidence("Skills: " + source), key=str.casefold)
