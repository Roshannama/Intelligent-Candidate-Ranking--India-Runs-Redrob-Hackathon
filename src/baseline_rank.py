import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"
CANDIDATE_FILE = ARTIFACT_DIR / "candidates.parquet"
EMBEDDING_FILE = ARTIFACT_DIR / "candidate_embeddings.npy"
JD_FILE = DATA_DIR / "jd.txt"
OUTPUT_FILE = OUTPUT_DIR / "baseline_top5000.csv"
REQUIRED_SKILLS = {
    "python",
    "embeddings",
    "retrieval",
    "ranking",
    "vector database",
    "hybrid search",
}
IMPORTANT_SKILLS = {
    "sentence transformers",
    "bge",
    "e5",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "faiss",
    "machine learning",
    "information retrieval",
    "nlp",
    "llm",
    "fine tuning",
    "lora",
    "qlora",
    "peft",
    "xgboost",
    "lightgbm",
    "ndcg",
    "mrr",
    "map",
    "evaluation",
    "ranking systems",
    "search",
    "recommendation",
}
NEGATIVE_TERMS = {
    "langchain",
    "openai api",
    "chatgpt api",
}
TARGET_MIN_EXP = 5
TARGET_MAX_EXP = 9
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#.\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default
def find_column(df, candidates):
    columns = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None
def parse_skills(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                name = item.get("name", "")
            else:
                name = item
            result.append(normalize_text(name))
        return set(result)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return {
                    normalize_text(
                        x.get("name", "") if isinstance(x, dict) else x
                    )
                    for x in parsed
                }
        except Exception:
            pass
        return {
            normalize_text(x)
            for x in value.split(",")
            if normalize_text(x)
        }
    return set()
def calculate_skill_score(candidate_skills):
    candidate_skills = {
        normalize_text(x)
        for x in candidate_skills
    }
    required_matches = 0
    for skill in REQUIRED_SKILLS:
        if skill in candidate_skills:
            required_matches += 1
    important_matches = 0
    for skill in IMPORTANT_SKILLS:
        if skill in candidate_skills:
            important_matches += 1
    required_score = required_matches / len(REQUIRED_SKILLS)
    important_score = min(
        important_matches / 8.0,
        1.0
    )
    score = (
        0.70 * required_score
        +
        0.30 * important_score
    )
    return min(score, 1.0)
def calculate_experience_score(years):
    years = safe_float(years)
    if TARGET_MIN_EXP <= years <= TARGET_MAX_EXP:
        return 1.0
    if years < TARGET_MIN_EXP:
        difference = TARGET_MIN_EXP - years
        return max(
            0.0,
            1.0 - difference / 5.0
        )
    difference = years - TARGET_MAX_EXP
    return max(
        0.0,
        1.0 - difference / 5.0
    )
def calculate_domain_score(text):
    text = normalize_text(text)
    positive_terms = [
        "machine learning",
        "ml engineer",
        "ai engineer",
        "ai",
        "machine learning engineer",
        "nlp",
        "search",
        "ranking",
        "retrieval",
        "recommendation",
        "data scientist",
        "software engineer",
        "backend engineer",
    ]
    matches = sum(
        1 for term in positive_terms
        if term in text
    )
    return min(matches / 4.0, 1.0)
def calculate_production_score(text):
    text = normalize_text(text)
    production_terms = [
        "production",
        "deployed",
        "real users",
        "real-time",
        "system",
        "service",
        "pipeline",
        "infrastructure",
        "scalable",
        "distributed",
    ]
    matches = sum(
        1 for term in production_terms
        if term in text
    )
    return min(matches / 5.0, 1.0)
def calculate_retrieval_score(text):
    text = normalize_text(text)
    terms = [
        "retrieval",
        "ranking",
        "search",
        "recommendation",
        "vector database",
        "faiss",
        "pinecone",
        "qdrant",
        "milvus",
        "elasticsearch",
        "opensearch",
        "embeddings",
        "hybrid search",
        "information retrieval",
        "ndcg",
        "mrr",
        "map",
    ]
    matches = sum(
        1 for term in terms
        if term in text
    )
    return min(matches / 5.0, 1.0)
def calculate_product_score(text):
    text = normalize_text(text)
    consulting_companies = [
        "tcs",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini",
    ]
    product_terms = [
        "product company",
        "saas",
        "startup",
        "platform",
        "product",
    ]
    consulting_penalty = any(
        company in text
        for company in consulting_companies
    )
    product_signal = sum(
        term in text
        for term in product_terms
    )
    score = min(product_signal / 2.0, 1.0)
    if consulting_penalty and score == 0:
        score = 0.2
    return score
def calculate_behavior_score(row):
    score = 0.0
    if str(row.get("open_to_work_flag", "")).lower() == "true":
        score += 0.30
    response_rate = safe_float(
        row.get("recruiter_response_rate", 0)
    )
    score += min(response_rate, 1.0) * 0.30
    github = safe_float(
        row.get("github_activity_score", 0)
    )
    if github >= 0:
        score += min(github / 100.0, 1.0) * 0.15
    saves = safe_float(
        row.get("saved_by_recruiters_30d", 0)
    )
    score += min(saves / 10.0, 1.0) * 0.15
    completeness = safe_float(
        row.get("profile_completeness_score", 0)
    )
    score += min(completeness / 100.0, 1.0) * 0.10
    return min(score, 1.0)
def calculate_penalty(text, years):
    text = normalize_text(text)
    penalty = 0.0
    if "research only" in text:
        penalty += 0.20
    if (
        "langchain" in text
        and "production" not in text
        and "retrieval" not in text
    ):
        penalty += 0.10
    domain_terms = [
        "computer vision",
        "speech recognition",
        "robotics",
    ]
    if any(term in text for term in domain_terms):
        if "nlp" not in text and "retrieval" not in text:
            penalty += 0.10
    return min(penalty, 0.50)
def main():
    print("Loading candidate features...")
    df = pd.read_parquet(CANDIDATE_FILE)
    print(f"Candidates loaded: {len(df):,}")
    id_col = find_column(
        df,
        ["candidate_id", "id"]
    )
    skills_col = find_column(
        df,
        ["normalized_skills", "skills"]
    )
    years_col = find_column(
        df,
        ["years_of_experience", "experience_years", "years"]
    )
    text_col = find_column(
        df,
        [
            "combined_text",
            "profile_text",
            "text",
            "summary"
        ]
    )
    if id_col is None:
        raise ValueError(
            "candidate_id column not found."
        )
    if text_col is None:
        text_columns = [
            c for c in df.columns
            if any(
                x in str(c).lower()
                for x in [
                    "summary",
                    "headline",
                    "experience",
                    "description",
                    "title"
                ]
            )
        ]
        df["_combined_text"] = (
            df[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        text_col = "_combined_text"
    print("Loading candidate embeddings...")
    embeddings = np.load(
        EMBEDDING_FILE,
        mmap_mode="r"
    )
    if len(embeddings) != len(df):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) "
            f"does not match candidates ({len(df)})."
        )
    jd_embedding_file = ARTIFACT_DIR / "jd_embedding.npy"
    if jd_embedding_file.exists():
        print("Loading JD embedding...")
        jd_embedding = np.load(
            jd_embedding_file
        )
        jd_embedding = jd_embedding.astype(
            np.float32
        )
        jd_norm = np.linalg.norm(jd_embedding)
        if jd_norm > 0:
            jd_embedding /= jd_norm
        candidate_norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )
        candidate_norms[
            candidate_norms == 0
        ] = 1
        semantic_scores = (
            embeddings @ jd_embedding
        ) / candidate_norms[:, 0]
        semantic_scores = (
            semantic_scores + 1
        ) / 2
    else:
        print(
            "WARNING: jd_embedding.npy not found."
        )
        print(
            "Semantic score will be disabled."
        )
        semantic_scores = np.zeros(
            len(df),
            dtype=np.float32
        )
    print("Calculating ranking features...")
    skill_scores = []
    experience_scores = []
    domain_scores = []
    production_scores = []
    retrieval_scores = []
    product_scores = []
    behavior_scores = []
    penalties = []
    for _, row in df.iterrows():
        skills = parse_skills(
            row.get(skills_col, [])
        )
        years = (
            row.get(years_col, 0)
            if years_col
            else 0
        )
        text = normalize_text(
            row.get(text_col, "")
        )
        skill_scores.append(
            calculate_skill_score(skills)
        )
        experience_scores.append(
            calculate_experience_score(years)
        )
        domain_scores.append(
            calculate_domain_score(text)
        )
        production_scores.append(
            calculate_production_score(text)
        )
        retrieval_scores.append(
            calculate_retrieval_score(text)
        )
        product_scores.append(
            calculate_product_score(text)
        )
        behavior_scores.append(
            calculate_behavior_score(row)
        )
        penalties.append(
            calculate_penalty(text, years)
        )
    skill_scores = np.asarray(
        skill_scores,
        dtype=np.float32
    )
    experience_scores = np.asarray(
        experience_scores,
        dtype=np.float32
    )
    domain_scores = np.asarray(
        domain_scores,
        dtype=np.float32
    )
    production_scores = np.asarray(
        production_scores,
        dtype=np.float32
    )
    retrieval_scores = np.asarray(
        retrieval_scores,
        dtype=np.float32
    )
    product_scores = np.asarray(
        product_scores,
        dtype=np.float32
    )
    behavior_scores = np.asarray(
        behavior_scores,
        dtype=np.float32
    )
    penalties = np.asarray(
        penalties,
        dtype=np.float32
    )
    final_score = (
        0.25 * skill_scores
        + 0.20 * semantic_scores
        + 0.15 * retrieval_scores
        + 0.12 * production_scores
        + 0.10 * experience_scores
        + 0.08 * domain_scores
        + 0.05 * product_scores
        + 0.05 * behavior_scores
        - penalties
    )
    result = pd.DataFrame({
        "candidate_id": df[id_col],
        "skill_score": skill_scores,
        "semantic_score": semantic_scores,
        "retrieval_score": retrieval_scores,
        "production_score": production_scores,
        "experience_score": experience_scores,
        "domain_score": domain_scores,
        "product_score": product_scores,
        "behavior_score": behavior_scores,
        "penalty": penalties,
        "final_score": final_score,
    })
    result = result.sort_values(
        "final_score",
        ascending=False
    ).reset_index(drop=True)
    result["rank"] = (
        np.arange(len(result)) + 1
    )
    top5000 = result.head(5000).copy()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    top5000.to_csv(
        OUTPUT_FILE,
        index=False
    )
    print()
    print("=" * 60)
    print("BASELINE RANKING COMPLETE")
    print("=" * 60)
    print(
        f"Candidates ranked : {len(result):,}"
    )
    print(
        f"Top candidates    : {len(top5000)}"
    )
    print(
        f"Output            : {OUTPUT_FILE}"
    )
    print()
    print(top5000.head(10).to_string(index=False))
if __name__ == "__main__":
    main()