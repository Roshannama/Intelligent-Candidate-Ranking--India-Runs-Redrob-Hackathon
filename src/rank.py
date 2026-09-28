import os
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
OUTPUT_DIR = ROOT / "outputs"
CANDIDATES_FILE = ARTIFACT_DIR / "candidates.parquet"
RANKER_FILE = ARTIFACT_DIR / "ranker.pkl"
JD_FILE = DATA_DIR / "jd.txt"
JD_FEATURES_FILE = ARTIFACT_DIR / "jd_features.json"
JD_EMBEDDING_FILE = ARTIFACT_DIR / "jd_embedding.npy"
CANDIDATE_EMBEDDING_FILE = (
    ARTIFACT_DIR / "candidate_embeddings.npy"
)
OUTPUT_FILE = OUTPUT_DIR / "submission.csv"
TOP_K = 100
MIN_EXPERIENCE = 5.0
MAX_EXPERIENCE = 9.0
ML_WEIGHT = 0.65
SEMANTIC_WEIGHT = 0.15
SKILL_WEIGHT = 0.12
EXPERIENCE_WEIGHT = 0.08
JD_SKILLS = {
    "python",
    "embeddings",
    "embedding",
    "retrieval",
    "ranking",
    "vector database",
    "vector databases",
    "information retrieval",
    "machine learning",
    "llm",
    "fine-tuning",
    "lora",
    "qlora",
    "peft",
    "learning-to-rank",
    "learning to rank",
    "xgboost",
    "faiss",
    "pinecone",
    "qdrant",
    "milvus",
    "elasticsearch",
    "opensearch",
    "sentence-transformers",
    "nlp",
    "evaluation",
    "ndcg",
    "mrr",
    "map",
    "production",
    "distributed systems",
    "recommendation",
    "search",
}
MODEL_FEATURES = [
    "years_experience",
    "education_tier",
    "profile_completeness",
    "open_to_work",
    "profile_views",
    "applications",
    "response_rate",
    "avg_response_hours",
    "connection_count",
    "endorsements",
    "notice_period",
    "github_activity",
    "search_appearance",
    "saved_by_recruiters",
    "interview_completion",
    "offer_acceptance",
    "verified_email",
    "verified_phone",
    "linkedin_connected",
    "assessment_average",
]
def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default
def safe_bool(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(bool(value))
    if isinstance(value, str):
        return (
            1.0
            if value.strip().lower()
            in {
                "true",
                "1",
                "yes",
                "y",
            }
            else 0.0
        )
    return 0.0
def education_tier_from_value(value):
    if value is None:
        return 0.0
    text = str(value).lower().strip()
    if text == "tier_1":
        return 1.0
    if text == "tier_2":
        return 2.0
    if text == "tier_3":
        return 3.0
    return 0.0
def get_education_tier(candidate):
    if "education_tier" in candidate:
        return safe_float(
            candidate.get(
                "education_tier"
            )
        )
    education = candidate.get(
        "education",
        None
    )
    if isinstance(
        education,
        list
    ) and len(education) > 0:
        first = education[0]
        if isinstance(
            first,
            dict
        ):
            return education_tier_from_value(
                first.get(
                    "tier",
                    ""
                )
            )
    if isinstance(
        education,
        str
    ):
        return education_tier_from_value(
            education
        )
    return 0.0
def get_assessment_average(candidate):
    if "assessment_average" in candidate:
        return safe_float(
            candidate.get(
                "assessment_average"
            )
        )
    scores = candidate.get(
        "skill_assessment_scores",
        None
    )
    if scores is None:
        scores = candidate.get(
            "assessments",
            None
        )
    if not isinstance(
        scores,
        dict
    ):
        return 0.0
    values = []
    for value in scores.values():
        try:
            values.append(
                float(value)
            )
        except Exception:
            continue
    if not values:
        return 0.0
    return sum(values) / len(values)
def build_model_features(candidates):
    print(
        "Building model features..."
    )
    rows = []
    for candidate in candidates:
        if isinstance(
            candidate,
            pd.Series
        ):
            candidate = candidate.to_dict()
        profile = candidate.get(
            "profile",
            {}
        )
        if not isinstance(
            profile,
            dict
        ):
            profile = {}
        signals = candidate.get(
            "redrob_signals",
            {}
        )
        if not isinstance(
            signals,
            dict
        ):
            signals = {}
        years_experience = safe_float(
            candidate.get(
                "years_experience",
                profile.get(
                    "years_of_experience",
                    0
                )
            )
        )
        education_tier = get_education_tier(
            candidate
        )
        assessment_average = (
            get_assessment_average(
                candidate
            )
        )
        profile_completeness = safe_float(
            candidate.get(
                "profile_completeness",
                signals.get(
                    "profile_completeness_score",
                    0
                )
            )
        )
        open_to_work = safe_bool(
            candidate.get(
                "open_to_work",
                signals.get(
                    "open_to_work_flag",
                    False
                )
            )
        )
        profile_views = safe_float(
            candidate.get(
                "profile_views",
                signals.get(
                    "profile_views_received_30d",
                    0
                )
            )
        )
        applications = safe_float(
            candidate.get(
                "applications",
                signals.get(
                    "applications_submitted_30d",
                    0
                )
            )
        )
        response_rate = safe_float(
            candidate.get(
                "response_rate",
                signals.get(
                    "recruiter_response_rate",
                    0
                )
            )
        )
        avg_response_hours = safe_float(
            candidate.get(
                "avg_response_hours",
                signals.get(
                    "avg_response_time_hours",
                    0
                )
            )
        )
        connection_count = safe_float(
            candidate.get(
                "connection_count",
                signals.get(
                    "connection_count",
                    0
                )
            )
        )
        endorsements = safe_float(
            candidate.get(
                "endorsements",
                signals.get(
                    "endorsements_received",
                    0
                )
            )
        )
        notice_period = safe_float(
            candidate.get(
                "notice_period",
                signals.get(
                    "notice_period_days",
                    0
                )
            )
        )
        github_activity = safe_float(
            candidate.get(
                "github_activity",
                signals.get(
                    "github_activity_score",
                    0
                )
            )
        )
        search_appearance = safe_float(
            candidate.get(
                "search_appearance",
                signals.get(
                    "search_appearance_30d",
                    0
                )
            )
        )
        saved_by_recruiters = safe_float(
            candidate.get(
                "saved_by_recruiters",
                signals.get(
                    "saved_by_recruiters_30d",
                    0
                )
            )
        )
        interview_completion = safe_float(
            candidate.get(
                "interview_completion",
                signals.get(
                    "interview_completion_rate",
                    0
                )
            )
        )
        offer_acceptance = safe_float(
            candidate.get(
                "offer_acceptance",
                signals.get(
                    "offer_acceptance_rate",
                    0
                )
            )
        )
        verified_email = safe_bool(
            candidate.get(
                "verified_email",
                signals.get(
                    "verified_email",
                    False
                )
            )
        )
        verified_phone = safe_bool(
            candidate.get(
                "verified_phone",
                signals.get(
                    "verified_phone",
                    False
                )
            )
        )
        linkedin_connected = safe_bool(
            candidate.get(
                "linkedin_connected",
                signals.get(
                    "linkedin_connected",
                    False
                )
            )
        )
        rows.append({
            "years_experience":
                years_experience,
            "education_tier":
                education_tier,
            "profile_completeness":
                profile_completeness,
            "open_to_work":
                open_to_work,
            "profile_views":
                profile_views,
            "applications":
                applications,
            "response_rate":
                response_rate,
            "avg_response_hours":
                avg_response_hours,
            "connection_count":
                connection_count,
            "endorsements":
                endorsements,
            "notice_period":
                notice_period,
            "github_activity":
                github_activity,
            "search_appearance":
                search_appearance,
            "saved_by_recruiters":
                saved_by_recruiters,
            "interview_completion":
                interview_completion,
            "offer_acceptance":
                offer_acceptance,
            "verified_email":
                verified_email,
            "verified_phone":
                verified_phone,
            "linkedin_connected":
                linkedin_connected,
            "assessment_average":
                assessment_average,
        })
    return pd.DataFrame(
        rows,
        columns=MODEL_FEATURES
    )
def load_jd():
    print(
        "Loading job description..."
    )
    if not JD_FILE.exists():
        raise FileNotFoundError(
            f"Missing JD file: {JD_FILE}"
        )
    with open(
        JD_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()
def load_jd_features():
    if JD_FEATURES_FILE.exists():
        print(
            "Loading precomputed JD features..."
        )
        try:
            with open(
                JD_FEATURES_FILE,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)
                if isinstance(
                    data,
                    dict
                ):
                    return data
        except Exception as e:
            print(
                f"Warning: could not load JD features: {e}"
            )
    print(
        "JD feature artifact not found."
    )
    print(
        "Using local JD skill configuration."
    )
    return {
        "skills":
            sorted(
                JD_SKILLS
            ),
        "min_experience":
            MIN_EXPERIENCE,
        "max_experience":
            MAX_EXPERIENCE,
    }
def parse_skills(value):
    if value is None:
        return set()
    if isinstance(
        value,
        (list, tuple, set)
    ):
        return {
            str(x).lower().strip()
            for x in value
            if str(x).strip()
        }
    if isinstance(
        value,
        str
    ):
        text = value.strip()
        if not text:
            return set()
        if (
            text.startswith("[")
            and
            text.endswith("]")
        ):
            try:
                parsed = json.loads(
                    text
                )
                if isinstance(
                    parsed,
                    list
                ):
                    return {
                        str(x)
                        .lower()
                        .strip()
                        for x in parsed
                    }
            except Exception:
                pass
        if "|" in text:
            return {
                x.strip().lower()
                for x in text.split("|")
                if x.strip()
            }
        if "," in text:
            return {
                x.strip().lower()
                for x in text.split(",")
                if x.strip()
            }
        return {
            text.lower()
        }
    return set()
def get_candidate_text(row):
    pieces = []
    for column in [
        "headline",
        "summary",
        "career_text",
        "experience_text",
        "skills_text",
        "current_title",
        "current_industry",
    ]:
        if column in row:
            value = row[column]
            if value is not None:
                text = str(
                    value
                ).strip()
                if text:
                    pieces.append(
                        text
                    )
    return " ".join(
        pieces
    ).lower()
def calculate_skill_match(
    row,
    jd_skills
):
    candidate_skills = set()
    if "normalized_skills" in row:
        candidate_skills.update(
            parse_skills(
                row["normalized_skills"]
            )
        )
    if "skills" in row:
        candidate_skills.update(
            parse_skills(
                row["skills"]
            )
        )
    text = get_candidate_text(
        row
    )
    if not candidate_skills:
        candidate_skills = {
            skill
            for skill in jd_skills
            if skill in text
        }
    normalized_jd = {
        str(skill)
        .lower()
        .strip()
        for skill in jd_skills
    }
    if not normalized_jd:
        return 0.0
    exact_matches = (
        candidate_skills
        &
        normalized_jd
    )
    score = (
        len(exact_matches)
        /
        len(normalized_jd)
    )
    if score == 0.0 and text:
        matched = 0
        for skill in normalized_jd:
            if skill in text:
                matched += 1
        score = (
            matched
            /
            len(normalized_jd)
        )
    return float(
        min(
            max(
                score,
                0.0
            ),
            1.0
        )
    )
def calculate_experience_match(
    years,
    min_exp,
    max_exp
):
    years = safe_float(
        years
    )
    if (
        min_exp
        <=
        years
        <=
        max_exp
    ):
        return 1.0
    if years < min_exp:
        difference = (
            min_exp - years
        )
    else:
        difference = (
            years - max_exp
        )
    return float(
        max(
            0.0,
            1.0 -
            difference / 5.0
        )
    )
def load_candidate_embeddings(
    number_of_candidates
):
    if not CANDIDATE_EMBEDDING_FILE.exists():
        print(
            "Candidate embedding artifact "
            "not found."
        )
        print(
            "Semantic similarity will be 0."
        )
        return None
    print(
        "Loading precomputed candidate embeddings..."
    )
    embeddings = np.load(
        CANDIDATE_EMBEDDING_FILE,
        mmap_mode="r"
    )
    print(
        f"Candidate embeddings shape: "
        f"{embeddings.shape}"
    )
    if len(embeddings) != number_of_candidates:
        raise ValueError(
            "Candidate embedding count does "
            "not match candidate count.\n"
            f"Candidates: {number_of_candidates}\n"
            f"Embeddings: {len(embeddings)}"
        )
    return embeddings
def load_jd_embedding():
    if not JD_EMBEDDING_FILE.exists():
        print(
            "JD embedding artifact not found."
        )
        print(
            "Semantic similarity disabled."
        )
        return None
    print(
        "Loading precomputed JD embedding..."
    )
    embedding = np.load(
        JD_EMBEDDING_FILE
    )
    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(-1)
    print(
        f"JD embedding dimension: "
        f"{len(embedding)}"
    )
    return embedding
def normalize_embedding_matrix(
    embeddings
):
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )
    norms[
        norms == 0
    ] = 1.0
    return (
        embeddings / norms
    )
def calculate_semantic_similarity(
    candidate_embeddings,
    jd_embedding
):
    if (
        candidate_embeddings is None
        or
        jd_embedding is None
    ):
        return np.zeros(
            len(candidate_embeddings)
            if candidate_embeddings is not None
            else 0,
            dtype=np.float32
        )
    candidate_embeddings = (
        np.asarray(
            candidate_embeddings,
            dtype=np.float32
        )
    )
    jd_embedding = (
        np.asarray(
            jd_embedding,
            dtype=np.float32
        )
        .reshape(-1)
    )
    if (
        candidate_embeddings.ndim != 2
    ):
        raise ValueError(
            "Candidate embeddings must "
            "be a 2D matrix."
        )
    if (
        candidate_embeddings.shape[1]
        !=
        len(jd_embedding)
    ):
        raise ValueError(
            "Embedding dimensions do not match.\n"
            f"Candidates: "
            f"{candidate_embeddings.shape[1]}\n"
            f"JD: "
            f"{len(jd_embedding)}"
        )
    jd_norm = np.linalg.norm(
        jd_embedding
    )
    if jd_norm == 0:
        return np.zeros(
            len(candidate_embeddings),
            dtype=np.float32
        )
    jd_embedding = (
        jd_embedding
        /
        jd_norm
    )
    candidate_norms = np.linalg.norm(
        candidate_embeddings,
        axis=1
    )
    candidate_norms[
        candidate_norms == 0
    ] = 1.0
    normalized_candidates = (
        candidate_embeddings
        /
        candidate_norms[:, None]
    )
    similarity = (
        normalized_candidates
        @
        jd_embedding
    )
    return similarity.astype(
        np.float32
    )
def build_matching_features(
    candidates,
    jd_features,
    semantic_scores
):
    print(
        "Building JD-candidate matching features..."
    )
    jd_skills = set(
        jd_features.get(
            "skills",
            JD_SKILLS
        )
    )
    min_exp = safe_float(
        jd_features.get(
            "min_experience",
            MIN_EXPERIENCE
        ),
        MIN_EXPERIENCE
    )
    max_exp = safe_float(
        jd_features.get(
            "max_experience",
            MAX_EXPERIENCE
        ),
        MAX_EXPERIENCE
    )
    skill_scores = np.zeros(
        len(candidates),
        dtype=np.float32
    )
    experience_scores = np.zeros(
        len(candidates),
        dtype=np.float32
    )
    for i, (_, row) in enumerate(
        candidates.iterrows()
    ):
        skill_scores[i] = (
            calculate_skill_match(
                row,
                jd_skills
            )
        )
        years = safe_float(
            row.get(
                "years_experience",
                row.get(
                    "years_of_experience",
                    0
                )
            )
        )
        experience_scores[i] = (
            calculate_experience_match(
                years,
                min_exp,
                max_exp
            )
        )
    return (
        skill_scores,
        experience_scores
    )
def load_ranker():
    print(
        "Loading trained ranker..."
    )
    if not RANKER_FILE.exists():
        raise FileNotFoundError(
            f"Missing trained model:\n"
            f"{RANKER_FILE}"
        )
    package = joblib.load(
        RANKER_FILE
    )
    if isinstance(
        package,
        dict
    ):
        if "model" not in package:
            raise ValueError(
                "ranker.pkl is a dictionary "
                "but has no 'model' key."
            )
        model = package["model"]
        features = package.get(
            "features",
            MODEL_FEATURES
        )
    else:
        model = package
        features = MODEL_FEATURES
    print(
        f"Model: {type(model).__name__}"
    )
    print(
        f"Model features: {features}"
    )
    return (
        model,
        list(features)
    )
def prepare_model_input(
    model_features,
    feature_columns
):
    missing = [
        column
        for column in feature_columns
        if column
        not in model_features.columns
    ]
    if missing:
        print(
            "Warning: missing model features:"
        )
        for column in missing:
            print(
                f"  {column} -> using 0"
            )
            model_features[
                column
            ] = 0.0
    X = model_features[
        feature_columns
    ].copy()
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )
    X = X.replace(
        [
            np.inf,
            -np.inf
        ],
        0
    )
    X = X.fillna(
        0.0
    )
    return X
def normalize_ml_score(
    predictions
):
    predictions = np.asarray(
        predictions,
        dtype=np.float32
    )
    predictions = np.clip(
        predictions,
        0.0,
        4.0
    )
    return (
        predictions / 4.0
    )
def behavioral_adjustment(
    model_features
):
    open_to_work = (
        model_features[
            "open_to_work"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )
    response_rate = (
        model_features[
            "response_rate"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )
    saved = (
        model_features[
            "saved_by_recruiters"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )
    saved_score = np.clip(
        saved / 10.0,
        0.0,
        1.0
    )
    behavior_score = (
        0.45 * open_to_work
        +
        0.35 * response_rate
        +
        0.20 * saved_score
    )
    return behavior_score.astype(
        np.float32
    )
def calculate_final_scores(
    ml_score,
    semantic_score,
    skill_score,
    experience_score,
    behavior_score
):
    final_score = (
        ML_WEIGHT
        *
        ml_score
        +
        SEMANTIC_WEIGHT
        *
        semantic_score
        +
        SKILL_WEIGHT
        *
        skill_score
        +
        EXPERIENCE_WEIGHT
        *
        experience_score
    )
    final_score = (
        0.95 * final_score
        +
        0.05 * behavior_score
    )
    return final_score.astype(
        np.float32
    )
def generate_reason(
    row,
    skill_score,
    experience_score,
    semantic_score
):
    years = safe_float(
        row.get(
            "years_experience",
            row.get(
                "years_of_experience",
                0
            )
        )
    )
    title = str(
        row.get(
            "current_title",
            ""
        )
    )
    if not title:
        title = "candidate"
    parts = []
    if experience_score >= 0.9:
        parts.append(
            f"{years:.1f} years of experience"
        )
    elif experience_score >= 0.6:
        parts.append(
            f"{years:.1f} years of experience with "
            "reasonable alignment to the role"
        )
    else:
        parts.append(
            f"{years:.1f} years of experience"
        )
    if skill_score >= 0.5:
        parts.append(
            "strong overlap with the required technical skills"
        )
    elif skill_score >= 0.25:
        parts.append(
            "moderate overlap with the required technical skills"
        )
    else:
        parts.append(
            "limited explicit skill overlap"
        )
    if semantic_score >= 0.65:
        parts.append(
            "high semantic similarity to the job description"
        )
    elif semantic_score >= 0.45:
        parts.append(
            "good semantic alignment with the job description"
        )
    return (
        f"{title}: "
        + "; ".join(parts)
        + "."
    )
def main():
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    print()
    print("=" * 70)
    print(
        "REDROB CANDIDATE RANKING SYSTEM"
    )
    print("=" * 70)
    print()
    print(
        "[1/8] Loading candidates..."
    )
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Missing candidate file:\n"
            f"{CANDIDATES_FILE}"
        )
    candidates = pd.read_parquet(
        CANDIDATES_FILE
    )
    print(
        f"Loaded {len(candidates):,} candidates"
    )
    if len(candidates) == 0:
        raise ValueError(
            "Candidate dataset is empty."
        )
    print()
    print(
        "[2/8] Loading job description..."
    )
    jd_text = load_jd()
    print(
        f"JD characters: {len(jd_text):,}"
    )
    print()
    print(
        "[3/8] Loading JD features..."
    )
    jd_features = (
        load_jd_features()
    )
    print(
        "JD skills:"
    )
    print(
        jd_features.get(
            "skills",
            []
        )
    )
    print()
    print(
        "[4/8] Loading precomputed embeddings..."
    )
    candidate_embeddings = (
        load_candidate_embeddings(
            len(candidates)
        )
    )
    jd_embedding = (
        load_jd_embedding()
    )
    print(
        "No embedding model is loaded."
    )
    print(
        "No network request is made."
    )
    print()
    print(
        "[5/8] Calculating semantic similarity..."
    )
    semantic_scores = (
        calculate_semantic_similarity(
            candidate_embeddings,
            jd_embedding
        )
    )
    if len(semantic_scores):
        print(
            "Semantic similarity calculated."
        )
        print(
            f"Min: {semantic_scores.min():.4f}"
        )
        print(
            f"Max: {semantic_scores.max():.4f}"
        )
        print(
            f"Mean: {semantic_scores.mean():.4f}"
        )
    else:
        semantic_scores = np.zeros(
            len(candidates),
            dtype=np.float32
        )
    print()
    print(
        "[6/8] Building model features..."
    )
    candidate_records = (
        candidates.to_dict(
            orient="records"
        )
    )
    model_features = (
        build_model_features(
            candidate_records
        )
    )
    print(
        f"Feature matrix: "
        f"{model_features.shape}"
    )
    print()
    print(
        "[7/8] Loading trained ranking model..."
    )
    ranker, feature_columns = (
        load_ranker()
    )
    X = prepare_model_input(
        model_features,
        feature_columns
    )
    print()
    print(
        "Predicting candidate relevance..."
    )
    predictions = ranker.predict(
        X
    )
    ml_scores = (
        normalize_ml_score(
            predictions
        )
    )
    print(
        "ML prediction completed."
    )
    print(
        f"ML score range: "
        f"{ml_scores.min():.4f} - "
        f"{ml_scores.max():.4f}"
    )
    print()
    print(
        "Calculating skill and experience matching..."
    )
    skill_scores, experience_scores = (
        build_matching_features(
            candidates,
            jd_features,
            semantic_scores
        )
    )
    behavior_scores = (
        behavioral_adjustment(
            model_features
        )
    )
    print()
    print(
        "Calculating final ranking score..."
    )
    final_scores = (
        calculate_final_scores(
            ml_scores,
            semantic_scores,
            skill_scores,
            experience_scores,
            behavior_scores
        )
    )
    results = pd.DataFrame({
        "candidate_id":
            candidates[
                "candidate_id"
            ].values,
        "ml_score":
            ml_scores,
        "semantic_score":
            semantic_scores,
        "skill_score":
            skill_scores,
        "experience_score":
            experience_scores,
        "behavior_score":
            behavior_scores,
        "final_score":
            final_scores,
    })
    if "current_title" in candidates.columns:
        results[
            "current_title"
        ] = candidates[
            "current_title"
        ].values
    if "years_experience" in candidates.columns:
        results[
            "years_experience"
        ] = candidates[
            "years_experience"
        ].values
    print()
    print(
        "Sorting candidates..."
    )
    results = (
        results
        .sort_values(
            "final_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )
    top_candidates = (
        results
        .head(
            TOP_K
        )
        .copy()
    )
    print()
    print(
        f"Selected top {len(top_candidates)} candidates."
    )
    print()
    print(
        "Generating deterministic explanations..."
    )
    candidate_lookup = (
        candidates
        .set_index(
            "candidate_id"
        )
    )
    reasons = []
    for _, result in (
        top_candidates.iterrows()
    ):
        candidate_id = (
            result[
                "candidate_id"
            ]
        )
        if candidate_id in candidate_lookup.index:
            row = candidate_lookup.loc[
                candidate_id
            ]
            reason = generate_reason(
                row,
                result[
                    "skill_score"
                ],
                result[
                    "experience_score"
                ],
                result[
                    "semantic_score"
                ]
            )
        else:
            reason = (
                "Candidate ranked using "
                "the learned relevance model "
                "and matching signals."
            )
        reasons.append(
            reason
        )
    top_candidates[
        "reason"
    ] = reasons
    ranking_output = (
        OUTPUT_DIR /
        "ranking_results.csv"
    )
    top_candidates.to_csv(
        ranking_output,
        index=False
    )
    submission = pd.DataFrame({
        "candidate_id":
            top_candidates[
                "candidate_id"
            ].values,
        "relevance":
            np.clip(
                np.rint(
                    top_candidates[
                        "ml_score"
                    ].values
                    *
                    4.0
                ),
                0,
                4
            ).astype(int)
    })
    submission.to_csv(
        OUTPUT_FILE,
        index=False
    )
    elapsed = (
        time.perf_counter()
        -
        start_time
    )
    print()
    print("=" * 70)
    print(
        "RANKING COMPLETED"
    )
    print("=" * 70)
    print(
        f"Candidates processed : "
        f"{len(candidates):,}"
    )
    print(
        f"Top candidates       : "
        f"{len(top_candidates)}"
    )
    print(
        f"Runtime              : "
        f"{elapsed:.2f} seconds"
    )
    print(
        f"Submission           : "
        f"{OUTPUT_FILE}"
    )
    print(
        f"Detailed ranking     : "
        f"{ranking_output}"
    )
    print()
    print(
        "Network calls: 0"
    )
    print(
        "LLM calls: 0"
    )
    print(
        "Embedding model calls: 0"
    )
    print()
    print(
        "Top 10 candidates:"
    )
    print(
        top_candidates[
            [
                "candidate_id",
                "final_score",
                "ml_score",
                "semantic_score",
                "skill_score",
                "experience_score"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )
    print()
    print(
        "Done."
    )
if __name__ == "__main__":
    main()
