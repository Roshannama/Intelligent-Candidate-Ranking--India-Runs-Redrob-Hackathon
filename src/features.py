import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
from datetime import datetime
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)
from config import RANKING_FEATURES_FILE,RANKING_DATE
REQUIRED_SKILLS = {
    "python",
    "embedding",
    "embeddings",
    "retrieval",
    "ranking",
    "vector_database",
    "information_retrieval",
}
PREFERRED_SKILLS = {
    "llm_finetuning",
    "lora",
    "qlora",
    "peft",
    "learning_to_rank",
    "machine_learning",
    "deep_learning",
}
def skill_set(value):
    if not isinstance(value, str):
        return set()
    return {
        x.strip().lower()
        for x in value.split("|")
        if x.strip()
    }
def keyword_score(text, keywords):
    text = str(text).lower()
    matched = sum(
        1
        for keyword in keywords
        if keyword in text
    )
    return matched / max(
        len(keywords),
        1
    )
def calculate_features(row):
    skills = skill_set(
        row["skills"]
    )
    career_text = (
        str(row["career_text"])
        + " "
        + str(row["headline"])
        + " "
        + str(row["summary"])
    ).lower()
    required_match = len(
        skills.intersection(
            REQUIRED_SKILLS
        )
    ) / len(REQUIRED_SKILLS)
    preferred_match = len(
        skills.intersection(
            PREFERRED_SKILLS
        )
    ) / len(PREFERRED_SKILLS)
    years = float(
        row["years_experience"]
    )
    if 5 <= years <= 9:
        experience_match = 1.0
    elif years < 5:
        experience_match = years / 5
    else:
        experience_match = max(
            0,
            1 - (years - 9) / 10
        )
    production_score = keyword_score(
        career_text,
        [
            "production",
            "deployed",
            "real users",
            "production system",
            "production systems"
        ]
    )
    retrieval_score = keyword_score(
        career_text,
        [
            "retrieval",
            "search",
            "information retrieval",
            "vector database",
            "semantic search",
            "recommendation"
        ]
    )
    ranking_score = keyword_score(
        career_text,
        [
            "ranking",
            "ranker",
            "learning to rank",
            "reranking",
            "re-ranking"
        ]
    )
    embedding_score = keyword_score(
        career_text,
        [
            "embedding",
            "embeddings",
            "sentence transformer",
            "bge",
            "e5"
        ]
    )
    ml_score = keyword_score(
        career_text,
        [
            "machine learning",
            "machine-learning",
            "ml model",
            "model training"
        ]
    )
    consulting_companies = {
        "tcs",
        "infosys",
        "wipro",
        "accenture",
        "cognizant",
        "capgemini"
    }
    companies = str(
        row["career_companies"]
    ).lower()
    consulting_only = int(
        companies
        and all(
            company in consulting_companies
            for company in companies.split("|")
            if company.strip()
        )
    )
    product_company_signal = 1 - consulting_only
    open_to_work = float(
        row["open_to_work"]
    )
    response_rate = float(
        row["response_rate"]
    )
    recruiter_interest = np.log1p(
        float(row["saved_by_recruiters"])
    )
    github = float(
        row["github_activity"]
    )
    if github < 0:
        github = 0
    github /= 100.0
    profile_quality = (
        float(row["profile_completeness"])
        / 100.0
    )
    notice = float(
        row["notice_period"]
    )
    if notice <= 30:
        notice_score = 1.0
    elif notice <= 60:
        notice_score = 0.7
    elif notice <= 90:
        notice_score = 0.4
    else:
        notice_score = 0.1
    location = str(
        row["location"]
    ).lower()
    location_match = int(
        any(
            city in location
            for city in [
                "pune",
                "noida",
                "delhi",
                "mumbai",
                "hyderabad"
            ]
        )
    )
    try:
        last_active = datetime.fromisoformat(
            str(row["last_active_date"])
        )
        reference = datetime.fromisoformat(
            RANKING_DATE
        )
        days_inactive = (
            reference - last_active
        ).days
    except:
        days_inactive = 999
    if days_inactive <= 30:
        activity_score = 1.0
    elif days_inactive <= 90:
        activity_score = 0.7
    elif days_inactive <= 180:
        activity_score = 0.4
    else:
        activity_score = 0.1
    return {
        "dense_score":
            row["dense_score"],
        "bm25_score":
            row["bm25_score"],
        "required_skill_match":
            required_match,
        "preferred_skill_match":
            preferred_match,
        "experience_match":
            experience_match,
        "production_score":
            production_score,
        "retrieval_score":
            retrieval_score,
        "ranking_score":
            ranking_score,
        "embedding_score":
            embedding_score,
        "ml_score":
            ml_score,
        "product_company_signal":
            product_company_signal,
        "consulting_only":
            consulting_only,
        "open_to_work":
            open_to_work,
        "activity_score":
            activity_score,
        "response_rate":
            response_rate,
        "recruiter_interest":
            recruiter_interest,
        "github_score":
            github,
        "profile_quality":
            profile_quality,
        "notice_score":
            notice_score,
        "location_match":
            location_match,
        "assessment_score":
            float(row["assessment_average"] / 100.0),
        "interview_completion":
            float(row["interview_completion"]),
        "verified":
            (
                float(row["verified_email"])
                +
                float(row["verified_phone"])
            ) / 2.0
    }
def main():
    df = pd.read_parquet(
        RANKING_FEATURES_FILE
    )
    features = []
    for _, row in df.iterrows():
        features.append(
            calculate_features(row)
        )
    feature_df = pd.DataFrame(
        features
    )
    feature_df.insert(
        0,
        "candidate_id",
        df["candidate_id"].values
    )
    feature_df["name"] = df["name"].values
    feature_df["current_title"] = (
        df["current_title"].values
    )
    feature_df["years_experience"] = (
        df["years_experience"].values
    )
    feature_df["skills"] = df["skills"].values
    feature_df.to_parquet(
        RANKING_FEATURES_FILE,
        index=False
    )
    print(
        "Final feature matrix:",
        feature_df.shape
    )
if __name__ == "__main__":
    main()