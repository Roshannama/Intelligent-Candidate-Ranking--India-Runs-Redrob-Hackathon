import sys
from pathlib import Path
import json
import re
import pandas as pd
from tqdm import tqdm
import orjson
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CANDIDATES_FILE,CANDIDATES_PARQUET,ARTIFACT_DIR
SKILL_ALIASES = {
    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "llm fine tuning": "llm_finetuning",
    "fine-tuning llms": "llm_finetuning",
    "fine tuning llms": "llm_finetuning",
    "large language models": "llm",
    "sentence transformers": "sentence_transformers",
    "vector database": "vector_database",
    "vector databases": "vector_database",
    "machine learning": "machine_learning",
    "deep learning": "deep_learning",
    "natural language processing": "nlp",
    "information retrieval": "information_retrieval",
    "learning to rank": "learning_to_rank",
}
def normalize_skill(skill):
    skill = skill.lower().strip()
    skill = re.sub(r"\s+", " ", skill)
    return SKILL_ALIASES.get(skill, skill)
def process_candidate(candidate):
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    normalized_skills = [
        normalize_skill(s.get("name", ""))
        for s in skills
        if s.get("name")
    ]
    career_text = []
    companies = []
    titles = []
    industries = []
    for job in history:
        titles.append(job.get("title", ""))
        companies.append(job.get("company", ""))
        industries.append(job.get("industry", ""))
        career_text.append(
            job.get("description", "")
        )
    education_tier = 5
    if education:
        tier = education[0].get("tier", "tier_5")
        try:
            education_tier = int(
                tier.replace("tier_", "")
            )
        except:
            education_tier = 5
    assessment_scores = signals.get(
        "skill_assessment_scores", {}
    )
    assessment_average = 0.0
    if assessment_scores:
        assessment_average = sum(
            assessment_scores.values()
        ) / len(assessment_scores)
    search_text = " ".join([
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(titles),
        " ".join(career_text),
        " ".join(normalized_skills)
    ])
    return {
        "candidate_id":
            candidate.get("candidate_id"),
        "name":
            profile.get("anonymized_name", ""),
        "headline":
            profile.get("headline", ""),
        "summary":
            profile.get("summary", ""),
        "location":
            profile.get("location", ""),
        "country":
            profile.get("country", ""),
        "years_experience":
            profile.get("years_of_experience", 0),
        "current_title":
            profile.get("current_title", ""),
        "current_company":
            profile.get("current_company", ""),
        "current_company_size":
            profile.get("current_company_size", ""),
        "current_industry":
            profile.get("current_industry", ""),
        "education_tier":
            education_tier,
        "skills":
            "|".join(normalized_skills),
        "career_titles":
            "|".join(titles),
        "career_companies":
            "|".join(companies),
        "career_industries":
            "|".join(industries),
        "career_text":
            " ".join(career_text),
        "search_text":
            search_text,
        "profile_completeness":
            signals.get("profile_completeness_score", 0),
        "last_active_date":
            signals.get("last_active_date"),
        "open_to_work":
            int(signals.get("open_to_work_flag", False)),
        "profile_views":
            signals.get("profile_views_received_30d", 0),
        "applications":
            signals.get("applications_submitted_30d", 0),
        "response_rate":
            signals.get("recruiter_response_rate", 0),
        "avg_response_hours":
            signals.get("avg_response_time_hours", 999),
        "connection_count":
            signals.get("connection_count", 0),
        "endorsements":
            signals.get("endorsements_received", 0),
        "notice_period":
            signals.get("notice_period_days", 180),
        "github_activity":
            signals.get("github_activity_score", -1),
        "search_appearance":
            signals.get("search_appearance_30d", 0),
        "saved_by_recruiters":
            signals.get("saved_by_recruiters_30d", 0),
        "interview_completion":
            signals.get("interview_completion_rate", 0),
        "offer_acceptance":
            signals.get("offer_acceptance_rate", 0),
        "verified_email":
            int(signals.get("verified_email", False)),
        "verified_phone":
            int(signals.get("verified_phone", False)),
        "linkedin_connected":
            int(signals.get("linkedin_connected", False)),
        "assessment_average":
            assessment_average,
    }
def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    rows = []
    with open(CANDIDATES_FILE, "rb") as f:
        for line in tqdm(
            f,
            desc="Processing candidates"
        ):
            line = line.strip()
            if not line:
                continue
            candidate = orjson.loads(line)
            rows.append(
                process_candidate(candidate)
            )
    df = pd.DataFrame(rows)
    print("Candidates:", len(df))
    df.to_parquet(
        CANDIDATES_PARQUET,
        index=False
    )
    print(
        f"Saved: {CANDIDATES_PARQUET}"
    )
if __name__ == "__main__":
    main()