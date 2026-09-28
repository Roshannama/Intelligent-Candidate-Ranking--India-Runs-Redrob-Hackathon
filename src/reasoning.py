def generate_reason(candidate):
    years = candidate.get(
        "years_of_experience",
        0
    )
    title = candidate.get(
        "current_title",
        ""
    )
    company = candidate.get(
        "current_company",
        ""
    )
    skills = candidate.get(
        "normalized_skills",
        []
    )
    if isinstance(skills, str):
        skills = skills.split(",")
    skills_text = ", ".join(
        [str(s) for s in skills[:6]]
    )
    return (
        f"{years} years of experience as "
        f"{title} at {company}. "
        f"Relevant skills include {skills_text}."
    )