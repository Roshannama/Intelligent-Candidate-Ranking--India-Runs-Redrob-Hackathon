# Redrob Hackathon — Intelligent Candidate Discovery & Ranking System

An offline, CPU-efficient candidate discovery and ranking system designed to rank a large candidate pool against a fixed Job Description (JD).

The system combines:

- Structured candidate information
- Skill matching
- Career-history signals
- Semantic embeddings
- BM25 / lexical relevance
- Production experience
- Experience matching
- Redrob behavioral signals
- A LightGBM machine-learning model
- Deterministic explainability

The final ranking stage is designed to run with:

- CPU only
- Network OFF
- No external API calls
- No hosted LLM calls
- A target runtime of <= 5 minutes for approximately 100K candidates

---

## 1. Problem Statement

The goal is to rank approximately 100,000 candidates against a Job Description.

The system must identify candidates who are not only keyword matches, but who demonstrate actual relevance through their profile, career history, technical experience, and platform behavior.

For the provided Senior AI Engineer JD, important signals include:

### Required

- Python
- Embeddings
- Retrieval
- Ranking
- Vector databases
- Information retrieval
- Production AI / software experience

### Preferred

- LLM fine-tuning
- LoRA / QLoRA / PEFT
- Learning-to-rank
- Machine learning
- Deep learning
- HR-tech experience
- Distributed systems

### Negative / disqualifying signals described by the JD

- Pure research background
- Only recent LangChain/OpenAI exposure
- No production coding
- Consulting-only background
- CV/speech/robotics experience without NLP/IR relevance
- Closed-source-only experience where the JD requires relevant production evidence

The system therefore avoids relying on simple keyword counting.

---

# 2. Evaluation Metrics

The hackathon evaluation uses:

- NDCG@10 — weight 0.50
- NDCG@50 — weight 0.30
- MAP — weight 0.15
- P@10 — weight 0.05

Final score:

```text
Final =
    0.50 * NDCG@10
  + 0.30 * NDCG@50
  + 0.15 * MAP
  + 0.05 * P@10
```

The relevance scale is:

```text
0 = Not relevant
1 = Weak relevance
2 = Some / moderate relevance
3 = Strong relevance
4 = Excellent relevance
```

For P@10, a candidate is considered relevant when:

```text
relevance >= 3
```

---

# 3. Important Constraint

The final ranking stage must not depend on network access.

Therefore the final ranking process does NOT call:

```text
OpenAI
Anthropic
Gemini
Cohere
OpenRouter
Tavily
Any hosted LLM API
Any external embedding API
```

Instead, expensive processing is performed offline.

The final ranking process uses locally stored artifacts.

---

# 4. Architecture

```text
                         OFFLINE PIPELINE
                         ================

                    candidates.jsonl
                           |
                           v
                    preprocess.py
                           |
                           v
                  candidates.parquet
                           |
              +------------+-------------+
              |                          |
              v                          v
       Structured Features         Candidate Text
              |                          |
              |                          v
              |                 Local Embedding Model
              |                          |
              |                          v
              |                 candidate_embeddings.npy
              |                          |
              |                          v
              |                      FAISS Index
              |                          |
              +------------+-------------+
                           |
                           v
                    features.py
                           |
                           v
               ranking_features.parquet
                           |
                           v
                    labels.csv
                           |
                           v
                     train.py
                           |
                           v
                     ranker.pkl


                         FINAL RANKING
                         =============

                          Fixed JD
                            |
                            v
                    JD feature extraction
                            |
              +-------------+-------------+
              |                           |
              v                           v
        Skill matching              JD embedding
              |                           |
              +-------------+-------------+
                            |
                            v
                  Candidate feature matrix
                            |
                            v
                       LightGBM
                            |
                            v
                      Relevance score
                            |
                            v
                    Sort descending
                            |
                            v
                       Top candidates
                            |
                            v
                     submission.csv
```

---

# 5. Offline vs Final Ranking

The most important design decision is separating expensive processing from the final ranking stage.

## Offline stage

The following can be generated beforehand:

- Candidate normalized text
- Candidate embeddings
- FAISS index
- Candidate structured features
- JD embedding
- JD structured features
- Trained LightGBM model

These artifacts are saved locally.

## Final ranking stage

Only the following happens:

```text
Load artifacts
      |
      v
Load candidate features
      |
      v
Calculate final JD-dependent features
      |
      v
LightGBM prediction
      |
      v
Sort
      |
      v
Generate submission
```

No network call is needed.

---

# 6. Project Structure

```text
redrob-hackathon/
│
├── data/
│   ├── candidates.jsonl
│   ├── jd.txt
│   └── labels.csv
│
├── artifacts/
│   ├── candidates.parquet
│   ├── ranking_features.parquet
│   ├── candidate_embeddings.npy
│   ├── jd_embedding.npy
│   ├── faiss.index
│   ├── jd_features.json
│   └── ranker.pkl
│
├── outputs/
│   └── submission.csv
│
├── src/
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── features.py
│   ├── baseline_rank.py
│   ├── train.py
│   ├── rank.py
│   └── reasoning.py
│
├── config.py
├── requirements.txt
└── README.md
```

---

# 7. Input Files

## 7.1 candidates.jsonl

The candidate pool is supplied as JSONL.

A candidate can contain fields such as:

```json
{
  "candidate_id": "CAND_0000001",
  "profile": {
    "anonymized_name": "Ira Vora",
    "headline": "Backend Engineer | SQL, Spark, Cloud",
    "summary": "Backend and data engineering experience...",
    "location": "Toronto, Canada",
    "current_title": "Backend Engineer",
    "current_company": "Mindtree"
  },
  "career_history": [],
  "education": [],
  "skills": [],
  "redrob_signals": {}
}
```

The exact fields depend on the supplied hackathon dataset.

---

# 8. Job Description

Store the fixed JD in:

```text
data/jd.txt
```

Example:

```text
Senior AI Engineer — Founding Team

Location: Pune / Noida

Experience: 5–9 years

Required:
- Production embeddings
- Retrieval
- Vector databases
- Python
- Evaluation frameworks
- Information retrieval

Preferred:
- LLM fine-tuning
- Learning-to-rank
- HR tech
- Distributed systems

Disqualifiers:
- Pure research
- Only recent LangChain/OpenAI exposure
- No production coding
- Consulting-only
- CV/speech/robotics without NLP/IR
- Closed-source only
```

The exact JD provided by the hackathon should be used rather than a generic JD.

---

# 9. Labels

The hackathon does not require manually labeling all 100,000 candidates.

For supervised learning, create a smaller labeled subset.

Example:

```csv
candidate_id,relevance
CAND_000001,4
CAND_000002,1
CAND_000003,0
CAND_000004,3
CAND_000005,2
```

A practical initial target is approximately:

```text
500–2,000 labeled candidates
```

The final model can then score the complete candidate pool.

The labeled candidates should be representative of the whole candidate population.

---

# 10. Relevance Labeling Guidelines

Use the following interpretation consistently.

## Relevance 0

Candidate is essentially unrelated.

Examples:

- No relevant AI / ML / NLP / retrieval experience
- Completely unrelated career
- Does not satisfy the central technical requirements

## Relevance 1

Weak relevance.

Examples:

- Some AI/ML exposure
- Mostly unrelated production experience
- Relevant technologies only appear in minor projects

## Relevance 2

Moderate relevance.

Examples:

- Relevant AI/ML experience
- Some NLP/retrieval/embedding exposure
- Missing important production requirements

## Relevance 3

Strong relevance.

Examples:

- Relevant production AI experience
- Strong overlap with retrieval / embeddings / vector search
- Experience approximately matches the JD
- Demonstrated production coding

## Relevance 4

Excellent relevance.

Examples:

- Strong production AI/NLP/IR experience
- Embeddings + retrieval + ranking/vector databases
- Relevant experience level
- Production systems
- Strong evidence from career history
- Good match with the preferred requirements

---

# 11. Candidate Text Construction

For semantic matching, combine useful candidate information into a single text representation.

For example:

```text
Headline
+
Summary
+
Current title
+
Career history
+
Skills
+
Relevant project descriptions
```

Example:

```text
Backend Engineer | AI Retrieval

6.5 years of experience building production backend
and data systems.

Worked on semantic search, vector databases,
embeddings, Python and machine learning systems.

Skills:
Python, FastAPI, embeddings, Milvus, retrieval,
PyTorch, SQL
```

This text is then converted to an embedding.

---

# 12. Embedding Strategy

Use a local sentence-transformer model.

Conceptually:

```text
Candidate text
      |
      v
Sentence Transformer
      |
      v
Embedding vector
      |
      v
candidate_embeddings.npy
```

The same embedding approach can be used for the JD.

```text
JD text
      |
      v
Sentence Transformer
      |
      v
jd_embedding.npy
```

Similarity can then be calculated using cosine similarity.

---

# 13. Why Precompute Embeddings?

Generating an embedding for every candidate during final ranking is unnecessary if the candidate data is static.

Instead:

```text
100,000 candidates
       |
       v
Embedding model
       |
       v
100,000 embeddings
       |
       v
Save to disk
```

Then during ranking:

```text
Load embeddings
       |
       v
Similarity computation
```

This significantly reduces runtime.

---

# 14. FAISS

FAISS is used for fast vector similarity search.

Architecture:

```text
candidate_embeddings.npy
           |
           v
         FAISS
           |
           v
       faiss.index
```

FAISS can quickly retrieve semantically similar candidates.

For example:

```text
JD
 |
 v
JD embedding
 |
 v
FAISS
 |
 +--> Candidate 1042
 +--> Candidate 8312
 +--> Candidate 392
 +--> Candidate 912
 ...
```

---

# 15. Why Not ChromaDB?

ChromaDB is not required for this project.

The candidate data is static and the ranking task is an offline batch computation.

A lightweight architecture is sufficient:

```text
NumPy
+
FAISS
+
Parquet
+
LightGBM
```

A vector database server would add unnecessary infrastructure.

---

# 16. Retrieval

Retrieval reduces the search space before expensive ranking.

Possible retrieval signals:

### Dense retrieval

```text
Embedding similarity
```

### Sparse retrieval

```text
BM25
```

### Skill retrieval

```text
Required skill overlap
```

### Hybrid retrieval

```text
dense similarity
+
BM25
+
skill overlap
```

A hybrid retrieval score can be:

```text
retrieval_score =
    0.50 * dense_score
  + 0.30 * bm25_score
  + 0.20 * skill_score
```

The exact weights should be tuned using validation data.

---

# 17. Feature Engineering

The ranking model should receive multiple independent signals.

Recommended feature groups:

## Semantic features

```text
dense_score
bm25_score
```

## Skill features

```text
required_skill_match
preferred_skill_match
embedding_skill_match
retrieval_skill_match
ranking_skill_match
```

## Experience features

```text
years_experience
experience_match
production_experience
```

## Technical evidence

```text
production_score
retrieval_score
ranking_score
embedding_score
ml_score
fine_tuning_score
```

## Company / career features

```text
product_company_signal
consulting_only
company_count
recent_production_experience
```

## Behavioral features

```text
open_to_work
activity_score
response_rate
recruiter_interest
github_score
profile_quality
notice_score
```

## Location features

```text
location_match
relocation_match
work_mode_match
```

## Platform quality

```text
assessment_score
interview_completion
verified
```

---

# 18. Required Features

For the current JD, the core features should include:

```text
required_skill_match
preferred_skill_match
experience_match
production_score
retrieval_score
ranking_score
embedding_score
ml_score
dense_score
bm25_score
```

These should have significant influence because they directly measure technical relevance.

---

# 19. Behavioral Features

Redrob-specific signals can help distinguish candidates who are both relevant and engaged.

Examples:

```text
open_to_work
activity_score
response_rate
recruiter_interest
github_score
profile_quality
notice_score
assessment_score
interview_completion
verified
```

These should complement technical relevance.

They should not completely override a strong technical mismatch.

---

# 20. Experience Matching

The JD requires approximately:

```text
5–9 years
```

A feature can represent how well the candidate fits this interval.

Example:

```text
5–9 years -> 1.0

<5 years -> decreasing score

>9 years -> gradually decreasing score
```

The exact formula can be tuned using labeled data.

---

# 21. Production Experience

Production experience is important for the JD.

Potential evidence includes terms such as:

```text
production
deployed
deployment
real users
production system
serving
monitoring
scaling
latency
reliability
```

The system should search career/project descriptions rather than relying only on the skills list.

---

# 22. Retrieval Experience

Useful evidence includes:

```text
retrieval
semantic search
information retrieval
vector database
vector search
search engine
hybrid search
reranking
re-ranking
recommendation
```

---

# 23. Ranking Experience

Useful evidence includes:

```text
ranking
ranker
learning-to-rank
LTR
reranking
re-ranking
ranking model
```

---

# 24. Embedding Experience

Useful evidence includes:

```text
embedding
embeddings
sentence transformer
BGE
E5
vector representation
semantic similarity
```

---

# 25. LLM Fine-Tuning

Preferred signals include:

```text
fine-tuning
fine tuning
LoRA
QLoRA
PEFT
instruction tuning
parameter-efficient fine-tuning
```

These can be represented as additional features.

---

# 26. Consulting Signal

The JD explicitly distinguishes consulting-heavy profiles.

A simple feature can identify whether a candidate's career history consists exclusively of known consulting companies.

Example:

```text
consulting_only = 1
```

Otherwise:

```text
consulting_only = 0
```

This should be treated as one signal rather than an automatic rejection unless the JD explicitly requires that behavior.

---

# 27. Baseline Ranking

Before training ML, build a deterministic baseline.

Example:

```text
baseline_score =
    0.25 * required_skill_match
  + 0.15 * preferred_skill_match
  + 0.15 * experience_match
  + 0.15 * production_score
  + 0.10 * retrieval_score
  + 0.10 * ranking_score
  + 0.05 * embedding_score
  + 0.05 * activity_score
```

This baseline provides a reference point for evaluating the ML model.

---

# 28. Machine Learning Ranker

The first implementation can use:

```text
LightGBM
```

A simple initial model can be:

```python
LGBMRegressor(
    objective="regression",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5
)
```

The model predicts a continuous relevance score.

The candidates are then sorted by this score.

---

# 29. Important Model Detail

The initial implementation is a regression-based ranker.

It learns:

```text
features -> predicted relevance
```

Then:

```text
predicted relevance
        |
        v
sort descending
        |
        v
ranking
```

This is a practical baseline.

A later improvement is to use:

```text
LightGBM LGBMRanker
```

with a ranking objective such as:

```text
lambdarank
```

and candidate groups associated with each query/JD.

That would make the training objective more directly aligned with NDCG.

---

# 30. Training Data

The training dataset is produced by joining:

```text
ranking_features.parquet
```

with:

```text
labels.csv
```

using:

```text
candidate_id
```

Example:

```text
candidate_id
     |
     +-------------------+
     |                   |
     v                   v
candidate features    relevance
     |                   |
     +---------+---------+
               |
               v
            training
```

Only labeled candidates are used for supervised training.

All candidates can still be scored after the model is trained.

---

# 31. Train/Test Split

A basic validation split can be used during initial development.

Example:

```text
80% training
20% validation
```

The split should preserve relevance levels when enough examples exist.

However, because the actual evaluation is ranking-based, final model selection should also compare:

```text
NDCG@10
NDCG@50
MAP
P@10
```

rather than relying only on regression MAE.

---

# 32. Evaluation

Create a validation ranking:

```text
candidate_id
predicted_score
true_relevance
```

Sort by:

```text
predicted_score DESC
```

Then calculate:

```text
NDCG@10
NDCG@50
MAP
P@10
```

The final objective is the weighted hackathon metric.

---

# 33. Ablation Experiments

A strong way to evaluate the architecture is to test several versions.

## Experiment 1

```text
Skill matching only
```

## Experiment 2

```text
Skill matching
+
experience
+
production signals
```

## Experiment 3

```text
Structured features
+
BM25
```

## Experiment 4

```text
Structured features
+
BM25
+
dense embeddings
```

## Experiment 5

```text
All technical features
+
behavioral signals
+
LightGBM
```

For each experiment record:

```text
NDCG@10
NDCG@50
MAP
P@10
runtime
```

This demonstrates which components actually improve the ranking.

---

# 34. Runtime Optimization

The 100K-candidate constraint makes efficiency important.

## Avoid

```text
LLM call per candidate
API call per candidate
Embedding generation at ranking time
Repeated JSON parsing
Repeated model loading
Python-heavy nested loops
```

## Prefer

```text
Precomputed embeddings
Parquet
NumPy arrays
FAISS
Vectorized Pandas operations
Single model load
Batch LightGBM prediction
```

---

# 35. Memory Strategy

For 100K candidates:

```text
candidate embeddings
structured features
ranking model
```

should fit comfortably within the available machine memory if dimensions and dtypes are chosen appropriately.

Use:

```python
float32
```

for embeddings when possible.

Avoid unnecessarily keeping multiple copies of large arrays in memory.

---

# 36. Deterministic Ranking

The final ranking should be deterministic.

Given:

```text
same candidates
+
same JD
+
same artifacts
```

the output should be the same.

Set fixed random seeds where applicable:

```python
random_state=42
```

This improves reproducibility.

---

# 37. Explainability

The system can provide deterministic reasons for a candidate's ranking.

Example:

```text
Candidate: CAND_000123

Reasons:
- 7.1 years of experience
- Strong retrieval experience
- Strong embedding experience
- Production AI systems
- High required-skill overlap
- Recently active
- Open to work
```

This can be produced without an LLM.

Example logic:

```python
reasons = []

if experience_match >= 0.8:
    reasons.append("experience aligns with the JD")

if required_skill_match >= 0.75:
    reasons.append("strong required-skill overlap")

if retrieval_score >= 0.7:
    reasons.append("strong retrieval experience")

if embedding_score >= 0.7:
    reasons.append("strong embedding experience")

if production_score >= 0.7:
    reasons.append("production experience")

if open_to_work:
    reasons.append("open to work")
```

---

# 38. Recommended Source Files

## preprocess.py

Responsible for:

```text
JSONL
  ->
normalized candidate records
  ->
Parquet
```

Do not put ranking logic here.

---

## embeddings.py

Responsible for:

```text
candidate text
  ->
local embedding model
  ->
candidate_embeddings.npy
```

It should also create the FAISS index if that responsibility is assigned here.

---

## retrieval.py

Responsible for:

```text
JD
  ->
candidate retrieval
  ->
dense score
  ->
BM25 score
```

---

## features.py

Responsible for:

```text
candidate data
+
JD
+
retrieval scores
+
Redrob signals
        |
        v
ranking features
```

It should NOT train the model.

---

## baseline_rank.py

Responsible for:

```text
hand-designed feature weights
        |
        v
baseline score
        |
        v
baseline ranking
```

---

## train.py

Responsible for:

```text
features
+
labels
        |
        v
LightGBM training
        |
        v
ranker.pkl
```

---

## rank.py

Responsible for the final ranking.

It should:

1. Load the trained model.
2. Load the exact model feature list.
3. Load candidate features.
4. Construct features in exactly the same way used during training.
5. Validate that all required columns exist.
6. Predict relevance scores.
7. Sort candidates.
8. Generate `submission.csv`.

---

## reasoning.py

Responsible for deterministic explanations.

It should not make external API calls.

---

# 39. Critical Training/Ranking Consistency Rule

The most important implementation rule is:

> The feature columns used during training must be exactly the feature columns supplied to the model during ranking.

For example, if training uses:

```text
dense_score
bm25_score
required_skill_match
preferred_skill_match
experience_match
production_score
retrieval_score
ranking_score
embedding_score
ml_score
```

then `rank.py` must construct those same columns.

Do not train on:

```text
years_experience
profile_completeness
github_activity
```

and then try to predict using a completely different feature set.

---

# 40. Model Artifact

The trained model should be saved together with its feature list.

Example:

```python
model_package = {
    "model": model,
    "features": feature_columns
}
```

Save using:

```python
joblib.dump(
    model_package,
    "artifacts/ranker.pkl"
)
```

Then ranking loads:

```python
package = joblib.load(
    "artifacts/ranker.pkl"
)

model = package["model"]
feature_columns = package["features"]
```

This prevents feature-order and feature-name mismatches.

---

# 41. Final Ranking Flow

The final `rank.py` should conceptually perform:

```text
Load ranker
     |
     v
Load ranking features
     |
     v
Read feature list from ranker artifact
     |
     v
Check every feature exists
     |
     v
Create X in exact feature order
     |
     v
model.predict(X)
     |
     v
Attach score
     |
     v
Sort descending
     |
     v
Assign rank
     |
     v
Save submission.csv
```

---

# 42. Common Error to Avoid

Do NOT do:

```python
model.predict(df)
```

if `df` contains unrelated columns or is missing training columns.

Instead:

```python
X = df[feature_columns]
predictions = model.predict(X)
```

where:

```python
feature_columns
```

comes directly from the trained model artifact.

---

# 43. Another Common Error

Do not confuse:

```text
feature engineering
```

with:

```text
model training
```

`features.py` creates:

```text
X
```

`train.py` learns:

```text
X -> relevance
```

`rank.py` performs:

```text
X -> predicted relevance
```

---

# 44. Full Development Pipeline

```bash
python src/preprocess.py
```

```bash
python src/embeddings.py
```

```bash
python src/retrieval.py
```

```bash
python src/features.py
```

```bash
python src/baseline_rank.py
```

```bash
python src/train.py
```

```bash
python src/rank.py
```

Final file:

```text
outputs/submission.csv
```

---

# 45. Requirements

Create `requirements.txt`:

```text
numpy
pandas
pyarrow
scikit-learn
sentence-transformers
faiss-cpu
lightgbm
joblib
rank-bm25
tqdm
```

Install:

```bash
pip install -r requirements.txt
```

---

# 46. Suggested Environment

Python:

```text
Python 3.10+
```

Recommended:

```text
64-bit Python
```

because the project may process large datasets and embedding matrices.

---

# 47. Initial Development Order

Do not try to implement everything simultaneously.

Build in this order:

```text
1. Load candidates
       |
2. Preprocess
       |
3. Save Parquet
       |
4. Build candidate text
       |
5. Generate embeddings
       |
6. Build FAISS index
       |
7. Implement JD skill extraction
       |
8. Implement BM25
       |
9. Implement feature engineering
       |
10. Build baseline
       |
11. Create labeled subset
       |
12. Train LightGBM
       |
13. Evaluate NDCG/MAP/P@10
       |
14. Fix feature consistency
       |
15. Optimize runtime
       |
16. Generate final submission
```

---

# 48. Development Philosophy

The first version should be simple and measurable.

Do not immediately add:

```text
multi-agent LLM system
complex orchestration
external APIs
large local LLM
multiple vector databases
microservices
```

The actual task is a ranking problem.

The strongest architecture should therefore remain centered around:

```text
Retrieval
+
Feature Engineering
+
Learning-to-Rank
+
Evaluation
```

---

# 49. Possible Future Improvements

After the baseline works, improvements can include:

### Learning-to-Rank

Replace:

```text
LGBMRegressor
```

with:

```text
LGBMRanker
```

using a ranking objective.

### Better semantic retrieval

Experiment with local embedding models such as:

```text
BGE
E5
MiniLM
```

subject to the final environment and runtime constraints.

### Better lexical retrieval

Improve BM25 preprocessing with:

```text
skill normalization
synonyms
abbreviations
technical phrase handling
```

### Better feature interactions

Add:

```text
production × retrieval
experience × skill match
recent activity × open-to-work
technical relevance × behavioral relevance
```

### Better validation

Use:

```text
NDCG@10
NDCG@50
MAP
P@10
```

during model selection.

---

# 50. Why This Architecture Fits the Hackathon

The system addresses the main constraints directly.

## 100K candidates

Use:

```text
Parquet
NumPy
FAISS
LightGBM
```

instead of expensive per-candidate LLM calls.

## CPU-only

LightGBM and FAISS are CPU-friendly.

## Network OFF

All required artifacts are stored locally.

## <=5 minutes

Candidate embeddings and other expensive operations are precomputed.

## Ranking quality

Use multiple independent signals:

```text
semantic relevance
+
skill relevance
+
experience
+
production evidence
+
behavioral signals
```

## Explainability

Use deterministic feature-based explanations.

---

# 51. Final Architecture Summary

```text
                     +----------------------+
                     |    candidates.jsonl  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |     Preprocessing     |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |  Candidate Features  |
                     +----------+-----------+
                                |
               +----------------+----------------+
               |                                 |
               v                                 v
      +------------------+             +------------------+
      | Local Embeddings |             | Structured Data  |
      +--------+---------+             +--------+---------+
               |                                |
               v                                |
      +------------------+                      |
      |      FAISS       |                      |
      +--------+---------+                      |
               |                                |
               +----------------+---------------+
                                |
                                v
                     +----------------------+
                     | Feature Engineering  |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |   Labeled Examples   |
                     +----------+-----------+
                                |
                                v
                     +----------------------+
                     |      LightGBM        |
                     +----------+-----------+
                                |
                                v
                         ranker.pkl


                       FINAL RANKING

                           JD
                           |
                           v
                  +------------------+
                  | JD Feature Build |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Candidate Matrix |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | LightGBM Predict |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Sort by Score    |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | submission.csv   |
                  +------------------+
```

---

# 52. Final Takeaway

The project is fundamentally a:

```text
Large-scale Candidate Retrieval
+
Feature Engineering
+
Learning-to-Rank
```

system.

The main principle is:

```text
Do expensive work once.
Store the results.
Perform fast numerical ranking at runtime.
```

The final runtime should therefore look like:

```text
Load artifacts
      ↓
Load candidate features
      ↓
Build JD-dependent features
      ↓
LightGBM prediction
      ↓
Sort
      ↓
Generate submission.csv
```

No external API is necessary during final ranking.
