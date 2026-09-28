import sys
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from rank_bm25 import BM25Okapi
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CANDIDATES_PARQUET,FAISS_INDEX_FILE,JD_EMBEDDING_FILE,JD_FILE,RANKING_FEATURES_FILE
def normalize_scores(scores):
    scores = np.asarray(
        scores,
        dtype=np.float32
    )
    minimum = scores.min()
    maximum = scores.max()
    if maximum == minimum:
        return np.zeros_like(scores)
    return (
        (scores - minimum)
        / (maximum - minimum)
    )
def dense_scores():
    df = pd.read_parquet(
        CANDIDATES_PARQUET
    )
    embeddings = np.load(
        "artifacts/candidate_embeddings.npy"
    )
    jd_embedding = np.load(
        JD_EMBEDDING_FILE
    )
    index = faiss.read_index(
        str(FAISS_INDEX_FILE)
    )
    scores = embeddings @ jd_embedding
    return normalize_scores(scores)
def bm25_scores():
    df = pd.read_parquet(
        CANDIDATES_PARQUET
    )
    documents = (
        df["search_text"]
        .fillna("")
        .str.lower()
        .str.split()
        .tolist()
    )
    bm25 = BM25Okapi(
        documents
    )
    jd_text = JD_FILE.read_text(
        encoding="utf-8"
    )
    query = jd_text.lower().split()
    scores = bm25.get_scores(
        query
    )
    return normalize_scores(scores)
def main():
    print("Calculating dense scores...")
    dense = dense_scores()
    print("Calculating BM25 scores...")
    bm25 = bm25_scores()
    df = pd.read_parquet(
        CANDIDATES_PARQUET
    )
    df["dense_score"] = dense
    df["bm25_score"] = bm25
    df.to_parquet(
        RANKING_FEATURES_FILE,
        index=False
    )
    print(
        "Saved ranking features:"
    )
    print(
        RANKING_FEATURES_FILE
    )
if __name__ == "__main__":
    main()