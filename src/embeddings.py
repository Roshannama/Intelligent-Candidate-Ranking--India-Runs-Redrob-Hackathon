import sys
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CANDIDATES_PARQUET,EMBEDDINGS_FILE,FAISS_INDEX_FILE,JD_FILE,JD_EMBEDDING_FILE,EMBEDDING_MODEL,EMBEDDING_BATCH_SIZE,ARTIFACT_DIR
def build_candidate_embeddings():
    df = pd.read_parquet(
        CANDIDATES_PARQUET
    )
    texts = (
        df["headline"].fillna("")
        + " "
        + df["summary"].fillna("")
        + " "
        + df["career_text"].fillna("")
    ).tolist()
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu"
    )
    print("Generating candidate embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    embeddings = embeddings.astype(
        "float32"
    )
    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )
    print(
        "Embedding shape:",
        embeddings.shape
    )
    return embeddings, model
def build_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(
        dimension
    )
    index.add(embeddings)
    faiss.write_index(
        index,
        str(FAISS_INDEX_FILE)
    )
    print(
        "FAISS index size:",
        index.ntotal
    )
def build_jd_embedding(model):
    jd_text = JD_FILE.read_text(
        encoding="utf-8"
    )
    jd_embedding = model.encode(
        [jd_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].astype("float32")
    np.save(
        JD_EMBEDDING_FILE,
        jd_embedding
    )
    print("JD embedding saved.")
def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    embeddings, model = (
        build_candidate_embeddings()
    )
    build_faiss(embeddings)
    build_jd_embedding(model)
if __name__ == "__main__":
    main()