from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMB_MODEL   = "all-MiniLM-L6-v2"
BATCH_SIZE  = 256

def build_index(chunks_parquet="data/chunks.parquet",
                index_path="data/index.faiss",
                meta_path="data/meta.parquet"):
    assert Path(chunks_parquet).exists(), "Falta data/chunks.parquet"

    model = SentenceTransformer(EMB_MODEL)
    print("[EMB] Cargando chunks (solo columnas necesarias)")
    df = pd.read_parquet(chunks_parquet, columns=["doc_id","title","url","vigencia","chunk_id","text"])

    dim = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    metas = []
    buf = []

    def _flush(embs):
        nonlocal index, buf
        embs = np.asarray(embs, dtype=np.float32)
        # normalizar para dot-product (IP)
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norms
        index.add(embs)
        buf.clear()

    print(f"[EMB] Total filas: {len(df)}")
    for start in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start:start+BATCH_SIZE]
        texts = batch["text"].tolist()
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        _flush(embs)
        metas.append(batch.drop(columns=["text"]))  # guardar metadatos sin el texto grande

    # guardar índice y metadatos
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    pd.concat(metas, ignore_index=True).to_parquet(meta_path, index=False)
    print(f"[OK] Index FAISS: {index.ntotal} vectores → {index_path}")
    print(f"[OK] Metadatos → {meta_path}")

if __name__ == "__main__":
    build_index()
