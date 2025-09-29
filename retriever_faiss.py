import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class RetrieverFAISS:
    def __init__(self,
                 faiss_path="index.faiss",
                 meta_path="meta.jsonl",
                 model="sentence-transformers/all-MiniLM-L6-v2"):
        assert os.path.exists(faiss_path), f"No existe {faiss_path}. Corre build_faiss.py"
        assert os.path.exists(meta_path), f"No existe {meta_path}. Corre build_faiss.py"

        self.index = faiss.read_index(faiss_path)
        self.model = SentenceTransformer(model)

        # carga metadatos a memoria (ligero)
        self.metas = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                self.metas.append(json.loads(line))

    def search(self, query: str, k: int = 8):
        qv = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(qv, k)  # similitudes IP
        idxs = I[0]
        sims = D[0]
        out = []
        for i, s in zip(idxs, sims):
            if i < 0:
                continue
            m = self.metas[int(i)]
            out.append({
                "title": m["title"],
                "text": m["text"],
                "score": float(s)
            })
        return out
