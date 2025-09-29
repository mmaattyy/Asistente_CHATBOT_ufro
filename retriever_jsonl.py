import os, json, numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_dir="index", model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)
        self.embeddings = np.load(os.path.join(index_dir, "embeddings.npz"))["E"]
        with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
            self.metas = [json.loads(l) for l in f]

    def search(self, query, k=5):
        qv = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.embeddings @ qv
        idx = np.argsort(-sims)[:k]
        results = []
        for i in idx:
            m = self.metas[int(i)]
            # usar título en limpio (nombre del archivo sin extensión ni guiones bajos)
            fname = os.path.basename(m["path"])
            title = fname.replace(".pdf","").replace(".txt","").replace("_"," ").title()
            results.append({
                "score": float(sims[i]),
                "title": title,
                "text": m["text"]
            })
        return results
