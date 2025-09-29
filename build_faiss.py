from pathlib import Path
import os, json
import numpy as np
import faiss

def main(index_dir="index", out_path="index.faiss", meta_out="meta.jsonl"):
    idx_dir = Path(index_dir)
    emb_path = idx_dir / "embeddings.npz"
    chunks_path = idx_dir / "chunks.jsonl"
    assert emb_path.exists() and chunks_path.exists(), "Faltan embeddings.npz o chunks.jsonl (corre chunk_embed.py ingest)"

    # carga embeddings
    E = np.load(str(emb_path))["E"].astype("float32")  # [N, d]
    # normaliza para producto interno (IP)
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    E = E / norms

    # crea índice FAISS
    d = E.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(E)

    # guarda índice
    faiss.write_index(index, out_path)

    # copia/estandariza metadatos para consulta ligera
    # (dejamos un jsonl plano con {title,text} para no depender del path completo)
    metas = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            fname = os.path.basename(m["path"])
            title = fname.replace(".pdf","").replace(".txt","").replace("_"," ").title()
            metas.append({"title": title, "text": m["text"]})

    with open(meta_out, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"✅ FAISS listo: {out_path} | metadatos: {meta_out} | vectores: {index.ntotal}")

if __name__ == "__main__":
    main()
