from pathlib import Path
import json, gc
from typing import Iterable, Dict, List
from pypdf import PdfReader
import pandas as pd

CHUNK_CHARS = 1500   # tamaño aprox por caracteres
OVERLAP     = 200
BATCH_SIZE  = 500    # cuántos chunks escribir por vez

def _yield_pdf_text(path: Path) -> Iterable[str]:
    reader = PdfReader(str(path))
    for page in reader.pages:
        t = page.extract_text() or ""
        t = " ".join(t.split())
        if t:
            yield t

def _yield_txt_text(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        buf = []
        for line in f:
            buf.append(line.strip())
            if len(buf) >= 400:  # vaciar cada cierto número de líneas
                yield " ".join(buf)
                buf = []
        if buf:
            yield " ".join(buf)

def _chunk_stream(text: str, size=CHUNK_CHARS, overlap=OVERLAP) -> Iterable[str]:
    n = len(text)
    if n == 0:
        return
    start = 0
    while start < n:
        end = min(start + size, n)
        c = text[start:end].strip()
        if c:
            yield c
        start = max(0, end - overlap)

def _load_sources(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig","utf-8","cp1252","latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()

def ingest_docs(
    docs_dir="data/docs",
    sources_csv="data/sources.csv",
    out_jsonl="data/chunks.jsonl",
    out_parquet="data/chunks.parquet"
):
    docs_dir = Path(docs_dir)
    assert docs_dir.exists(), f"No existe {docs_dir}"

    sources = _load_sources(Path(sources_csv))

    # limpiar archivos previos
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    if Path(out_jsonl).exists():
        Path(out_jsonl).unlink()

    total_chunks = 0
    batch: List[Dict] = []

    print(f"[INGEST] Leyendo {docs_dir.resolve()}")

    for path in list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.txt")):
        # metadatos opcionales desde sources.csv
        meta = {}
        if not sources.empty and "filename" in sources.columns:
            m = sources[sources["filename"] == path.name]
            if len(m):
                meta = m.iloc[0].to_dict()

        base = {
            "doc_id": path.stem,
            "title": meta.get("title", path.stem),
            "url": meta.get("url", ""),
            "vigencia": meta.get("vigencia", ""),
        }

        # stream por bloques
        blocks = _yield_pdf_text(path) if path.suffix.lower()==".pdf" else _yield_txt_text(path)

        chunk_i = 0
        for block in blocks:
            for ch in _chunk_stream(block):
                row = dict(base)
                row["chunk_id"] = f"{path.stem}-{chunk_i}"
                row["text"] = ch
                batch.append(row)
                chunk_i += 1
                total_chunks += 1

                if len(batch) >= BATCH_SIZE:
                    with open(out_jsonl, "a", encoding="utf-8") as f:
                        for r in batch:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    batch.clear()
                    gc.collect()

        print(f"[OK] {path.name}: {chunk_i} chunks")

    # flush final
    if batch:
        with open(out_jsonl, "a", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        batch.clear()

    # convertir jsonl → parquet (columnar y compacto)
    # se hace en streaming de nuevo
    records = []
    with open(out_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            records.append(json.loads(line))
            if len(records) >= 5000:
                df = pd.DataFrame(records)
                mode = "w" if not Path(out_parquet).exists() else "a"
                df.to_parquet(out_parquet, index=False, engine="pyarrow", compression="zstd", append=(mode=="a"))
                records.clear()
    if records:
        df = pd.DataFrame(records)
        df.to_parquet(out_parquet, index=False, engine="pyarrow", compression="zstd", append=Path(out_parquet).exists())

    print(f"[OK] Total chunks: {total_chunks} → {out_parquet}")

if __name__ == "__main__":
    ingest_docs()
