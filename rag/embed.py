from sentence_transformers import SentenceTransformer

def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)
