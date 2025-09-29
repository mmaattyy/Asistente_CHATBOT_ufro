import argparse
from collections import OrderedDict
from dotenv import load_dotenv

from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from retriever_faiss import RetrieverFAISS
from rag.prompts import SYSTEM_PROMPT


def get_provider(name: str):
    if name == "chatgpt":
        return ChatGPTProvider()
    if name == "deepseek":
        return DeepSeekProvider()
    raise ValueError("Proveedor no válido. Usa chatgpt | deepseek.")


def unique_titles(items):
    seen = OrderedDict()
    for r in items:
        t = r["title"]
        if t not in seen:
            seen[t] = True
    return list(seen.keys())


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Asistente Normativa UFRO (RAG)")
    parser.add_argument("question", type=str, nargs="+", help="Consulta")
    parser.add_argument("--provider", type=str, default="chatgpt", help="chatgpt|deepseek")
    args = parser.parse_args()

    provider = get_provider(args.provider)
    question = " ".join(args.question)

    # Recuperación con mayor cobertura
    retriever = RetrieverFAISS(faiss_path="index.faiss", meta_path="meta.jsonl")
    top = retriever.search(question, k=8)

    if not top:
        print("\n=== RESPUESTA ===\n")
        print("No encontrado en normativa UFRO. Para esta consulta, te sugiero contactar con la unidad correspondiente.")
        print("\n=== REFERENCIAS ===\n- (sin resultados)")
        return

    # Contexto y referencias
    context = "\n\n".join([r["text"] for r in top])
    ref_titles = unique_titles(top)
    refs_block = "\n".join([f"- {t}" for t in ref_titles])

    # Prompt afinado (precisión y citas)
    user_prompt = (
        f"Pregunta: {question}\n\n"
        "Contexto de normativa (fragmentos relevantes):\n"
        f"{context}\n\n"
        "Instrucciones de respuesta:\n"
        "- Responde SOLO en base al contexto anterior.\n"
        "- Si hay varias fechas/valores, selecciona la que responda EXACTAMENTE a la pregunta.\n"
        "- Sé explícito con la fecha/valor (formato: 'Lunes 4 de agosto de 2025', por ejemplo).\n"
        "- Si la información no está en el contexto, responde: 'No encontrado en normativa UFRO'.\n"
        "- Al final agrega:\n"
        f"Referencias:\n{refs_block}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    answer = provider.chat(messages)

    print("\n=== RESPUESTA ===\n")
    print(answer)
    print("\n=== REFERENCIAS ===")
    for t in ref_titles:
        print(f"- {t}")


if __name__ == "__main__":
    main()
