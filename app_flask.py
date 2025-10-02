from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from collections import OrderedDict
import traceback

from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.prompts import SYSTEM_PROMPT

load_dotenv()
app = Flask(__name__)

# --- carga perezosa del retriever (evita que el server se caiga al importar) ---
_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        from retriever_faiss import RetrieverFAISS
        _retriever = RetrieverFAISS(faiss_path="index.faiss", meta_path="meta.jsonl")
    return _retriever

def get_provider(name: str):
    if name == "chatgpt":
        return ChatGPTProvider()
    if name == "deepseek":
        return DeepSeekProvider()
    raise ValueError("Proveedor no válido: usa chatgpt | deepseek")

def unique_titles(items):
    seen = OrderedDict()
    for r in items:
        t = r["title"]
        if t not in seen:
            seen[t] = True
    return list(seen.keys())

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True) or {}
        question = (data.get("question") or "").strip()
        provider_name = (data.get("provider") or "chatgpt").strip().lower()
        if not question:
            return jsonify({"error": "Falta 'question'"}), 400

        provider = get_provider(provider_name)
        retriever = get_retriever()  # ← aquí podría fallar; si falla devolvemos JSON con trace

        top = retriever.search(question, k=8)
        if not top:
            answer = "No encontrado en normativa UFRO. Para esta consulta, te sugiero contactar con la unidad correspondiente."
            return jsonify({"answer": answer, "references": []})

        context = "\n\n".join([r["text"] for r in top])
        ref_titles = unique_titles(top)
        refs_block = "\n".join([f"- {t}" for t in ref_titles])

        user_prompt = (
            f"Pregunta: {question}\n\n"
            "Contexto de normativa (fragmentos relevantes):\n"
            f"{context}\n\n"
            "Instrucciones de respuesta:\n"
            "- Responde SOLO en base al contexto anterior.\n"
            "- Si hay varias fechas/valores, selecciona la que responda EXACTAMENTE a la pregunta.\n"
            "- Sé explícito con la fecha/valor (ej.: 'Lunes 4 de agosto de 2025').\n"
            "- Si la información no está en el contexto, responde: 'No encontrado en normativa UFRO'.\n"
            "- Al final agrega:\n"
            f"Referencias:\n{refs_block}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        answer = provider.chat(messages)
        return jsonify({"answer": answer, "references": ref_titles})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    print("➡️  Iniciando Flask en http://127.0.0.1:5000 ...")
    print("   - Asegúrate de tener templates/index.html")
    print("   - Y los archivos index.faiss + meta.jsonl (usa: python build_faiss.py)")
    app.run(host="127.0.0.1", port=5000, debug=True)
