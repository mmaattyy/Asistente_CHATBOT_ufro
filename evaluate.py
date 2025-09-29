import json, csv, time, re
from dotenv import load_dotenv
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from retriever_jsonl import Retriever
from rag.prompts import SYSTEM_PROMPT

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{3,}")

def get_provider(name: str):
    if name == "chatgpt":
        return ChatGPTProvider()
    if name == "deepseek":
        return DeepSeekProvider()
    raise ValueError("Proveedor no válido. Usa chatgpt | deepseek.")

def tokenize(s: str):
    if not s:
        return []
    return [w.lower() for w in WORD_RE.findall(s)]

def score_keywords(expected: str, answer: str, min_len=3, threshold=0.6):
    exp_tokens = set(tokenize(expected))
    ans_tokens = set(tokenize(answer))
    # quitar palabras muy comunes
    stop = {"de","del","la","el","lo","los","las","y","o","u","en","para","por","segun","según","un","una","al","con","que","se","es"}
    exp_tokens = {t for t in exp_tokens if t not in stop and len(t) >= min_len}
    if not exp_tokens:
        return False
    inter = len(exp_tokens & ans_tokens)
    cov = inter / len(exp_tokens)
    return cov >= threshold

def run_eval(gold_file="gold_set.json", out_file="results.csv", provider_name="chatgpt"):
    load_dotenv()
    provider = get_provider(provider_name)
    retriever = Retriever("index")

    with open(gold_file, encoding="utf-8") as f:
        gold = json.load(f)

    results = []
    for q in gold:
        question = q["question"]
        expected = q["expected"]

        start = time.time()
        top = retriever.search(question, k=8)
        context = "\n\n".join([r["text"] for r in top])
        refs = ", ".join(sorted(set([r["title"] for r in top])))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
             f"Pregunta: {question}\n\nContexto:\n{context}\n\n"
             "Responde SOLO con base al contexto. Si no hay información suficiente, responde: 'No encontrado en normativa UFRO'.\n"
             f"Al final agrega:\nReferencias:\n{refs}"}
        ]

        answer = provider.chat(messages)
        latency = time.time() - start
        abstained = "no encontrado en normativa ufro" in answer.lower()

        correct_kw = score_keywords(expected, answer)

        results.append({
            "question": question,
            "expected": expected,
            "answer": answer,
            "references": refs,
            "latency_sec": round(latency,2),
            "abstained": abstained,
            "correct_kw": correct_kw
        })
        print(f"[OK] {question} | correct_kw={correct_kw} | abstained={abstained}")

    # Guardar resultados
    with open(out_file,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    print(f"\n✅ Evaluación completada. Resultados guardados en {out_file}")

if __name__=="__main__":
    run_eval()
