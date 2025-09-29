import json, csv, time, re, argparse
from collections import defaultdict
from dotenv import load_dotenv
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from retriever_faiss import RetrieverFAISS
from rag.prompts import SYSTEM_PROMPT

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{3,}")
STOP = {"de","del","la","el","lo","los","las","y","o","u","en","para","por","segun","según","un","una","al","con","que","se","es"}

def tokenize(s: str):
    if not s:
        return []
    return [w.lower() for w in WORD_RE.findall(s)]

def score_keywords(expected: str, answer: str, min_len=3, threshold=0.6):
    exp_tokens = set(tokenize(expected))
    ans_tokens = set(tokenize(answer))
    exp_tokens = {t for t in exp_tokens if t not in STOP and len(t) >= min_len}
    if not exp_tokens:
        return False
    inter = len(exp_tokens & ans_tokens)
    cov = inter / len(exp_tokens)
    return cov >= threshold

def get_provider(name: str):
    if name == "chatgpt":
        return ChatGPTProvider()
    if name == "deepseek":
        return DeepSeekProvider()
    raise ValueError("Proveedor no válido. Usa chatgpt | deepseek.")

def run_one_provider(provider_name: str, gold, retriever, k=8):
    provider = get_provider(provider_name)
    rows = []
    for item in gold:
        question = item["question"]
        expected = item["expected"]

        t0 = time.time()
        top = retriever.search(question, k=k)
        context = "\n\n".join([r["text"] for r in top])
        refs = ", ".join(sorted(set([r["title"] for r in top])))

        user_prompt = (
            f"Pregunta: {question}\n\n"
            f"Contexto:\n{context}\n\n"
            "Responde SOLO con base al contexto. "
            "Si no hay información suficiente, responde: 'No encontrado en normativa UFRO'.\n"
            f"Al final agrega:\nReferencias:\n{refs}"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        answer = provider.chat(messages)
        latency = time.time() - t0
        abstained = "no encontrado en normativa ufro" in answer.lower()
        correct_kw = score_keywords(expected, answer)

        rows.append({
            "provider": provider_name,
            "question": question,
            "expected": expected,
            "answer": answer,
            "references": refs,
            "latency_sec": round(latency, 2),
            "abstained": abstained,
            "correct_kw": correct_kw
        })
        print(f"[{provider_name}] {question} | correct_kw={correct_kw} | abstained={abstained} | {latency:.2f}s")
    return rows

def summarize(rows):
    by_provider = defaultdict(list)
    for r in rows:
        by_provider[r["provider"]].append(r)

    summary = []
    for prov, lst in by_provider.items():
        n = len(lst)
        correct = sum(1 for x in lst if x["correct_kw"])
        abst = sum(1 for x in lst if x["abstained"])
        lat_mean = sum(x["latency_sec"] for x in lst) / n if n else 0.0
        summary.append({
            "provider": prov,
            "total_questions": n,
            "correct_kw_count": correct,
            "correct_kw_rate_%": round(100 * correct / n, 1) if n else 0.0,
            "abstained_count": abst,
            "abstained_rate_%": round(100 * abst / n, 1) if n else 0.0,
            "avg_latency_sec": round(lat_mean, 2),
        })
    return summary

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluación comparativa ChatGPT vs DeepSeek")
    parser.add_argument("--gold", default="gold_set.json", help="Ruta al gold set")
    parser.add_argument("--out", default="results_benchmark.csv", help="CSV combinado de salida")
    parser.add_argument("--summary", default="results_benchmark_summary.csv", help="Resumen por proveedor")
    parser.add_argument("--providers", nargs="+", default=["chatgpt","deepseek"], help="Lista de proveedores a evaluar")
    parser.add_argument("--k", type=int, default=8, help="Top-k para recuperación")
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)

    retriever = RetrieverFAISS("index.faiss", "meta.jsonl")
    all_rows = []
    for prov in args.providers:
        rows = run_one_provider(prov, gold, retriever, k=args.k)
        all_rows.extend(rows)

    # CSV combinado
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Resumen por proveedor
    summ = summarize(all_rows)
    with open(args.summary, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)

    print("\n=== RESUMEN POR PROVEEDOR ===")
    for s in summ:
        print(s)
    print(f"\n✅ Resultados guardados en: {args.out} y {args.summary}")

if __name__ == "__main__":
    main()
