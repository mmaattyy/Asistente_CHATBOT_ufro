import argparse, os
from dotenv import load_dotenv
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from rag.prompts import SYSTEM_PROMPT

def get_provider(name: str):
    if name == "chatgpt":
        return ChatGPTProvider()
    if name == "deepseek":
        return DeepSeekProvider()
    raise ValueError("Proveedor no válido. Usa chatgpt o deepseek.")

def main():
    load_dotenv()  # lee .env
    parser = argparse.ArgumentParser(description="Asistente Normativa UFRO (RAG)")
    parser.add_argument("question", type=str, nargs="+", help="Consulta")
    parser.add_argument("--provider", type=str, default="chatgpt", help="chatgpt|deepseek")
    args = parser.parse_args()

    provider = get_provider(args.provider)
    question = " ".join(args.question)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    answer = provider.chat(messages)
    print(answer)

if __name__ == "__main__":
    main()
