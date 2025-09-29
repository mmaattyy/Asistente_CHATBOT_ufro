import os
from openai import OpenAI
from .base import Provider

class ChatGPTProvider(Provider):
    name = "chatgpt"

    def __init__(self, model: str = "openai/gpt-4.1-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no está en .env")

        # Usar endpoint de OpenRouter
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0)
        )
        return resp.choices[0].message.content
