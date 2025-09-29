from abc import ABC, abstractmethod

class Provider(ABC):
    name: str

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        ...
