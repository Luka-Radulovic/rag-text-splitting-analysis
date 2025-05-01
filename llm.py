from langchain_ollama import OllamaLLM
from deepeval.models import DeepEvalBaseLLM


class CustomLlama3_8B(DeepEvalBaseLLM):
    def __init__(self):
        self.model = OllamaLLM(model="llama3.1:8b")

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt)
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self):
        return "llama3.1:8b"
    