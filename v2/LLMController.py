import dotenv
from typing import Optional
import getpass
import os
from langchain.chat_models import init_chat_model

dotenv.load_dotenv()

class LLMManager:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key
        self.model_name = model_name
        self.llm = self.load_model("gemini")
        self.embeddings = self.load_embeddings("gemini")
        self.rag_prompt = None
        self.rag_prompt_template = None
        self.rag_prompt_template_template = None
        self.rag_prompt_template_template_template = None
    
    def load_model(self, model_name: str):
        """
        Load model with different sources
        """
        if model_name == "gemini":
            if not os.environ.get("GOOGLE_API_KEY"):
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not found in environment variables")
                os.environ["GOOGLE_API_KEY"] = api_key
            
            self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
            return self.llm

        elif model_name == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
            
            self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")

        elif model_name == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
            
            self.llm = init_chat_model("claude-3-5-sonnet-20240620", model_provider="anthropic")

    def load_embeddings(self, embeddings_name: str, huggingface_model_name: Optional[str] = "sentence-transformers/all-mpnet-base-v2"):
        """
        Load embeddings with different sources
        """
        if embeddings_name == "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            
            if not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

        elif embeddings_name == "openai":
            from langchain_openai import OpenAIEmbeddings
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        elif embeddings_name == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            if not os.environ.get("HUGGINGFACE_API_KEY"):
                os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

            self.embeddings = HuggingFaceEmbeddings(model_name=huggingface_model_name)

        return self.embeddings