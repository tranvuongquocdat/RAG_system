"""
Configuration module for RAG system
Quản lý tất cả cấu hình chung cho hệ thống RAG
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    """Cấu hình cho các model AI"""
    # Gemini Configuration
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model_name: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
    
    # OpenAI Configuration (backup option)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
    
    # Temperature settings
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "2048"))

@dataclass
class QdrantConfig:
    """Cấu hình cho Qdrant vector database"""
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port: int = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    
    # Collection settings
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "enterprise_documents")
    vector_size: int = int(os.getenv("VECTOR_SIZE", "768"))  # Default for text-embedding-004
    distance_metric: str = os.getenv("DISTANCE_METRIC", "cosine")

@dataclass
class RAGConfig:
    """Cấu hình cho RAG pipeline"""
    # Retrieval settings
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    score_threshold: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.7"))
    
    # Text splitting settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Generation settings
    context_window: int = int(os.getenv("CONTEXT_WINDOW", "4000"))

@dataclass
class SecurityConfig:
    """Cấu hình bảo mật"""
    enable_auth: bool = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key-change-this")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

@dataclass
class AppConfig:
    """Cấu hình ứng dụng chính"""
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "rag_system.log")
    
    # Data directories
    data_dir: str = os.getenv("DATA_DIR", "./data")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    
    # API settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

class Config:
    """Main configuration class - tập trung tất cả cấu hình"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.qdrant = QdrantConfig()
        self.rag = RAGConfig()
        self.security = SecurityConfig()
        self.app = AppConfig()
    
    def validate(self) -> bool:
        """Kiểm tra tính hợp lệ của cấu hình"""
        errors = []
        
        # Check required API keys
        if not self.model.gemini_api_key and not self.model.openai_api_key:
            errors.append("Cần ít nhất một API key (GEMINI_API_KEY hoặc OPENAI_API_KEY)")
        
        # Check Qdrant connection
        if not self.qdrant.host:
            errors.append("QDRANT_HOST không được để trống")
        
        # Check data directories
        os.makedirs(self.app.data_dir, exist_ok=True)
        os.makedirs(self.app.upload_dir, exist_ok=True)
        
        if errors:
            for error in errors:
                print(f"❌ Lỗi cấu hình: {error}")
            return False
        
        print("✅ Cấu hình hợp lệ")
        return True
    
    def get_primary_llm_provider(self) -> str:
        """Xác định provider LLM chính để sử dụng"""
        if self.model.gemini_api_key:
            return "gemini"
        elif self.model.openai_api_key:
            return "openai"
        else:
            raise ValueError("Không có API key nào được cấu hình")

# Global config instance
config = Config()

# Utility functions
def get_config() -> Config:
    """Lấy instance config toàn cục"""
    return config

def validate_config() -> bool:
    """Kiểm tra cấu hình toàn cục"""
    return config.validate()
