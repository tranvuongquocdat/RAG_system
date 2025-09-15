"""
Embedding module for RAG system
Quản lý Gemini embedding model và các operations liên quan
"""

import time
from typing import List, Optional, Dict, Any
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

from config import get_config
from utils import setup_logging, clean_text

class EmbeddingManager:
    """
    Quản lý embedding operations sử dụng Gemini
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Khởi tạo EmbeddingManager
        
        Args:
            api_key: Gemini API key (optional, sẽ lấy từ config nếu không có)
            model_name: Tên model embedding (optional, sẽ lấy từ config nếu không có)
        """
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
        
        # Sử dụng config hoặc parameters
        self.api_key = api_key or self.config.model.gemini_api_key
        self.model_name = model_name or self.config.model.gemini_embedding_model
        
        if not self.api_key:
            raise ValueError("Gemini API key không được cấu hình")
        
        # Khởi tạo embedding model
        self._initialize_model()
        
        # Cache để tránh embedding lại text giống nhau
        self._embedding_cache: Dict[str, List[float]] = {}
        
        self.logger.info(f"✅ EmbeddingManager đã khởi tạo với model: {self.model_name}")
    
    def _initialize_model(self) -> None:
        """Khởi tạo Gemini embedding model"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                google_api_key=self.api_key,
                model=self.model_name
            )
            
            # Test connection với một text đơn giản
            test_embedding = self.embeddings.embed_query("test")
            self.vector_dimension = len(test_embedding)
            
            self.logger.info(f"✅ Kết nối Gemini thành công. Vector dimension: {self.vector_dimension}")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi khởi tạo Gemini embedding: {str(e)}")
            raise
    
    def get_vector_dimension(self) -> int:
        """
        Lấy số chiều của vector embedding
        
        Returns:
            Số chiều vector
        """
        return self.vector_dimension
    
    def embed_text(self, text: str, use_cache: bool = True) -> List[float]:
        """
        Tạo embedding cho một text
        
        Args:
            text: Text cần embedding
            use_cache: Có sử dụng cache không
        
        Returns:
            Vector embedding
        """
        if not text or not text.strip():
            self.logger.warning("Text rỗng được truyền vào embed_text")
            return [0.0] * self.vector_dimension
        
        # Làm sạch text
        cleaned_text = clean_text(text)
        
        # Kiểm tra cache
        if use_cache and cleaned_text in self._embedding_cache:
            self.logger.debug(f"✅ Sử dụng embedding từ cache cho text: {cleaned_text[:50]}...")
            return self._embedding_cache[cleaned_text]
        
        try:
            # Tạo embedding
            embedding = self.embeddings.embed_query(cleaned_text)
            
            # Lưu vào cache
            if use_cache:
                self._embedding_cache[cleaned_text] = embedding
            
            self.logger.debug(f"✅ Đã tạo embedding cho text: {cleaned_text[:50]}...")
            return embedding
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tạo embedding: {str(e)}")
            # Return zero vector in case of error
            return [0.0] * self.vector_dimension
    
    def embed_texts(self, texts: List[str], use_cache: bool = True, batch_size: int = 100) -> List[List[float]]:
        """
        Tạo embedding cho nhiều texts
        
        Args:
            texts: Danh sách texts cần embedding
            use_cache: Có sử dụng cache không
            batch_size: Kích thước batch để xử lý
        
        Returns:
            List of vector embeddings
        """
        if not texts:
            return []
        
        embeddings = []
        
        # Xử lý theo batch để tránh rate limiting
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                embedding = self.embed_text(text, use_cache)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            
            # Rate limiting - pause giữa các batch
            if i + batch_size < len(texts):
                time.sleep(0.1)  # 100ms delay
            
            self.logger.debug(f"✅ Đã xử lý batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        self.logger.info(f"✅ Đã tạo embedding cho {len(texts)} texts")
        return embeddings
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Tính cosine similarity giữa hai embeddings
        
        Args:
            embedding1: Vector embedding thứ nhất
            embedding2: Vector embedding thứ hai
        
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tính similarity: {str(e)}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Tìm embeddings tương đồng nhất với query
        
        Args:
            query_embedding: Embedding của query
            candidate_embeddings: Danh sách candidate embeddings
            top_k: Số lượng kết quả trả về
        
        Returns:
            List of dictionaries với index và similarity score
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append({
                'index': i,
                'similarity': similarity
            })
        
        # Sắp xếp theo similarity score giảm dần
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def clear_cache(self) -> None:
        """Xóa embedding cache"""
        self._embedding_cache.clear()
        self.logger.info("✅ Đã xóa embedding cache")
    
    def get_cache_size(self) -> int:
        """Lấy số lượng entries trong cache"""
        return len(self._embedding_cache)
    
    def embed_documents_with_metadata(
        self, 
        documents: List[Dict[str, Any]], 
        text_field: str = 'content'
    ) -> List[Dict[str, Any]]:
        """
        Tạo embedding cho documents kèm metadata
        
        Args:
            documents: Danh sách documents với metadata
            text_field: Tên field chứa text content
        
        Returns:
            Documents với embedding được thêm vào
        """
        texts = [doc.get(text_field, '') for doc in documents]
        embeddings = self.embed_texts(texts)
        
        # Thêm embedding vào documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
            doc['embedding_model'] = self.model_name
            doc['vector_dimension'] = self.vector_dimension
        
        return documents

class EmbeddingValidator:
    """
    Validator để kiểm tra chất lượng embeddings
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.logger = embedding_manager.logger
    
    def validate_embedding(self, embedding: List[float]) -> Dict[str, Any]:
        """
        Kiểm tra chất lượng của một embedding
        
        Args:
            embedding: Vector embedding cần kiểm tra
        
        Returns:
            Dictionary chứa kết quả validation
        """
        result = {
            'is_valid': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            vec = np.array(embedding)
            
            # Kiểm tra dimension
            expected_dim = self.embedding_manager.get_vector_dimension()
            if len(embedding) != expected_dim:
                result['is_valid'] = False
                result['issues'].append(f"Sai dimension: {len(embedding)} != {expected_dim}")
            
            # Kiểm tra zero vector
            if np.allclose(vec, 0):
                result['is_valid'] = False
                result['issues'].append("Zero vector")
            
            # Kiểm tra NaN hoặc infinite values
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                result['is_valid'] = False
                result['issues'].append("Chứa NaN hoặc infinite values")
            
            # Statistics
            result['stats'] = {
                'mean': float(np.mean(vec)),
                'std': float(np.std(vec)),
                'norm': float(np.linalg.norm(vec)),
                'min': float(np.min(vec)),
                'max': float(np.max(vec))
            }
            
        except Exception as e:
            result['is_valid'] = False
            result['issues'].append(f"Lỗi validation: {str(e)}")
        
        return result
    
    def validate_embeddings_batch(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Kiểm tra chất lượng của một batch embeddings
        
        Args:
            embeddings: List of embeddings cần kiểm tra
        
        Returns:
            Dictionary chứa kết quả validation tổng hợp
        """
        results = []
        for i, embedding in enumerate(embeddings):
            result = self.validate_embedding(embedding)
            result['index'] = i
            results.append(result)
        
        # Tổng hợp kết quả
        valid_count = sum(1 for r in results if r['is_valid'])
        
        summary = {
            'total_embeddings': len(embeddings),
            'valid_embeddings': valid_count,
            'invalid_embeddings': len(embeddings) - valid_count,
            'validation_rate': valid_count / len(embeddings) if embeddings else 0,
            'detailed_results': results
        }
        
        return summary

# Factory function để tạo EmbeddingManager
def create_embedding_manager(api_key: Optional[str] = None, model_name: Optional[str] = None) -> EmbeddingManager:
    """
    Factory function để tạo EmbeddingManager
    
    Args:
        api_key: Gemini API key (optional)
        model_name: Embedding model name (optional)
    
    Returns:
        EmbeddingManager instance
    """
    return EmbeddingManager(api_key=api_key, model_name=model_name)
