"""
Retrieval module for RAG system
Module truy xuất dữ liệu từ Qdrant vector database
"""

import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

from config import get_config
from utils import setup_logging, clean_text
from embedding import EmbeddingManager
from database import QdrantManager

@dataclass
class RetrievalResult:
    """
    Kết quả truy xuất từ vector database
    """
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    filename: str = ""
    chunk_id: int = 0
    document_type: str = "general"

class QueryProcessor:
    """
    Xử lý và tối ưu queries trước khi retrieval
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
    
    def preprocess_query(self, query: str) -> str:
        """
        Tiền xử lý query
        
        Args:
            query: Raw query từ user
            
        Returns:
            Processed query
        """
        # Làm sạch query
        processed_query = clean_text(query)
        
        # Loại bỏ stop words không cần thiết (có thể mở rộng)
        stop_words = ['là', 'của', 'và', 'có', 'được', 'trong', 'với', 'để']
        words = processed_query.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        if len(filtered_words) < len(words) * 0.3:  # Nếu quá nhiều stop words bị loại
            return processed_query  # Giữ nguyên
        
        return ' '.join(filtered_words)
    
    def expand_query(self, query: str, expansion_terms: Optional[List[str]] = None) -> str:
        """
        Mở rộng query với các từ đồng nghĩa hoặc liên quan
        
        Args:
            query: Query gốc
            expansion_terms: Danh sách terms để mở rộng
            
        Returns:
            Expanded query
        """
        if not expansion_terms:
            return query
        
        # Thêm expansion terms vào query
        expanded = f"{query} {' '.join(expansion_terms)}"
        return expanded
    
    def generate_query_variations(self, query: str) -> List[str]:
        """
        Tạo các biến thể của query để tăng recall
        
        Args:
            query: Query gốc
            
        Returns:
            Danh sách query variations
        """
        variations = [query]
        
        # Variation 1: Thêm context keywords
        business_context = f"{query} doanh nghiệp tổ chức công ty"
        variations.append(business_context)
        
        # Variation 2: Question form
        if not query.endswith('?'):
            question_form = f"{query}?"
            variations.append(question_form)
        
        # Variation 3: Formal form
        formal_form = f"thông tin về {query}"
        variations.append(formal_form)
        
        return variations

class RetrievalEngine:
    """
    Engine chính cho retrieval operations
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        qdrant_manager: QdrantManager
    ):
        """
        Khởi tạo RetrievalEngine
        
        Args:
            embedding_manager: EmbeddingManager instance
            qdrant_manager: QdrantManager instance
        """
        self.embedding_manager = embedding_manager
        self.qdrant_manager = qdrant_manager
        self.query_processor = QueryProcessor(embedding_manager)
        
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
        
        # Cache cho query embeddings
        self._query_cache: Dict[str, List[float]] = {}
        
        self.logger.info("✅ RetrievalEngine đã khởi tạo")
    
    def retrieve_similar_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True
    ) -> List[RetrievalResult]:
        """
        Truy xuất documents tương đồng với query
        
        Args:
            query: Query string
            top_k: Số lượng kết quả (default từ config)
            score_threshold: Ngưỡng điểm tối thiểu (default từ config)
            filters: Bộ lọc metadata
            rerank: Có rerank kết quả không
            
        Returns:
            Danh sách RetrievalResult
        """
        try:
            # Sử dụng default values từ config
            if top_k is None:
                top_k = self.config.rag.top_k
            if score_threshold is None:
                score_threshold = self.config.rag.score_threshold
            
            # Preprocess query
            processed_query = self.query_processor.preprocess_query(query)
            
            # Generate query embedding
            query_embedding = self._get_query_embedding(processed_query)
            
            # Search in Qdrant
            raw_results = self.qdrant_manager.search_similar(
                query_vector=query_embedding,
                top_k=top_k * 2 if rerank else top_k,  # Get more for reranking
                score_threshold=score_threshold,
                filters=filters
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in raw_results:
                retrieval_result = RetrievalResult(
                    id=result['id'],
                    content=result['content'],
                    score=result['score'],
                    metadata=result['metadata'],
                    source=result['source'],
                    filename=result['filename'],
                    chunk_id=result['chunk_id'],
                    document_type=result['document_type']
                )
                results.append(retrieval_result)
            
            # Rerank if requested
            if rerank and len(results) > top_k:
                results = self._rerank_results(processed_query, results)[:top_k]
            
            self.logger.debug(f"✅ Retrieved {len(results)} documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi retrieve documents: {str(e)}")
            return []
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Lấy embedding cho query (với cache)
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        if query in self._query_cache:
            return self._query_cache[query]
        
        embedding = self.embedding_manager.embed_text(query)
        self._query_cache[query] = embedding
        
        return embedding
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank kết quả dựa trên semantic similarity và metadata
        
        Args:
            query: Query string
            results: Danh sách kết quả ban đầu
            
        Returns:
            Reranked results
        """
        try:
            # Simple reranking based on content length and recency
            for result in results:
                # Boost score based on content length (not too short, not too long)
                content_length = len(result.content)
                if 100 <= content_length <= 2000:
                    result.score *= 1.1
                elif content_length < 50:
                    result.score *= 0.8
                
                # Boost score for certain document types
                if result.document_type in ['manual', 'policy', 'report']:
                    result.score *= 1.05
                
                # Boost score for recent documents
                created_at = result.metadata.get('created_at')
                if created_at:
                    try:
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        days_old = (datetime.now() - created_date.replace(tzinfo=None)).days
                        if days_old < 30:
                            result.score *= 1.1
                        elif days_old > 365:
                            result.score *= 0.95
                    except:
                        pass
            
            # Sort by updated scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi rerank results: {str(e)}")
            return results
    
    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Hybrid search kết hợp semantic và keyword search
        
        Args:
            query: Query string
            top_k: Số lượng kết quả
            semantic_weight: Trọng số cho semantic search
            keyword_weight: Trọng số cho keyword search
            filters: Bộ lọc metadata
            
        Returns:
            Hybrid search results
        """
        try:
            if top_k is None:
                top_k = self.config.rag.top_k
            
            # Semantic search
            semantic_results = self.retrieve_similar_documents(
                query=query,
                top_k=top_k * 2,
                filters=filters,
                rerank=False
            )
            
            # Keyword search simulation (tìm documents chứa keywords)
            query_keywords = query.lower().split()
            keyword_results = []
            
            for result in semantic_results:
                content_lower = result.content.lower()
                keyword_matches = sum(1 for kw in query_keywords if kw in content_lower)
                keyword_score = keyword_matches / len(query_keywords) if query_keywords else 0
                
                if keyword_score > 0:
                    # Create a copy with keyword score
                    keyword_result = RetrievalResult(
                        id=result.id,
                        content=result.content,
                        score=keyword_score,
                        metadata=result.metadata,
                        source=result.source,
                        filename=result.filename,
                        chunk_id=result.chunk_id,
                        document_type=result.document_type
                    )
                    keyword_results.append(keyword_result)
            
            # Combine scores
            combined_results = {}
            
            # Add semantic results
            for result in semantic_results:
                combined_results[result.id] = {
                    'result': result,
                    'semantic_score': result.score,
                    'keyword_score': 0
                }
            
            # Add keyword scores
            for result in keyword_results:
                if result.id in combined_results:
                    combined_results[result.id]['keyword_score'] = result.score
            
            # Calculate final scores
            final_results = []
            for doc_id, scores in combined_results.items():
                result = scores['result']
                final_score = (
                    semantic_weight * scores['semantic_score'] + 
                    keyword_weight * scores['keyword_score']
                )
                
                final_result = RetrievalResult(
                    id=result.id,
                    content=result.content,
                    score=final_score,
                    metadata=result.metadata,
                    source=result.source,
                    filename=result.filename,
                    chunk_id=result.chunk_id,
                    document_type=result.document_type
                )
                final_results.append(final_result)
            
            # Sort and return top_k
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.debug(f"✅ Hybrid search returned {len(final_results[:top_k])} results")
            return final_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi hybrid search: {str(e)}")
            return self.retrieve_similar_documents(query, top_k, filters=filters)
    
    def multi_query_retrieval(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        final_top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieval với multiple queries để tăng coverage
        
        Args:
            queries: Danh sách queries
            top_k_per_query: Số kết quả mỗi query
            final_top_k: Số kết quả cuối cùng
            
        Returns:
            Combined and deduplicated results
        """
        try:
            if final_top_k is None:
                final_top_k = self.config.rag.top_k
            
            all_results = {}  # doc_id -> best result
            
            for query in queries:
                results = self.retrieve_similar_documents(
                    query=query,
                    top_k=top_k_per_query,
                    rerank=False
                )
                
                for result in results:
                    # Keep the result with highest score for each document
                    if result.id not in all_results or result.score > all_results[result.id].score:
                        all_results[result.id] = result
            
            # Convert to list and sort
            final_results = list(all_results.values())
            final_results.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.debug(f"✅ Multi-query retrieval: {len(queries)} queries -> {len(final_results)} unique results")
            return final_results[:final_top_k]
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi multi-query retrieval: {str(e)}")
            return []
    
    def retrieve_by_document_type(
        self,
        query: str,
        document_types: List[str],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieval lọc theo loại document
        
        Args:
            query: Query string
            document_types: Danh sách loại documents
            top_k: Số lượng kết quả
            
        Returns:
            Filtered results
        """
        filters = {
            'document_type': document_types
        }
        
        return self.retrieve_similar_documents(
            query=query,
            top_k=top_k,
            filters=filters
        )
    
    def retrieve_recent_documents(
        self,
        query: str,
        days_back: int = 30,
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        Retrieval chỉ documents gần đây
        
        Args:
            query: Query string
            days_back: Số ngày về trước
            top_k: Số lượng kết quả
            
        Returns:
            Recent documents
        """
        # Calculate date threshold
        threshold_date = datetime.now() - timedelta(days=days_back)
        
        filters = {
            'created_at': {
                'range': {
                    'gte': threshold_date.isoformat()
                }
            }
        }
        
        return self.retrieve_similar_documents(
            query=query,
            top_k=top_k,
            filters=filters
        )
    
    def get_document_context(
        self,
        document_id: str,
        context_chunks: int = 2
    ) -> List[RetrievalResult]:
        """
        Lấy context xung quanh một document (chunks before/after)
        
        Args:
            document_id: ID của document
            context_chunks: Số chunks context mỗi bên
            
        Returns:
            Context documents
        """
        try:
            # This is a simplified implementation
            # In practice, you'd need to store chunk relationships
            
            # Get the target document first
            results = self.qdrant_manager.search_similar(
                query_vector=[0] * self.embedding_manager.get_vector_dimension(),
                top_k=1,
                filters={'id': document_id}
            )
            
            if not results:
                return []
            
            target_doc = results[0]
            filename = target_doc['filename']
            chunk_id = target_doc['chunk_id']
            
            # Get surrounding chunks
            context_results = []
            for offset in range(-context_chunks, context_chunks + 1):
                target_chunk_id = chunk_id + offset
                if target_chunk_id >= 0:
                    chunk_results = self.qdrant_manager.search_similar(
                        query_vector=[0] * self.embedding_manager.get_vector_dimension(),
                        top_k=1,
                        filters={
                            'filename': filename,
                            'chunk_id': target_chunk_id
                        }
                    )
                    
                    if chunk_results:
                        result = chunk_results[0]
                        retrieval_result = RetrievalResult(
                            id=result['id'],
                            content=result['content'],
                            score=1.0,  # Max score for context
                            metadata=result['metadata'],
                            source=result['source'],
                            filename=result['filename'],
                            chunk_id=result['chunk_id'],
                            document_type=result['document_type']
                        )
                        context_results.append(retrieval_result)
            
            # Sort by chunk_id
            context_results.sort(key=lambda x: x.chunk_id)
            
            self.logger.debug(f"✅ Retrieved {len(context_results)} context chunks for {document_id}")
            return context_results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi get document context: {str(e)}")
            return []
    
    def clear_query_cache(self) -> None:
        """Xóa query cache"""
        self._query_cache.clear()
        self.logger.info("✅ Đã xóa query cache")

class ContextBuilder:
    """
    Xây dựng context từ retrieved documents
    """
    
    def __init__(self, retrieval_engine: RetrievalEngine):
        self.retrieval_engine = retrieval_engine
        self.config = get_config()
        self.logger = retrieval_engine.logger
    
    def build_context(
        self,
        results: List[RetrievalResult],
        max_context_length: Optional[int] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Xây dựng context string từ retrieval results
        
        Args:
            results: Danh sách RetrievalResult
            max_context_length: Độ dài context tối đa
            include_metadata: Có bao gồm metadata không
            
        Returns:
            Context string
        """
        if not results:
            return ""
        
        if max_context_length is None:
            max_context_length = self.config.rag.context_window
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # Format document part
            doc_part = f"[Tài liệu {i+1}"
            
            if include_metadata:
                if result.filename:
                    doc_part += f" - {result.filename}"
                if result.document_type != "general":
                    doc_part += f" ({result.document_type})"
                if result.score:
                    doc_part += f" - Độ tương đồng: {result.score:.2f}"
            
            doc_part += "]\n"
            doc_part += result.content
            doc_part += "\n\n"
            
            # Check length limit
            if current_length + len(doc_part) > max_context_length:
                if context_parts:  # Nếu đã có ít nhất 1 document
                    break
                else:  # Nếu document đầu tiên quá dài, cắt ngắn
                    remaining_length = max_context_length - current_length - len(f"[Tài liệu {i+1}]\n") - 4
                    if remaining_length > 100:
                        truncated_content = result.content[:remaining_length] + "..."
                        doc_part = f"[Tài liệu {i+1}]\n{truncated_content}\n\n"
                        context_parts.append(doc_part)
                    break
            
            context_parts.append(doc_part)
            current_length += len(doc_part)
        
        context = "".join(context_parts).strip()
        
        self.logger.debug(f"✅ Built context: {len(context)} chars from {len(context_parts)} documents")
        return context

# Factory functions
def create_retrieval_engine(
    embedding_manager: EmbeddingManager,
    qdrant_manager: QdrantManager
) -> RetrievalEngine:
    """
    Factory function để tạo RetrievalEngine
    """
    return RetrievalEngine(embedding_manager, qdrant_manager)

def create_context_builder(retrieval_engine: RetrievalEngine) -> ContextBuilder:
    """
    Factory function để tạo ContextBuilder
    """
    return ContextBuilder(retrieval_engine)
