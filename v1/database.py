"""
Database module for RAG system
Quản lý Qdrant vector database và các operations liên quan
"""

import time
import uuid
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, Range, GeoBoundingBox,
    SearchParams, UpdateStatus
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from config import get_config
from utils import setup_logging
from embedding import EmbeddingManager

class QdrantManager:
    """
    Quản lý Qdrant vector database operations
    """
    
    def __init__(
        self, 
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        Khởi tạo QdrantManager
        
        Args:
            host: Qdrant host (optional, sẽ lấy từ config)
            port: Qdrant port (optional, sẽ lấy từ config)
            api_key: Qdrant API key (optional)
            collection_name: Tên collection (optional, sẽ lấy từ config)
        """
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
        
        # Sử dụng config hoặc parameters
        self.host = host or self.config.qdrant.host
        self.port = port or self.config.qdrant.port
        self.api_key = api_key or self.config.qdrant.api_key
        self.collection_name = collection_name or self.config.qdrant.collection_name
        
        # Khởi tạo Qdrant client
        self._initialize_client()
        
        self.logger.info(f"✅ QdrantManager đã khởi tạo - Host: {self.host}:{self.port}")
    
    def _initialize_client(self) -> None:
        """Khởi tạo Qdrant client"""
        try:
            # Tạo client
            if self.api_key:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port
                )
            
            # Test connection
            collections = self.client.get_collections()
            self.logger.info(f"✅ Kết nối Qdrant thành công. Có {len(collections.collections)} collections")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi kết nối Qdrant: {str(e)}")
            raise
    
    def create_collection(
        self, 
        vector_size: int,
        distance_metric: str = "cosine",
        force_recreate: bool = False
    ) -> bool:
        """
        Tạo collection trong Qdrant
        
        Args:
            vector_size: Kích thước vector
            distance_metric: Metric tính khoảng cách (cosine, dot, euclidean)
            force_recreate: Có xóa và tạo lại collection nếu đã tồn tại
        
        Returns:
            True nếu tạo thành công
        """
        try:
            # Kiểm tra collection đã tồn tại chưa
            collections = self.client.get_collections()
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            
            if collection_exists:
                if force_recreate:
                    self.logger.info(f"🗑️ Xóa collection cũ: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    self.logger.info(f"✅ Collection đã tồn tại: {self.collection_name}")
                    return True
            
            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "dot": Distance.DOT,
                "euclidean": Distance.EUCLID
            }
            
            distance = distance_map.get(distance_metric.lower(), Distance.COSINE)
            
            # Tạo collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                ),
                # Cấu hình tối ưu cho performance
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=2
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0
                )
            )
            
            self.logger.info(f"✅ Đã tạo collection: {self.collection_name} (size: {vector_size}, metric: {distance_metric})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tạo collection: {str(e)}")
            return False
    
    def insert_documents(
        self, 
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> bool:
        """
        Insert documents vào Qdrant
        
        Args:
            documents: Danh sách documents với embedding và metadata
            batch_size: Kích thước batch để insert
        
        Returns:
            True nếu insert thành công
        """
        try:
            points = []
            
            for doc in documents:
                # Tạo point ID
                point_id = doc.get('id', str(uuid.uuid4()))
                
                # Vector embedding
                vector = doc.get('embedding', [])
                if not vector:
                    self.logger.warning(f"Document {point_id} không có embedding")
                    continue
                
                # Payload (metadata)
                payload = {
                    'content': doc.get('content', ''),
                    'filename': doc.get('filename', ''),
                    'document_type': doc.get('document_type', 'general'),
                    'chunk_id': doc.get('chunk_id', 0),
                    'created_at': doc.get('created_at', datetime.now().isoformat()),
                    'source': doc.get('source', ''),
                    'metadata': doc.get('metadata', {})
                }
                
                # Tạo point
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)
            
            # Insert theo batch
            total_inserted = 0
            for i in range(0, len(points), batch_size):
                batch_points = points[i:i + batch_size]
                
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch_points,
                    wait=True
                )
                
                if operation_info.status == UpdateStatus.COMPLETED:
                    total_inserted += len(batch_points)
                    self.logger.debug(f"✅ Đã insert batch {i//batch_size + 1}: {len(batch_points)} documents")
                else:
                    self.logger.error(f"❌ Lỗi insert batch {i//batch_size + 1}")
            
            self.logger.info(f"✅ Đã insert {total_inserted}/{len(documents)} documents vào Qdrant")
            return total_inserted == len(documents)
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi insert documents: {str(e)}")
            return False
    
    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm documents tương đồng
        
        Args:
            query_vector: Vector của query
            top_k: Số lượng kết quả trả về
            score_threshold: Ngưỡng điểm tối thiểu
            filters: Bộ lọc metadata
        
        Returns:
            Danh sách documents tương đồng
        """
        try:
            # Tạo filter nếu có
            query_filter = None
            if filters:
                conditions = []
                
                for key, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                    elif isinstance(value, list):
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(any=value)
                            )
                        )
                    elif isinstance(value, dict) and 'range' in value:
                        range_filter = value['range']
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(
                                    gte=range_filter.get('gte'),
                                    lte=range_filter.get('lte')
                                )
                            )
                        )
                
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Thực hiện search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Format kết quả
            results = []
            for hit in search_result:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'content': hit.payload.get('content', ''),
                    'filename': hit.payload.get('filename', ''),
                    'document_type': hit.payload.get('document_type', ''),
                    'chunk_id': hit.payload.get('chunk_id', 0),
                    'metadata': hit.payload.get('metadata', {}),
                    'source': hit.payload.get('source', '')
                }
                results.append(result)
            
            self.logger.debug(f"✅ Tìm thấy {len(results)} documents tương đồng")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi search: {str(e)}")
            return []
    
    def delete_documents(
        self, 
        document_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Xóa documents từ Qdrant
        
        Args:
            document_ids: Danh sách IDs cần xóa (optional)
            filters: Bộ lọc để xóa theo điều kiện (optional)
        
        Returns:
            True nếu xóa thành công
        """
        try:
            if document_ids:
                # Xóa theo IDs
                operation_info = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(
                        points=document_ids
                    ),
                    wait=True
                )
            elif filters:
                # Xóa theo filters
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                
                query_filter = Filter(must=conditions)
                operation_info = self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=query_filter
                    ),
                    wait=True
                )
            else:
                self.logger.error("Cần cung cấp document_ids hoặc filters để xóa")
                return False
            
            success = operation_info.status == UpdateStatus.COMPLETED
            if success:
                self.logger.info("✅ Đã xóa documents thành công")
            else:
                self.logger.error("❌ Lỗi khi xóa documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi xóa documents: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về collection
        
        Returns:
            Dictionary chứa thông tin collection
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            info = {
                'name': collection_info.config.params.vectors.size,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.name,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'status': collection_info.status.name,
                'optimizer_status': collection_info.optimizer_status.name if collection_info.optimizer_status else 'N/A'
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi lấy thông tin collection: {str(e)}")
            return {}
    
    def count_documents(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Đếm số lượng documents
        
        Args:
            filters: Bộ lọc (optional)
        
        Returns:
            Số lượng documents
        """
        try:
            if filters:
                # Count với filter
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                
                query_filter = Filter(must=conditions)
                result = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=query_filter
                )
            else:
                # Count tất cả
                result = self.client.count(collection_name=self.collection_name)
            
            return result.count
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi đếm documents: {str(e)}")
            return 0
    
    def update_document_metadata(
        self, 
        document_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Cập nhật metadata của document
        
        Args:
            document_id: ID của document
            metadata: Metadata mới
        
        Returns:
            True nếu cập nhật thành công
        """
        try:
            operation_info = self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[document_id],
                wait=True
            )
            
            success = operation_info.status == UpdateStatus.COMPLETED
            if success:
                self.logger.debug(f"✅ Đã cập nhật metadata cho document: {document_id}")
            else:
                self.logger.error(f"❌ Lỗi cập nhật metadata cho document: {document_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi cập nhật metadata: {str(e)}")
            return False

class QdrantVectorStoreManager:
    """
    Wrapper cho LangChain QdrantVectorStore
    Tích hợp với LangChain ecosystem
    """
    
    def __init__(
        self, 
        qdrant_manager: QdrantManager, 
        embedding_manager: EmbeddingManager
    ):
        """
        Khởi tạo QdrantVectorStoreManager
        
        Args:
            qdrant_manager: QdrantManager instance
            embedding_manager: EmbeddingManager instance
        """
        self.qdrant_manager = qdrant_manager
        self.embedding_manager = embedding_manager
        self.logger = qdrant_manager.logger
        
        # Tạo LangChain vector store
        self.vector_store = QdrantVectorStore(
            client=qdrant_manager.client,
            collection_name=qdrant_manager.collection_name,
            embeddings=embedding_manager.embeddings
        )
        
        self.logger.info("✅ QdrantVectorStoreManager đã khởi tạo")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Thêm LangChain Documents vào vector store
        
        Args:
            documents: Danh sách LangChain Documents
        
        Returns:
            Danh sách IDs của documents đã thêm
        """
        try:
            ids = self.vector_store.add_documents(documents)
            self.logger.info(f"✅ Đã thêm {len(documents)} documents vào vector store")
            return ids
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi thêm documents: {str(e)}")
            return []
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Tìm kiếm documents tương đồng với query
        
        Args:
            query: Query string
            k: Số lượng kết quả
            filter: Bộ lọc metadata
        
        Returns:
            Danh sách LangChain Documents
        """
        try:
            docs = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            self.logger.debug(f"✅ Tìm thấy {len(docs)} documents cho query: {query[:50]}...")
            return docs
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Tìm kiếm documents với similarity scores
        
        Args:
            query: Query string
            k: Số lượng kết quả
            filter: Bộ lọc metadata
        
        Returns:
            Danh sách tuples (Document, score)
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            
            self.logger.debug(f"✅ Tìm thấy {len(results)} documents với scores")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi search với scores: {str(e)}")
            return []
    
    def as_retriever(self, **kwargs):
        """
        Tạo LangChain retriever từ vector store
        
        Returns:
            VectorStoreRetriever
        """
        return self.vector_store.as_retriever(**kwargs)

# Factory functions
def create_qdrant_manager(
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    collection_name: Optional[str] = None
) -> QdrantManager:
    """
    Factory function để tạo QdrantManager
    """
    return QdrantManager(
        host=host,
        port=port,
        api_key=api_key,
        collection_name=collection_name
    )

def create_vector_store_manager(
    qdrant_manager: QdrantManager,
    embedding_manager: EmbeddingManager
) -> QdrantVectorStoreManager:
    """
    Factory function để tạo QdrantVectorStoreManager
    """
    return QdrantVectorStoreManager(qdrant_manager, embedding_manager)
