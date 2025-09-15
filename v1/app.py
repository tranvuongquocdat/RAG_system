"""
Main Application for RAG System
Ứng dụng chính để chạy hệ thống RAG với CLI và API
"""

import os
import sys
import asyncio
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import shutil

# Import các modules của hệ thống
from config import get_config, validate_config
from utils import setup_logging, format_file_size, create_directory_if_not_exists
from embedding import create_embedding_manager
from database import create_qdrant_manager
from ingestion import create_ingestion_pipeline
from retrieval import create_retrieval_engine, create_context_builder
from generation import create_rag_chain, create_conversational_rag
from model import create_llm_manager

# Pydantic models cho API
class QueryRequest(BaseModel):
    question: str = Field(..., description="Câu hỏi từ người dùng")
    max_sources: Optional[int] = Field(5, description="Số lượng sources tối đa")
    filters: Optional[Dict[str, Any]] = Field(None, description="Bộ lọc cho retrieval")

class QueryResponse(BaseModel):
    answer: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]

class IngestionRequest(BaseModel):
    file_path: str = Field(..., description="Đường dẫn file cần ingest")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata bổ sung")

class IngestionResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    processing_time: float

class ConversationRequest(BaseModel):
    conversation_id: str = Field(..., description="ID cuộc hội thoại")
    question: str = Field(..., description="Câu hỏi")
    use_context: Optional[bool] = Field(True, description="Sử dụng context hội thoại")

class SystemStats(BaseModel):
    total_documents: int
    collection_info: Dict[str, Any]
    embedding_cache_size: int
    system_status: str

class RAGSystem:
    """
    Hệ thống RAG chính
    """
    
    def __init__(self):
        """Khởi tạo hệ thống RAG"""
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
        
        # Validate configuration
        if not validate_config():
            raise ValueError("Cấu hình không hợp lệ")
        
        self.logger.info("🚀 Đang khởi tạo RAG System...")
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("✅ RAG System đã khởi tạo thành công")
    
    def _initialize_components(self) -> None:
        """Khởi tạo các components của hệ thống"""
        try:
            # Initialize embedding manager
            self.embedding_manager = create_embedding_manager()
            
            # Initialize Qdrant manager
            self.qdrant_manager = create_qdrant_manager()
            
            # Initialize ingestion pipeline
            self.ingestion_pipeline = create_ingestion_pipeline(
                self.embedding_manager, 
                self.qdrant_manager
            )
            
            # Initialize LLM manager
            self.llm_manager = create_llm_manager()
            
            # Initialize retrieval engine
            self.retrieval_engine = create_retrieval_engine(
                self.embedding_manager,
                self.qdrant_manager
            )
            
            # Initialize context builder
            self.context_builder = create_context_builder(self.retrieval_engine)
            
            # Initialize RAG chain
            self.rag_chain = create_rag_chain(
                self.llm_manager,
                self.retrieval_engine,
                self.context_builder
            )
            
            # Initialize conversational RAG
            self.conversational_rag = create_conversational_rag(self.rag_chain)
            
            self.logger.info("✅ Tất cả components đã được khởi tạo")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khởi tạo components: {str(e)}")
            raise
    
    def query(
        self,
        question: str,
        max_sources: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Xử lý câu hỏi và trả về kết quả
        
        Args:
            question: Câu hỏi từ người dùng
            max_sources: Số lượng sources tối đa
            filters: Bộ lọc cho retrieval
            
        Returns:
            Dictionary chứa kết quả
        """
        try:
            result = self.rag_chain.generate_answer(
                question=question,
                max_sources=max_sources,
                filters=filters
            )
            
            # Convert sources to dict format
            sources_dict = []
            for source in result.sources:
                sources_dict.append({
                    'id': source.id,
                    'content': source.content[:200] + "..." if len(source.content) > 200 else source.content,
                    'score': source.score,
                    'filename': source.filename,
                    'document_type': source.document_type,
                    'chunk_id': source.chunk_id,
                    'source': source.source
                })
            
            return {
                'answer': result.answer,
                'confidence_score': result.confidence_score,
                'sources': sources_dict,
                'processing_time': result.processing_time,
                'response_type': result.response_type.value,
                'metadata': result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi xử lý query: {str(e)}")
            raise
    
    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest một file vào hệ thống
        
        Args:
            file_path: Đường dẫn file
            metadata: Metadata bổ sung
            
        Returns:
            Dictionary chứa kết quả ingestion
        """
        try:
            import time
            start_time = time.time()
            
            # Process and ingest file
            documents = self.ingestion_pipeline.process_single_file(file_path, metadata)
            
            if documents:
                success = self.ingestion_pipeline.ingest_documents(documents)
                processing_time = time.time() - start_time
                
                return {
                    'success': success,
                    'message': f"Đã ingest {len(documents)} documents từ file {Path(file_path).name}",
                    'documents_processed': len(documents),
                    'processing_time': processing_time
                }
            else:
                return {
                    'success': False,
                    'message': f"Không thể xử lý file {Path(file_path).name}",
                    'documents_processed': 0,
                    'processing_time': time.time() - start_time
                }
                
        except Exception as e:
            self.logger.error(f"❌ Lỗi ingest file: {str(e)}")
            raise
    
    def ingest_directory(
        self,
        directory_path: str,
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest toàn bộ directory
        
        Args:
            directory_path: Đường dẫn thư mục
            recursive: Có xử lý đệ quy không
            
        Returns:
            Dictionary chứa kết quả ingestion
        """
        try:
            import time
            start_time = time.time()
            
            success = self.ingestion_pipeline.ingest_directory(directory_path, recursive)
            processing_time = time.time() - start_time
            
            # Get stats after ingestion
            stats = self.ingestion_pipeline.get_ingestion_stats()
            
            return {
                'success': success,
                'message': f"Đã ingest directory {directory_path}",
                'total_documents': stats.get('total_documents', 0),
                'processing_time': processing_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi ingest directory: {str(e)}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê hệ thống
        
        Returns:
            Dictionary chứa thống kê
        """
        try:
            ingestion_stats = self.ingestion_pipeline.get_ingestion_stats()
            collection_info = self.qdrant_manager.get_collection_info()
            
            return {
                'total_documents': ingestion_stats.get('total_documents', 0),
                'collection_info': collection_info,
                'embedding_cache_size': self.embedding_manager.get_cache_size(),
                'system_status': 'healthy',
                'config': {
                    'model_provider': self.config.get_primary_llm_provider(),
                    'embedding_model': self.config.model.gemini_embedding_model,
                    'vector_dimension': ingestion_stats.get('vector_dimension', 0),
                    'collection_name': self.config.qdrant.collection_name
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi lấy system stats: {str(e)}")
            return {
                'total_documents': 0,
                'collection_info': {},
                'embedding_cache_size': 0,
                'system_status': 'error',
                'error': str(e)
            }

# Global RAG system instance
rag_system: Optional[RAGSystem] = None

def get_rag_system() -> RAGSystem:
    """Dependency để lấy RAG system instance"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system

# FastAPI app
app = FastAPI(
    title="Enterprise RAG System",
    description="Hệ thống RAG cho doanh nghiệp sử dụng LangChain và Qdrant",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Enterprise RAG System API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        system = get_rag_system()
        stats = system.get_system_stats()
        return {"status": "healthy", "stats": stats}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    system: RAGSystem = Depends(get_rag_system)
):
    """Query documents endpoint"""
    try:
        result = system.query(
            question=request.question,
            max_sources=request.max_sources,
            filters=request.filters
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/file", response_model=IngestionResponse)
async def ingest_file_endpoint(
    request: IngestionRequest,
    system: RAGSystem = Depends(get_rag_system)
):
    """Ingest file endpoint"""
    try:
        result = system.ingest_file(
            file_path=request.file_path,
            metadata=request.metadata
        )
        
        return IngestionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    system: RAGSystem = Depends(get_rag_system)
):
    """Upload và ingest file endpoint"""
    try:
        import json
        import tempfile
        
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata != "{}" else {}
        except json.JSONDecodeError:
            metadata_dict = {}
        
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Ingest the file
            result = system.ingest_file(temp_path, metadata_dict)
            return result
        finally:
            # Clean up temp file
            os.unlink(temp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation")
async def conversation_endpoint(
    request: ConversationRequest,
    system: RAGSystem = Depends(get_rag_system)
):
    """Conversational query endpoint"""
    try:
        result = system.conversational_rag.chat(
            conversation_id=request.conversation_id,
            question=request.question,
            use_conversation_context=request.use_context
        )
        
        # Convert to dict format
        sources_dict = []
        for source in result.sources:
            sources_dict.append({
                'id': source.id,
                'content': source.content[:200] + "..." if len(source.content) > 200 else source.content,
                'score': source.score,
                'filename': source.filename,
                'document_type': source.document_type
            })
        
        return {
            'answer': result.answer,
            'confidence_score': result.confidence_score,
            'sources': sources_dict,
            'processing_time': result.processing_time,
            'conversation_id': request.conversation_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=SystemStats)
async def get_stats(system: RAGSystem = Depends(get_rag_system)):
    """Get system statistics"""
    try:
        stats = system.get_system_stats()
        return SystemStats(
            total_documents=stats['total_documents'],
            collection_info=stats['collection_info'],
            embedding_cache_size=stats['embedding_cache_size'],
            system_status=stats['system_status']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    system: RAGSystem = Depends(get_rag_system)
):
    """Delete a document"""
    try:
        success = system.qdrant_manager.delete_documents([document_id])
        return {"success": success, "message": f"Document {document_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CLI Functions
def run_cli():
    """Chạy CLI interface"""
    config = get_config()
    logger = setup_logging(config.app.log_level, config.app.log_file)
    
    print("🚀 Enterprise RAG System CLI")
    print("=" * 50)
    
    try:
        # Initialize system
        system = RAGSystem()
        
        while True:
            print("\nCác lệnh có sẵn:")
            print("1. query - Đặt câu hỏi")
            print("2. ingest - Ingest file/directory")
            print("3. stats - Xem thống kê hệ thống")
            print("4. exit - Thoát")
            
            choice = input("\nChọn lệnh (1-4): ").strip()
            
            if choice == "1":
                question = input("Nhập câu hỏi: ").strip()
                if question:
                    print("\n🔍 Đang xử lý...")
                    result = system.query(question)
                    
                    print(f"\n✅ Câu trả lời (Confidence: {result['confidence_score']:.2f}):")
                    print("-" * 50)
                    print(result['answer'])
                    print(f"\n📚 Sources: {len(result['sources'])} documents")
                    print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
            
            elif choice == "2":
                path = input("Nhập đường dẫn file/directory: ").strip()
                if path and os.path.exists(path):
                    print("\n📥 Đang ingest...")
                    
                    if os.path.isfile(path):
                        result = system.ingest_file(path)
                    else:
                        result = system.ingest_directory(path)
                    
                    if result['success']:
                        print(f"✅ {result['message']}")
                        print(f"📄 Documents: {result.get('documents_processed', result.get('total_documents', 0))}")
                        print(f"⏱️ Time: {result['processing_time']:.2f}s")
                    else:
                        print(f"❌ {result['message']}")
                else:
                    print("❌ Đường dẫn không tồn tại")
            
            elif choice == "3":
                print("\n📊 Thống kê hệ thống:")
                print("-" * 30)
                stats = system.get_system_stats()
                print(f"📄 Total documents: {stats['total_documents']}")
                print(f"🧠 Embedding cache: {stats['embedding_cache_size']}")
                print(f"🔧 Status: {stats['system_status']}")
                print(f"🤖 Model: {stats['config']['model_provider']}")
                print(f"🔢 Vector dimension: {stats['config']['vector_dimension']}")
            
            elif choice == "4":
                print("👋 Tạm biệt!")
                break
            
            else:
                print("❌ Lựa chọn không hợp lệ")
                
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")
    except Exception as e:
        logger.error(f"❌ Lỗi CLI: {str(e)}")
        print(f"❌ Lỗi: {str(e)}")

def run_api():
    """Chạy API server"""
    config = get_config()
    
    print("🚀 Starting Enterprise RAG System API...")
    print(f"📡 Host: {config.app.api_host}:{config.app.api_port}")
    print("📖 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host=config.app.api_host,
        port=config.app.api_port,
        reload=False,
        log_level=config.app.log_level.lower()
    )

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enterprise RAG System")
    parser.add_argument(
        "mode",
        choices=["cli", "api"],
        help="Chế độ chạy: cli hoặc api"
    )
    parser.add_argument(
        "--config-check",
        action="store_true",
        help="Kiểm tra cấu hình"
    )
    
    args = parser.parse_args()
    
    # Check configuration
    if args.config_check:
        print("🔧 Kiểm tra cấu hình...")
        if validate_config():
            print("✅ Cấu hình hợp lệ")
        else:
            print("❌ Cấu hình không hợp lệ")
            sys.exit(1)
        return
    
    # Run in selected mode
    try:
        if args.mode == "cli":
            run_cli()
        elif args.mode == "api":
            run_api()
    except Exception as e:
        print(f"❌ Lỗi khởi động: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()


