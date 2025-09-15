# Enterprise RAG System

Hệ thống RAG (Retrieval-Augmented Generation) hoàn chỉnh cho doanh nghiệp, được xây dựng với **LangChain** và **Qdrant** vector database.

## 🎯 Tính năng chính

- **🤖 AI Models**: Hỗ trợ Gemini 2.0 Flash và các LLM khác
- **🔍 Vector Search**: Qdrant database với hybrid search
- **📄 Multi-format**: Hỗ trợ PDF, DOCX, Excel, PowerPoint, Markdown
- **🔒 Enterprise-ready**: Bảo mật, scalable, monitoring
- **🌐 API & CLI**: Giao diện API REST và command-line
- **💬 Conversational**: Hỗ trợ hội thoại có context

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   Embedding     │    │   Vector DB     │
│  (PDF, DOCX,    │───▶│   (Gemini)      │───▶│   (Qdrant)      │
│   Excel, etc.)  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Response      │◀───│   Generation    │◀───│   Retrieval     │
│                 │    │   (Gemini LLM)  │    │   Engine        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Yêu cầu hệ thống

- Python 3.8+
- Docker & Docker Compose
- 4GB RAM (tối thiểu)
- 10GB disk space

## 🚀 Cài đặt nhanh

### 1. Clone và cài đặt dependencies

```bash
git clone <repository-url>
cd RAG_system/v1

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Cấu hình environment

```bash
# Copy template và chỉnh sửa
cp env_template.txt .env

# Chỉnh sửa .env file với API keys của bạn
nano .env
```

**Cấu hình tối thiểu trong `.env`:**
```env
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
LOG_LEVEL=INFO
```

### 3. Khởi động Qdrant database

```bash
# Khởi động Qdrant
docker-compose up -d qdrant

# Kiểm tra status
docker-compose ps
```

### 4. Chạy hệ thống

#### CLI Mode:
```bash
python app.py cli
```

#### API Mode:
```bash
python app.py api

# API docs tại: http://localhost:8000/docs
```

## 📖 Hướng dẫn sử dụng

### CLI Commands

```bash
# Kiểm tra cấu hình
python app.py cli --config-check

# Chạy CLI interactive
python app.py cli
```

**Trong CLI:**
- `1` - Đặt câu hỏi
- `2` - Ingest file/directory  
- `3` - Xem thống kê hệ thống
- `4` - Thoát

### API Endpoints

#### Query Documents
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Quy trình onboarding nhân viên mới như thế nào?",
    "max_sources": 5
  }'
```

#### Upload & Ingest File
```bash
curl -X POST "http://localhost:8000/ingest/upload" \
  -F "file=@document.pdf" \
  -F "metadata={\"department\": \"HR\"}"
```

#### Conversational Query
```bash
curl -X POST "http://localhost:8000/conversation" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "user123",
    "question": "Chính sách nghỉ phép như thế nào?",
    "use_context": true
  }'
```

#### System Stats
```bash
curl -X GET "http://localhost:8000/stats"
```

### Python SDK

```python
from app import RAGSystem

# Khởi tạo hệ thống
system = RAGSystem()

# Query
result = system.query("Chính sách bảo mật công ty?")
print(result['answer'])

# Ingest file
result = system.ingest_file("path/to/document.pdf")
print(f"Processed {result['documents_processed']} documents")

# Conversational
conv_rag = system.conversational_rag
result = conv_rag.chat("user123", "Quy trình nghỉ việc?")
print(result.answer)
```

## 📁 Cấu trúc dự án

```
RAG_system/v1/
├── app.py              # Main application
├── config.py           # Configuration management  
├── utils.py            # Utility functions
├── embedding.py        # Gemini embedding manager
├── model.py            # LLM manager
├── database.py         # Qdrant operations
├── ingestion.py        # Document processing
├── retrieval.py        # Document retrieval
├── generation.py       # Answer generation
├── requirements.txt    # Dependencies
├── docker-compose.yml  # Qdrant setup
├── env_template.txt    # Environment template
└── README.md          # This file
```

## 🔧 Cấu hình nâng cao

### Vector Database
```env
# Qdrant settings
QDRANT_COLLECTION_NAME=enterprise_documents
VECTOR_SIZE=768
DISTANCE_METRIC=cosine

# Performance tuning
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.7
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Security
```env
# Enable authentication
ENABLE_AUTH=true
JWT_SECRET=your-super-secret-key
JWT_EXPIRATION_HOURS=24

# API rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

### Multi-service Setup
```bash
# Với Redis cache
docker-compose --profile cache up -d

# Với PostgreSQL metadata
docker-compose --profile metadata-db up -d

# Với Web UI
docker-compose --profile web-ui up -d
```

## 📊 Monitoring & Logs

### Logs
```bash
# Xem logs realtime
tail -f logs/rag_system.log

# Logs với level khác nhau
export LOG_LEVEL=DEBUG
python app.py cli
```

### Health Check
```bash
curl http://localhost:8000/health
```

### System Stats
- Total documents indexed
- Embedding cache size
- Vector database status
- Processing performance

## 🔍 Troubleshooting

### Lỗi thường gặp

**1. Qdrant connection failed**
```bash
# Kiểm tra Qdrant status
docker-compose ps qdrant

# Restart Qdrant
docker-compose restart qdrant
```

**2. Gemini API errors**
```bash
# Kiểm tra API key
python -c "from config import get_config; print(get_config().model.gemini_api_key[:10] + '...')"

# Test API connection
curl -H "x-goog-api-key: YOUR_API_KEY" \
  https://generativelanguage.googleapis.com/v1beta/models
```

**3. Memory issues**
```bash
# Giảm batch size
export CHUNK_SIZE=500
export RAG_TOP_K=3

# Clear embedding cache
# Trong Python: embedding_manager.clear_cache()
```

**4. Slow performance**
```bash
# Enable caching
docker-compose --profile cache up -d

# Optimize chunk size
export CHUNK_SIZE=800
export CHUNK_OVERLAP=100
```

## 🚀 Production Deployment

### Docker Production
```bash
# Build production image
docker build -t rag-system:prod .

# Run with production config
docker run -d \
  --name rag-system \
  -p 8000:8000 \
  -v ./data:/app/data \
  -v ./.env:/app/.env \
  rag-system:prod
```

### Environment Variables
```env
# Production settings
DEBUG=false
LOG_LEVEL=INFO
ENABLE_AUTH=true

# Security
JWT_SECRET=complex-production-secret
RATE_LIMIT_ENABLED=true

# Performance
WORKERS=4
MAX_CONNECTIONS=1000
```

### Backup & Recovery
```bash
# Backup Qdrant data
docker-compose exec qdrant tar -czf /backup/qdrant-$(date +%Y%m%d).tar.gz /qdrant/storage

# Backup configuration
cp .env .env.backup
```

## 📚 API Documentation

Sau khi khởi động API server, truy cập:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

MIT License - xem file LICENSE để biết chi tiết.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: Wiki
- **Email**: support@company.com

---

**Phiên bản**: v1.0.0  
**Cập nhật**: 2025-01-12  
**Tác giả**: PathTech Team


