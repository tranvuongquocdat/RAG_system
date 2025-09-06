Auto-generated API Documentation at: `http://localhost:8000/docs`

## API Endpoints

### 1. Health Check
**GET** `/`
- **Description**: Check server status
- **Response**:
```json
{
  "message": "RAG System Server is running",
  "documents_count": 5,
  "total_chunks": 150
}
```

### 2. Add Document
**POST** `/documents/`
- **Description**: Upload and add document to Knowledge Base
- **Headers**: 
  - `X-GEMINI-API-KEY`: API key (optional, if not set in environment)
- **Body**: Form-data with file upload
- **Supported formats**: PDF, DOCX, TXT
- **Response**:
```json
{
  "message": "Document added successfully",
  "document_id": "abc123def456",
  "chunks_count": 25
}
```

#### Example curl:
```bash
curl -X POST "http://localhost:8000/documents/" \
  -H "X-GEMINI-API-KEY: your-api-key" \
  -F "file=@document.pdf"
```

### 3. List Documents
**GET** `/documents/`
- **Description**: Get list of all documents in Knowledge Base
- **Response**:
```json
[
  {
    "id": "abc123def456",
    "name": "document.pdf",
    "created_at": "2024-01-01T10:00:00",
    "chunks_count": 25
  }
]
```

### 4. Delete Document
**DELETE** `/documents/{document_id}`
- **Description**: Remove document from Knowledge Base
- **Parameters**: 
  - `document_id`: ID of document to delete
- **Response**:
```json
{
  "message": "Document deleted successfully"
}
```

### 5. Query Knowledge Base
**POST** `/query/`
- **Description**: Ask AI questions based on uploaded documents
- **Headers**: 
  - `X-GEMINI-API-KEY`: API key (optional)
- **Body**:
```json
{
  "query": "Your question here",
  "instruction": "Custom instruction for AI (optional)",
  "top_k": 5
}
```
- **Response**:
```json
{
  "answer": "AI generated answer",
  "relevant_chunks": ["chunk1", "chunk2"],
  "sources": ["document1.pdf", "document2.docx"]
}
```

#### Example curl:
```bash
curl -X POST "http://localhost:8000/query/" \
  -H "Content-Type: application/json" \
  -H "X-GEMINI-API-KEY: your-api-key" \
  -d '{
    "query": "What is project ABC about?",
    "top_k": 3
  }'
```

## Data Models

### Document Model
```json
{
  "id": "string",           // MD5 hash of content
  "name": "string",         // Original filename
  "content": "string",      // Full text content
  "chunks": ["string"],     // List of text chunks
  "embeddings": [[float]],  // Vector embeddings
  "created_at": "string"    // Creation timestamp (ISO format)
}
```

### Query Request Model
```json
{
  "query": "string",        // Question (required)
  "instruction": "string",  // Custom instruction (optional)
  "top_k": 5               // Number of relevant chunks (default: 5)
}
```

## How It Works

### 1. Document Processing
1. Upload file (PDF, DOCX, TXT)
2. Extract text from file
3. Split text into chunks (1000 words, 200 words overlap)
4. Generate embeddings for each chunk using Gemini
5. Store in memory and JSON file

### 2. Query Processing
1. Generate embedding for the question
2. Calculate cosine similarity with all chunks
3. Retrieve top_k most similar chunks
4. Create context from these chunks
5. Use Gemini to generate answer

## Configuration

### Adjustable Parameters
- `chunk_size`: 1000 words (in `split_text_into_chunks` function)
- `overlap`: 200 words
- `top_k`: 5 chunks default
- `embedding_model`: "models/text-embedding-004"
- `generation_model`: "gemini-2.0-flash-exp"

### Storage File
- `knowledge_base.json`: Stores all documents and embeddings

## Error Handling

### Common Errors
- **400**: Unsupported file format, no valid text found
- **409**: Document already exists
- **404**: Document not found
- **500**: Internal server error

### Troubleshooting
1. **Invalid API Key**: Check GEMINI_API_KEY
2. **Unreadable File**: Ensure file is not corrupted
3. **Memory Issues**: Reduce chunk_size or number of documents
4. **Embedding Failures**: Check internet connection and API quota

## Security

### Recommendations
- Don't hardcode API key in code
- Use HTTPS in production
- Implement authentication/authorization
- Validate input files before processing
- Rate limiting for API calls

## Performance

### Optimization
- Cache embeddings to avoid recalculation
- Use database instead of JSON file for production
- Implement async processing for large files
- Batch embedding requests

### Limitations
- File size: Depends on available RAM
- Concurrent requests: Depends on Gemini API limits
- Storage: JSON file can become large with many documents

## Usage Examples


### JavaScript Client Example
```javascript
// Upload document
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/documents/', {
    method: 'POST',
    body: formData,
    headers: {
        'X-GEMINI-API-KEY': 'your-api-key'
    }
})
.then(response => response.json())
.then(data => console.log(data));

// Query
fetch('http://localhost:8000/query/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-GEMINI-API-KEY': 'your-api-key'
    },
    body: JSON.stringify({
        query: 'What is project ABC about?',
        top_k: 3
    })
})
.then(response => response.json())
.then(data => console.log(data.answer));
```

## Advanced Features

### Custom Instructions
You can provide custom instructions to modify AI behavior:

```json
{
  "query": "Explain the technical architecture",
  "instruction": "Provide a detailed technical explanation with bullet points. Focus on system design and implementation details.",
  "top_k": 5
}
```

### Default AI Instructions
The system uses Vietnamese by default with these instructions:
- Answer based only on provided context
- Be concise and focused on the query
- Pay attention to proper names and project names
- Don't expand beyond what's asked
- Prioritize short, clear answers