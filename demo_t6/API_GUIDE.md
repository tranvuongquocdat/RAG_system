# API Guide for FastAPI Server

This document describes the available API endpoints in `server.py` for interacting with the Knowledge Base (KB) system.

---

## 1. Add PDF to Knowledge Base

- **Endpoint:** `/add_pdf`
- **Method:** `POST`
- **Description:** Upload a PDF file. The server extracts text and adds it to the KB.
- **Request:**
  - Form-data: `file` (PDF file)
- **Response:**
  - `{"message": "File '<filename>.txt' added successfully"}`

**Example using curl:**
```bash
curl -F "file=@yourfile.pdf" http://localhost:8000/add_pdf
```

---

## 2. List Files in Knowledge Base

- **Endpoint:** `/list_files`
- **Method:** `GET`
- **Description:** Lists all files currently stored in the KB and in the text directory, with metadata.
- **Response:**
  ```json
  {
    "kb_files": [ ... ],
    "text_files": [ ... ],
    "summary": {
      "total_documents": 2,
      "total_chunks": 123
    }
  }
  ```

**Example:**
```bash
curl http://localhost:8000/list_files
```

---

## 3. Delete a File from Knowledge Base

- **Endpoint:** `/delete_file/{filename}`
- **Method:** `DELETE`
- **Description:** Deletes a `.txt` file from the text directory and rebuilds the KB.
- **Path Parameter:**
  - `filename`: Name of the `.txt` file to delete (e.g., `document.txt`)
- **Response:**
  - `{"message": "File '<filename>' deleted and KB updated"}`

**Example:**
```bash
curl -X DELETE http://localhost:8000/delete_file/document.txt
```

---

## 4. Query the Knowledge Base

- **Endpoint:** `/query`
- **Method:** `POST`
- **Description:** Query the KB using a question. Returns the answer, mode used, sources, and query time.
- **Request (JSON):**
  ```json
  {
    "query": "Your question here",
    "mode": "hybrid", // optional: see modes below
    "instruction": "Instruction for the model" // optional
  }
  ```
- **Response:**
  ```json
  {
    "result": "Answer from KB",
    "mode_used": "hybrid",
    "sources": ["file1.txt", ...],
    "query_time": "2024-06-01T12:34:56"
  }
  ```

**Example:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is pricing optimization?", "mode": "hybrid"}'
```

### Query Modes

| Mode    | Description (EN)                                                                 | Description (VI)                                                        |
|---------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| hybrid  | Combines local and global search for best results (Recommended)                  | Kết hợp tìm kiếm cục bộ và toàn cục, cho kết quả tốt nhất               |
| mix     | Integrates knowledge graph and vector retrieval                                  | Tích hợp knowledge graph và vector retrieval                            |
| local   | Focuses on context-dependent information                                         | Tập trung vào thông tin phụ thuộc ngữ cảnh cụ thể                       |
| global  | Uses global knowledge, suitable for overview questions                           | Sử dụng kiến thức toàn cục, phù hợp cho câu hỏi tổng quan               |
| naive   | Basic, fast search, less accurate                                                | Tìm kiếm cơ bản, nhanh nhưng kém chính xác                              |

**Example with mode:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tóm tắt tài liệu", "mode": "local"}'
```

---

## 5. Get Knowledge Base Status

- **Endpoint:** `/kb_status`
- **Method:** `GET`
- **Description:** Returns the status of the KB, including number of documents, total chunks, storage file info, and working directory.
- **Response:**
  ```json
  {
    "status": "active",
    "documents": 2,
    "total_chunks": 123,
    "storage_files": { ... },
    "working_directory": "./lightrag_kb"
  }
  ```

**Example:**
```bash
curl http://localhost:8000/kb_status
```

---

## 6. Set/Update Gemini API Key

- **Endpoint:** `/set_gemini_api_key`
- **Method:** `POST`
- **Description:** Set or update the Gemini API key at runtime.
- **Request (JSON):**
  ```json
  {
    "api_key": "your-new-gemini-api-key"
  }
  ```
- **Response:**
  - `{"message": "Gemini API key updated successfully"}`

**Example:**
```bash
curl -X POST http://localhost:8000/set_gemini_api_key \
  -H "Content-Type: application/json" \
  -d '{"api_key": "your-new-gemini-api-key"}'
```

---

## Notes

- All endpoints return JSON responses.
- For PDF upload, only `.pdf` files are accepted.
- The server must be running (default: `http://localhost:8000`).
- For best results, use the appropriate query mode for your question.

---

**Author:** Auto-generated from `server.py` and `demo_ui.py`
