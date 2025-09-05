import os
import shutil
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pypdf import PdfReader
import google.generativeai as genai
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from typing import List
import asyncio

# Configure Gemini API
genai.configure(api_key="AIzaSyBa0RIPexzkRmQU-hXizc54O0yypUyJCF8")

# Directories
WORKING_DIR = "./lightrag_kb"
KB_TEXT_DIR = "./kb_texts"
os.makedirs(KB_TEXT_DIR, exist_ok=True)

# Gemini LLM function (async)
async def gemini_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
    model = genai.GenerativeModel(
        'gemini-2.5-flash-lite',
        generation_config={"temperature": kwargs.get("temperature", 0.0)}
    )
    full_prompt = system_prompt + "\n" + prompt if system_prompt else prompt
    contents = []
    for msg in history_messages:
        role = 'user' if msg.get('role') == 'user' else 'model'
        contents.append({'role': role, 'parts': [msg.get('content', '')]})
    contents.append({'role': 'user', 'parts': [full_prompt]})
    response = model.generate_content(contents)
    return response.text

# Gemini embedding function (async, batch support)
async def gemini_embed(texts: list[str]) -> np.ndarray:
    result = genai.embed_content(
        model='models/text-embedding-004',
        content=texts,
        task_type="retrieval_document",
    )
    embeddings = result['embedding']
    
    # Ensure consistent 2D array format: (num_texts, embedding_dim)
    if len(texts) == 1:
        # For single text, embeddings is 1D, convert to 2D
        return np.array(embeddings).reshape(1, -1)
    else:
        # For multiple texts, embeddings should already be 2D
        return np.array(embeddings)

# Embedding dimension for text-embedding-004
embedding_dim = 768

# Initialize LightRAG
rag = None

async def init_rag():
    global rag
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=gemini_embed
        )
    )
    
    # Initialize storages as required by LightRAG
    await rag.initialize_storages()
    
    # Initialize pipeline status
    from lightrag.kg.shared_storage import initialize_pipeline_status
    await initialize_pipeline_status()
    
    all_text = ""
    for filename in os.listdir(KB_TEXT_DIR):
        with open(os.path.join(KB_TEXT_DIR, filename), 'r', encoding='utf-8') as f:
            all_text += f"\n\n--- File: {filename} ---\n\n" + f.read()
    if all_text:
        await rag.ainsert(all_text)  # Use async insert

# Start RAG initialization
# init_rag() # This line is removed as init_rag is now async

# FastAPI app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await init_rag()

# API Endpoints
@app.post("/add_pdf")
async def add_pdf(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        print("Reading file contents")
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print("Extracted text from PDF")
        filename = file.filename.replace('.pdf', '.txt')
        with open(os.path.join(KB_TEXT_DIR, filename), 'w', encoding='utf-8') as f:
            f.write(text)
        print("Saved text file")
        global rag
        await init_rag()  # Use async init_rag
        return JSONResponse(content={"message": f"File '{filename}' added and KB rebuilt"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding file: {str(e)}")

@app.get("/list_files")
async def list_files():
    print("Listing files")
    files = os.listdir(KB_TEXT_DIR)
    return JSONResponse(content={"files": files})

@app.delete("/delete_file/{filename}")
async def delete_file(filename: str):
    print(f"Deleting file: {filename}")
    file_path = os.path.join(KB_TEXT_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File deleted")
        global rag
        await init_rag()  # Use async init_rag
        return JSONResponse(content={"message": f"File '{filename}' deleted and KB rebuilt"})
    raise HTTPException(status_code=404, detail="File not found")

@app.post("/query")
async def query_kb(query: dict):
    print(f"Querying KB: {query}")
    q = query.get("query")
    mode = query.get("mode", "hybrid")  # Default to hybrid
    instruction = query.get("instruction", "Trả lời tập trung vào câu hỏi, không trả lời thông tin không liên quan, nhất là chú ý đến các keyword, tên dự án, đừng để bị nhầm giữa các keyword, và cùng không kiểu hỏi requirements trả lời thêm price, trả lời chính xác tới câu hỏi của người dùng")  # Optional instruction
    
    if not q:
        raise HTTPException(status_code=400, detail="No query provided")
    try:
        print(f"Querying KB with mode: {mode}")
        
        # Add instruction to the query if provided
        final_query = q
        if instruction.strip():
            final_query = f"Instruction: {instruction.strip()}\n\nQuery: {q}"
        
        result = await rag.aquery(final_query, param=QueryParam(mode=mode))
        return JSONResponse(content={"result": result, "mode_used": mode})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)