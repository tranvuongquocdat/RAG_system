import os
import logging
import asyncio
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoTokenizer, AutoModel
from groq import Groq
from pathlib import Path
from tqdm import tqdm
import torch
from tika import parser

# Setup logging
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# Directory configuration
WORKING_DIR = "./rag_project"
DATA_PDF_DIR = "./data_pdf"
os.makedirs(WORKING_DIR, exist_ok=True)

# Load model once globally
model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# BGE-M3 embedding function
async def bge_m3_embedding(texts):
    """Local BGE-M3 embedding function"""
    # Handle both single string and list of strings
    if isinstance(texts, str):
        texts = [texts]
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    
    # Always return list of vectors, even for single input
    return embeddings.tolist()

# Groq LLM function
async def groq_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Groq LLM function with proper error handling"""
    client = Groq(api_key="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")
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=1,
            max_completion_tokens=4080,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error calling Groq LLM: {e}")
        return "Error occurred while processing the request."

# PDF text extraction function
def extract_pdf_texts(pdf_dir):
    """Extract text from PDF files using Tika"""
    documents = []
    sources = []
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        logging.warning(f"No PDF files found in directory {pdf_dir}")
        return documents, sources
    
    logging.info(f"Starting extraction of {total_files} PDF files...")
    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs", unit="file"):
        try:
            parsed = parser.from_file(str(pdf_file), requestOptions={'timeout': 300})
            text = parsed.get("content", "").strip()
            if not text:
                logging.warning(f"File {pdf_file.name} contains no text content or may be a scanned PDF.")
                continue
            source = pdf_file.stem  # Get filename without extension
            documents.append(text)
            sources.append(source)
            logging.info(f"Processed: {pdf_file.name}")
        except Exception as e:
            logging.error(f"Error processing {pdf_file}: {e}")
    return documents, sources

# Response formatting with citations
def format_response_with_citation(response, sources):
    """Format response with source citations"""
    if not sources:
        return response
    
    unique_sources = list(set(sources))  # Remove duplicates
    citations = ", ".join([f"({source})" for source in unique_sources])
    return f"{response}\n\nSources: {citations}"

# Initialize LightRAG
async def initialize_rag():
    """Initialize LightRAG with proper configuration"""
    logging.info("Initializing LightRAG...")
    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=groq_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,  # BGE-M3 embedding dimension
                max_token_size=512,
                func=bge_m3_embedding
            ),
            # Additional configuration for better performance
            chunk_token_size=1200,
            chunk_overlap_token_size=100,
            entity_extract_max_gleaning=1,
            llm_model_max_async=4,
            embedding_batch_num=32,
            embedding_func_max_async=16,
        )
        
        # CRITICAL: Both initialization calls are required!
        await rag.initialize_storages()  # Initialize storage backends
        await initialize_pipeline_status()  # Initialize processing pipeline
        
        logging.info("LightRAG successfully initialized.")
        return rag
    except Exception as e:
        logging.error(f"Error initializing LightRAG: {e}")
        return None

# Main function
async def main():
    """Main execution function"""
    # Initialize RAG
    rag = await initialize_rag()
    if not rag:
        logging.error("Failed to initialize LightRAG. Exiting program.")
        return
    
    try:
        # Extract text from PDFs
        logging.info("Starting PDF text extraction...")
        documents, sources = extract_pdf_texts(DATA_PDF_DIR)
        if not documents:
            logging.error("No documents were extracted. Exiting program.")
            return
        
        logging.info(f"Successfully extracted {len(documents)} documents.")
        
        # Insert documents into LightRAG with progress tracking
        logging.info("Starting document insertion into LightRAG...")
        for doc, source in tqdm(zip(documents, sources), total=len(documents), desc="Inserting documents", unit="document"):
            try:
                # Insert document with source information
                await rag.ainsert(f"{doc}\nSource: {source}")
            except Exception as e:
                logging.error(f"Error inserting document {source}: {e}")
        
        logging.info("Document insertion completed.")
        
        # Example queries with different modes
        query = "How to cope with anxiety?"
        modes = ["naive", "local", "global", "hybrid"]
        
        logging.info("Starting queries...")
        for mode in tqdm(modes, desc="Querying modes", unit="mode"):
            logging.info(f"Querying with mode: {mode}...")
            try:
                result = await rag.aquery(query, param=QueryParam(mode=mode))
                formatted_result = format_response_with_citation(result, sources)
                print(f"\n{'='*50}")
                print(f"RESULT WITH MODE: {mode.upper()}")
                print(f"{'='*50}")
                print(f"{formatted_result}")
                print(f"{'='*50}\n")
            except Exception as e:
                logging.error(f"Error querying with mode {mode}: {e}")
        
        logging.info("Query execution completed.")
        
    except Exception as e:
        logging.error(f"An error occurred in main execution: {e}")
    finally:
        # Clean up resources
        if rag:
            try:
                await rag.finalize_storages()
                logging.info("RAG resources cleaned up successfully.")
            except Exception as e:
                logging.error(f"Error during cleanup: {e}")

# Run the program
if __name__ == "__main__":
    # Check if data directory exists
    if not os.path.exists(DATA_PDF_DIR):
        os.makedirs(DATA_PDF_DIR)
        print(f"Created directory: {DATA_PDF_DIR}")
        print("Please add your PDF files to this directory and run the script again.")
    else:
        asyncio.run(main())