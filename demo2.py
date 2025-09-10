# Install required packages if not already installed
# pip install langchain langchain-google-genai langchain-milvus langchain-text-splitters langchain-community gradio pymilvus flashrank pypdf2

import os
import getpass
import json
from typing import List

import gradio as gr
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import dotenv
from langchain.chat_models import init_chat_model

dotenv.load_dotenv()

# Set up Google API key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Initialize embeddings and LLM using Gemini API - FIX: Use correct model name
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

# Milvus VectorDB setup (local file-based) - FIX: Simplified setup like langchain.ipynb
URI = "./milvus_rag.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)

# Text splitter using recursive chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Custom prompt with instructions for citing sources
custom_prompt = PromptTemplate.from_template("""
Bạn là một trợ lý cho các nhiệm vụ trả lời câu hỏi. 
Sử dụng các đoạn ngữ cảnh được cung cấp để trả lời câu hỏi. 
Trả lời câu hỏi cần có sự logic, các thông tin cần có độ chính xác nhưng cũng cần logic.
Nếu không biết câu trả lời, chỉ cần nói rằng bạn không biết, trả lời ngắn gọn tối đa ba câu và trích dẫn nguồn [phần nào của tên tệp nào].
Chỗ nguồn nó là tóm tắt ngắn gọn không quá 5 từ của nội dung đang nói tới.

Context: {context}

Question: {question}

Answer:
""")

# Simple retriever - FIX: Use similarity search like langchain.ipynb
base_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

# Format documents for context - FIX: Simplified formatting
def format_docs(docs):
    formatted_context = []
    for doc in docs:
        source = doc.metadata.get('source', 'unknown')
        content = doc.page_content.strip()
        formatted_context.append(f"{content}\n[Source: {source}]")
    return "\n\n".join(formatted_context)

# RAG chain for generation
rag_chain = (
    {"context": base_retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

# Global variable to track documents - FIX: Use simple tracking
documents_tracker = set()

# Function to add a document - FIX: Simplified approach
def add_document(file_path):
    if not file_path:
        return "No file uploaded."
    
    try:
        filename = os.path.basename(file_path)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"File not found: {filename}"
        
        # Check if document already exists
        if filename in documents_tracker:
            return f"Document '{filename}' already exists."
        
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        if not docs:
            return f"No content found in document: {filename}"
        
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            return f"No text chunks created from document: {filename}"
        
        # Add source metadata to each chunk - FIX: Include author field
        for split in splits:
            split.metadata["source"] = filename
            split.metadata["author"] = "unknown"  # Add required author field
        
        # Add documents to vector store - FIX: Use simple add_documents like langchain.ipynb
        vector_store.add_documents(splits)
        
        # Track the document
        documents_tracker.add(filename)
        
        return f"✅ Document '{filename}' added successfully with {len(splits)} chunks."
        
    except Exception as e:
        return f"❌ Error adding document '{filename}': {str(e)}"

# Function to delete a document - FIX: Use expr instead of filter
def delete_document(filename):
    if not filename:
        return "No document selected."
    
    try:
        if filename not in documents_tracker:
            return f"Document '{filename}' not found."
        
        # Delete from vector store using expression
        expr = f'source == "{filename}"'
        vector_store.delete(expr=expr)
        
        # Remove from tracker
        documents_tracker.discard(filename)
        
        return f"✅ Document '{filename}' deleted successfully from both database and tracking."
        
    except Exception as e:
        return f"❌ Error removing document '{filename}': {str(e)}"

# Function to load existing documents from Milvus - BETTER APPROACH
def load_existing_documents():
    """Load existing documents from Milvus database on startup"""
    try:
        # Use Milvus client to query unique sources
        from pymilvus import connections, Collection
        
        # Connect to the same database
        connections.connect("default", uri=URI)
        
        # Get collection (Milvus creates collection automatically)
        collection_name = "LangChainCollection"  # Default name used by langchain-milvus
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # Query to get all unique sources
            # Note: This is a workaround since Milvus doesn't have direct "distinct" query
            results = collection.query(
                expr="pk > 0",  # Get all records
                output_fields=["source"],
                limit=1000  # Adjust based on your needs
            )
            
            # Extract unique sources
            existing_sources = set()
            for result in results:
                source = result.get('source')
                if source and source != 'unknown':
                    existing_sources.add(source)
            
            # Update the tracker
            documents_tracker.update(existing_sources)
            
            print(f"Loaded {len(existing_sources)} existing documents: {existing_sources}")
            return existing_sources
            
        except Exception as collection_error:
            print(f"Collection query error: {collection_error}")
            # Fallback to similarity search method
            return load_existing_documents_fallback()
            
    except Exception as e:
        print(f"Error connecting to Milvus: {str(e)}")
        # Fallback to similarity search method
        return load_existing_documents_fallback()

def load_existing_documents_fallback():
    """Fallback method using similarity search"""
    try:
        # Use a broad search to get existing documents
        all_docs = vector_store.similarity_search(" ", k=1000)
        
        existing_sources = set()
        for doc in all_docs:
            source = doc.metadata.get('source')
            if source and source != 'unknown':
                existing_sources.add(source)
        
        documents_tracker.update(existing_sources)
        print(f"Fallback: Loaded {len(existing_sources)} existing documents")
        return existing_sources
        
    except Exception as e:
        print(f"Fallback method also failed: {str(e)}")
        return set()

# Initialize existing documents on startup
print("Loading existing documents from Milvus...")
load_existing_documents()

# Function to list documents - UPDATED to show loaded count
def list_documents():
    docs = sorted(list(documents_tracker))
    print(f"Currently tracking {len(docs)} documents: {docs}")
    return docs

# Function for querying - FIX: Simplified like langchain.ipynb
def query(question):
    if not question:
        return "No question provided.", []
    
    try:
        # Use similarity search directly like langchain.ipynb
        docs = vector_store.similarity_search(question, k=5)
        
        if not docs:
            return "No relevant documents found for your question.", []
        
        # Generate answer using the chain
        answer = rag_chain.invoke(question)
        
        # Format related documents
        related_docs = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content_snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            }
            for doc in docs
        ]
        
        return answer, related_docs
        
    except Exception as e:
        return f"Error processing query: {str(e)}", []

# Gradio interface - FIX: Simplified interface
with gr.Blocks(title="RAG System") as demo:
    gr.Markdown("# Hệ thống kho tri thức")
    
    with gr.Tab("Manage Documents"):
        gr.Markdown("### Thêm, xóa, và danh sách tài liệu")
        
        # Add refresh button
        with gr.Row():
            refresh_button = gr.Button("🔄 Làm mới danh sách tài liệu", variant="secondary")
        
        file_upload = gr.File(label="Upload PDF File")
        add_button = gr.Button("Thêm tài liệu", variant="primary")
        add_output = gr.Textbox(label="Add Status", lines=2)
        
        # Document management
        with gr.Row():
            with gr.Column(scale=1):
                doc_list = gr.Dropdown(
                    label="Chọn tài liệu để xóa", 
                    choices=[], 
                    interactive=True
                )
                delete_button = gr.Button("Xóa tài liệu", variant="stop")
                delete_output = gr.Textbox(label="Delete Status", lines=2)
            
            with gr.Column(scale=1):
                current_docs = gr.Textbox(
                    label="Danh sách tài liệu hiện tại", 
                    lines=8, 
                    interactive=False,
                    placeholder="No documents found"
                )
        
        # Helper functions
        def update_doc_list():
            docs = list_documents()
            docs_display = "\n".join([f"📄 {doc}" for doc in docs]) if docs else "Không tìm thấy tài liệu"
            return gr.update(choices=docs), docs_display
        
        def add_document_and_update(file):
            if file is None:
                return "Không có tài liệu được chọn.", gr.update(), ""
            
            result = add_document(file.name)
            updated_choices, docs_display = update_doc_list()
            return result, updated_choices, docs_display
        
        def delete_document_and_update(selected):
            if not selected:
                return "Không có tài liệu được chọn.", gr.update(), ""
            
            result = delete_document(selected)
            updated_choices, docs_display = update_doc_list()
            return result, updated_choices, docs_display
        
        # Event handlers
        add_button.click(
            fn=add_document_and_update,
            inputs=file_upload,
            outputs=[add_output, doc_list, current_docs]
        )
        
        delete_button.click(
            fn=delete_document_and_update,
            inputs=doc_list,
            outputs=[delete_output, doc_list, current_docs]
        )
        
        refresh_button.click(
            fn=update_doc_list,
            outputs=[doc_list, current_docs]
        )
        
        # Initialize on load
        demo.load(
            fn=update_doc_list, 
            outputs=[doc_list, current_docs]
        )
    
    with gr.Tab("Q&A"):
        gr.Markdown("### Hỏi câu hỏi về tài liệu của bạn")
        
        question_input = gr.Textbox(
            label="Enter your question", 
            placeholder="Nhập câu hỏi...",
            lines=2
        )
        ask_button = gr.Button("Ask", variant="primary")
        
        answer_output = gr.Textbox(label="Answer", lines=5)
        docs_output = gr.JSON(label="Related Documents (with Sources)")
        
        ask_button.click(
            fn=query,
            inputs=question_input,
            outputs=[answer_output, docs_output]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=9999)