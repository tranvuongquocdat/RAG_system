from vectorDBController import VectorDBController
from LLMController import LLMManager
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.docs import get_redoc_html

import gradio as gr

class RAGServer:
    def __init__(self, llm_manager: LLMManager):
        self.embeddings = llm_manager.embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_db_controller = VectorDBController(embeddings=self.embeddings, text_splitter=self.text_splitter)
        self.vector_db_controller.load_vector_db("qdrant")
        self.vector_db_controller.load_existing_documents()
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.llm = llm_manager.llm
        self.base_retriever = self.vector_db_controller.vector_db.as_retriever(search_kwargs={"k": 20, "embeddings": self.embeddings})
        self.app.include_router(self.vector_db_controller.router)
        self.custom_prompt = PromptTemplate.from_template("""
        Bạn là một trợ lý cho các nhiệm vụ trả lời câu hỏi.
        Sử dụng các đoạn ngữ cảnh được cung cấp để trả lời câu hỏi.
        Trả lời câu hỏi cần có sự logic, các thông tin cần có độ chính xác cao.
        Nếu không biết câu trả lời, chỉ cần nói rằng bạn không biết hoặc không có đủ thông tin cho phần nào của câu hỏi.
        Trả lời ngắn gọn và trích dẫn nguồn [Sau từng ý trả lời, hãy trích dẫn nguồn từ các đoạn ngữ cảnh đã cung cấp].

        Context:
        {context}
        Question: {question}
        Answer:
        """)
        self.rag_chain = (
            {"context": self.base_retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.custom_prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs: List[Document]):
        """
        Format the documents
        """
        formatted_context = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            formatted_context.append(f"Source: {source}\nContent: {content}")
        return "\n\n".join(formatted_context)

    def query(self, query: str, top_k: int = 10):
        """
        Query the vector DB
        """
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        if top_k <= 0:
            raise HTTPException(status_code=400, detail="Top k must be greater than 0")
        
        try:
            docs = self.vector_db_controller.vector_db.similarity_search(query, k=top_k, threshold=0.5)

            if not docs:
                raise HTTPException(status_code=404, detail="No documents found")

            # FIX: Invoke với query thay vì {"input": query, "context": docs}
            answer = self.rag_chain.invoke(query)
            
            #Generate answer documents
            related_docs = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "content_snippet": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                }
                for doc in docs
            ]

            return answer, related_docs
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def list_documents(self):
        docs = sorted(list(self.vector_db_controller.document_tracker))
        print(f"Currently tracking {len(docs)} documents: {docs}")
        return docs

llm_manager = LLMManager()
rag_server = RAGServer(llm_manager)

#build simplified gradio interface
def gradio_query(question, top_k):
    try:
        answer, related_docs = rag_server.query(question, int(top_k))
        return answer
    except Exception as e:
        return f"Lỗi: {str(e)}"

def gradio_add_document(file):
    if file is None:
        return "Vui lòng chọn file."
    return rag_server.vector_db_controller.add_documents(file.name)

def gradio_delete_document(filename):
    return rag_server.vector_db_controller.delete_documents(filename)

def gradio_list_documents():
    docs = rag_server.list_documents()
    return "\n".join(docs)

with gr.Blocks() as demo:
    gr.Markdown("RAG System Server")
    with gr.Tab("Query"):
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="Question", placeholder="Enter your question here")
                top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=10)
                submit = gr.Button("Submit")
            with gr.Column():
                answer = gr.Textbox(label="Answer", placeholder="Answer will be displayed here")
        submit.click(fn=gradio_query, inputs=[query, top_k], outputs=answer)

    with gr.Tab("Add Document"):
        file_input = gr.File(label="Chọn file để upload")  # Thay vì Textbox
        add_btn = gr.Button("Add Document")
        add_result = gr.Textbox(label="Result")
        add_btn.click(fn=gradio_add_document, inputs=file_input, outputs=add_result)

    with gr.Tab("Delete Document"):
        file_dropdown = gr.Dropdown(label="Chọn file để xoá", choices=rag_server.list_documents())
        del_btn = gr.Button("Delete Document")
        del_result = gr.Textbox(label="Result")
        del_btn.click(fn=gradio_delete_document, inputs=file_dropdown, outputs=del_result)
        # Thêm nút làm mới danh sách file
        refresh_btn = gr.Button("Làm mới danh sách")
        refresh_btn.click(fn=lambda: gr.update(choices=rag_server.list_documents()), inputs=None, outputs=file_dropdown)

    with gr.Tab("List Documents"):
        list_btn = gr.Button("List Documents")
        list_result = gr.Textbox(label="Existing Documents")
        list_btn.click(fn=gradio_list_documents, inputs=None, outputs=list_result)

if __name__ == "__main__":
    demo.launch()
