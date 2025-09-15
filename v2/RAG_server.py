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
        self.base_retriever = self.vector_db_controller.vector_db.as_retriever(search_kwargs={"k": 20})
        # self.app.include_router(self.vector_db_controller.router)
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
            docs_with_scores = self.vector_db_controller.vector_db.similarity_search_with_score(query, k=top_k)
            docs = [doc for doc, score in docs_with_scores if score >= 0.5]

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
        return "Vui lòng chọn file.", gr.update()
    try:
        result = rag_server.vector_db_controller.add_documents(file.name)
        # Refresh danh sách documents
        updated_choices = rag_server.list_documents()
        return result, gr.update(choices=updated_choices)
    except Exception as e:
        return f"Lỗi: {str(e)}", gr.update()

def gradio_delete_document(filename):
    if not filename:
        return "Vui lòng chọn file để xóa.", gr.update()
    try:
        result = rag_server.vector_db_controller.delete_documents(filename)
        # Refresh danh sách documents
        updated_choices = rag_server.list_documents()
        return result, gr.update(choices=updated_choices)
    except Exception as e:
        return f"Lỗi: {str(e)}", gr.update()

def gradio_list_documents():
    docs = rag_server.list_documents()
    return "\n".join(docs) if docs else "Chưa có tài liệu nào."

def refresh_document_list():
    rag_server.vector_db_controller.load_existing_documents()  # Đọc lại từ DB
    return gr.update(choices=rag_server.list_documents())

with gr.Blocks() as demo:
    gr.Markdown("# RAG System Server")
    
    with gr.Tab("Query"):
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(label="Question", placeholder="Nhập câu hỏi của bạn...")
                top_k = gr.Slider(label="Top K", minimum=1, maximum=20, value=10)
                submit = gr.Button("Submit", variant="primary")
            with gr.Column():
                answer = gr.Textbox(label="Answer", placeholder="Câu trả lời sẽ hiển thị ở đây", lines=10)
        submit.click(fn=gradio_query, inputs=[query, top_k], outputs=answer)

    with gr.Tab("Quản lý Tài liệu"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Thêm Tài liệu")
                file_input = gr.File(label="Chọn file để upload (PDF, TXT)")
                add_btn = gr.Button("Thêm Tài liệu", variant="primary")
                add_result = gr.Textbox(label="Kết quả", lines=3)
                
            with gr.Column():
                gr.Markdown("### Xóa Tài liệu")
                with gr.Row():
                    file_dropdown = gr.Dropdown(
                        label="Chọn file để xóa", 
                        choices=rag_server.list_documents(),
                        interactive=True
                    )
                    refresh_btn = gr.Button("🔄", size="sm")
                del_btn = gr.Button("Xóa Tài liệu", variant="stop")
                del_result = gr.Textbox(label="Kết quả", lines=3)
                
                gr.Markdown("### Danh sách Tài liệu Hiện tại")
                list_result = gr.Textbox(
                    label="Tài liệu đã có", 
                    value=gradio_list_documents(),
                    lines=8,
                    interactive=False
                )
        
        # Event handlers với auto-refresh
        add_btn.click(
            fn=gradio_add_document, 
            inputs=file_input, 
            outputs=[add_result, file_dropdown]
        ).then(
            fn=gradio_list_documents,
            outputs=list_result
        )
        
        del_btn.click(
            fn=gradio_delete_document, 
            inputs=file_dropdown, 
            outputs=[del_result, file_dropdown]
        ).then(
            fn=gradio_list_documents,
            outputs=list_result
        )
        
        refresh_btn.click(
            fn=refresh_document_list,
            outputs=file_dropdown
        ).then(
            fn=gradio_list_documents,
            outputs=list_result
        )

if __name__ == "__main__":
    demo.launch(share=False)