import gradio as gr
import requests
import json
from typing import List, Optional
import time

# Server configuration
SERVER_URL = "http://localhost:8000"

# Thêm biến toàn cục để lưu API key
api_key = ""

def set_api_key(user_api_key):
    global api_key
    api_key = user_api_key
    return f"Đã lưu API key: {user_api_key[:4]}***"

def upload_documents(files):
    """Upload multiple documents to server"""
    if not files:
        return "Vui lòng chọn ít nhất một tài liệu!", None
    
    results = []
    progress_messages = []
    
    for i, file in enumerate(files):
        try:
            progress_messages.append(f"Đang xử lý tài liệu {i+1}/{len(files)}: {file.name}")
            yield "\n".join(progress_messages), None
            
            with open(file.name, "rb") as f:
                files_data = {"file": (file.name.split("/")[-1], f, "application/octet-stream")}
                response = requests.post(
                    f"{SERVER_URL}/documents/",
                    files=files_data,
                    headers={"X-GEMINI-API-KEY": api_key} if api_key else None
                )
            
            if response.status_code == 200:
                result = response.json()
                progress_messages.append(f"✓ {file.name.split('/')[-1]}: Đã thêm {result['chunks_count']} chunks")
                results.append(f"✓ {file.name.split('/')[-1]}: Success")
            elif response.status_code == 409:
                progress_messages.append(f"⚠ {file.name.split('/')[-1]}: Tài liệu đã tồn tại")
                results.append(f"⚠ {file.name.split('/')[-1]}: Already exists")
            else:
                error_msg = response.json().get("detail", "Unknown error")
                progress_messages.append(f"✗ {file.name.split('/')[-1]}: Lỗi - {error_msg}")
                results.append(f"✗ {file.name.split('/')[-1]}: Error")
                
        except Exception as e:
            progress_messages.append(f"✗ {file.name.split('/')[-1]}: Exception - {str(e)}")
            results.append(f"✗ {file.name.split('/')[-1]}: Exception")
        
        yield "\n".join(progress_messages), None
    
    # Final result
    final_message = "\n".join(progress_messages) + "\n\n=== KẾT QUA CUỐI CÙNG ==="
    updated_docs = get_documents_list()
    yield final_message, updated_docs

def get_documents_list():
    """Get list of documents from server"""
    try:
        response = requests.get(f"{SERVER_URL}/documents/")
        if response.status_code == 200:
            documents = response.json()
            if not documents:
                return [["Không có tài liệu nào", "", "", ""]]
            
            return [[doc["name"], doc["id"][:8], doc["created_at"][:19], str(doc["chunks_count"])] 
                   for doc in documents]
        else:
            return [["Lỗi khi tải danh sách tài liệu", "", "", ""]]
    except Exception as e:
        return [[f"Lỗi: {str(e)}", "", "", ""]]

def delete_documents(selected_docs):
    """Delete selected documents"""
    if selected_docs is None or len(selected_docs) == 0:
        return "Vui lòng chọn tài liệu để xóa!", None

    results = []
    for doc_item in selected_docs:  # Renamed to avoid shadowing
        doc_name = "Unknown"  # Initialize to avoid UnboundLocalError
        try:
            # Handle both list and tuple formats
            if isinstance(doc_item, (list, tuple)) and len(doc_item) >= 2:
                doc_name, doc_id_short = doc_item[0], doc_item[1]
            else:
                results.append(f"✗ Invalid document format: {doc_item}")
                continue
            
            # Get full document ID
            response = requests.get(f"{SERVER_URL}/documents/")
            if response.status_code == 200:
                documents = response.json()
                full_doc_id = None
                for document in documents:  # Renamed to avoid shadowing
                    if document["id"].startswith(doc_id_short) and document["name"] == doc_name:
                        full_doc_id = document["id"]
                        break
                
                if full_doc_id:
                    delete_response = requests.delete(f"{SERVER_URL}/documents/{full_doc_id}")
                    if delete_response.status_code == 200:
                        results.append(f"✓ Đã xóa: {doc_name}")
                    else:
                        results.append(f"✗ Lỗi khi xóa: {doc_name}")
                else:
                    results.append(f"✗ Không tìm thấy: {doc_name}")
            else:
                results.append(f"✗ Lỗi kết nối server khi xóa: {doc_name}")
            
        except Exception as e:
            results.append(f"✗ Exception: {doc_name} - {str(e)}")
    
    result_message = "\n".join(results)
    updated_docs = get_documents_list()
    # Ensure all rows are lists, not tuples
    updated_docs = [list(row) for row in updated_docs]
    return result_message, updated_docs

def query_knowledge_base(query, instruction):
    """Query the knowledge base"""
    if not query.strip():
        return "Vui lòng nhập câu hỏi!", "", ""
    
    global api_key
    try:
        payload = {
            "query": query,
            "instruction": instruction if instruction.strip() else None,
            "top_k": 5
        }
        
        response = requests.post(
            f"{SERVER_URL}/query/",
            json=payload,
            headers={"X-GEMINI-API-KEY": api_key} if api_key else None
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Format relevant chunks
            chunks_text = ""
            if result["relevant_chunks"]:
                for i, (chunk, source) in enumerate(zip(result["relevant_chunks"], result["sources"])):
                    chunks_text += f"**Chunk {i+1} (từ {source}):**\n{chunk}\n\n"
            
            # Format sources
            sources_text = ""
            if result["sources"]:
                unique_sources = list(set(result["sources"]))
                sources_text = "**Nguồn tài liệu:** " + ", ".join(unique_sources)
            
            return result["answer"], chunks_text, sources_text
        
        else:
            error_detail = response.json().get("detail", "Unknown error")
            return f"Lỗi: {error_detail}", "", ""
            
    except Exception as e:
        return f"Lỗi kết nối: {str(e)}", "", ""

def refresh_documents_list():
    """Refresh the documents list"""
    docs = get_documents_list()
    # Ensure dropdown choices are tuples for proper handling
    dropdown_choices = [(doc[0], doc[1]) for doc in docs if len(doc) >= 2]
    return docs, dropdown_choices

def check_server_status():
    """Check if server is running"""
    try:
        response = requests.get(f"{SERVER_URL}/")
        if response.status_code == 200:
            data = response.json()
            return f"✓ Server đang hoạt động\n📄 Số tài liệu: {data['documents_count']}\n📝 Tổng chunks: {data['total_chunks']}"
        else:
            return "✗ Server có vấn đề"
    except Exception as e:
        return f"✗ Không thể kết nối server: {str(e)}"

def get_documents_table():
    """Get documents list for table only"""
    return get_documents_list()

# Create Gradio interface
with gr.Blocks(title="Hệ thống kho tri thức", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Hệ thống kho tri thức")
    gr.Markdown("Hệ thống sử dụng LLM để trích xuất tri thức từ tài liệu và trả lời câu hỏi dựa trên tài liệu đó")
    # Server status
    with gr.Row():
        status_btn = gr.Button("🔍 Kiểm tra Server", variant="secondary")
        status_output = gr.Textbox(label="Trạng thái Server", interactive=False)
    
    status_btn.click(check_server_status, outputs=status_output)
    
    # Document management section
    with gr.Tab("📁 Quản lý Tài liệu"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Thêm Tài liệu")
                file_input = gr.File(
                    label="Chọn tài liệu (PDF, DOCX, TXT)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt"]
                )
                upload_btn = gr.Button("📤 Thêm vào Knowledge Base", variant="primary")
                upload_progress = gr.Textbox(label="Tiến trình xử lý", lines=8, interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### Danh sách Tài liệu")
                refresh_btn = gr.Button("🔄 Làm mới danh sách")
                docs_table = gr.Dataframe(
                    headers=["Tên tài liệu", "ID", "Ngày tạo", "Số chunks"],
                    datatype=["str", "str", "str", "str"],
                    value=get_documents_list(),
                    interactive=True
                )
                
                delete_dropdown = gr.Dropdown(
                    label="Chọn tài liệu để xóa",
                    choices=[(doc[0], doc[1]) for doc in get_documents_list() if len(doc) >= 2],  # Tuples (Tên, ID rút gọn)
                    multiselect=True,
                    allow_custom_value=True  # Add this to prevent warnings
                )
                delete_btn = gr.Button("🗑️ Xóa tài liệu đã chọn", variant="stop")
                delete_result = gr.Textbox(label="Kết quả xóa", interactive=False)
    
    # Query section
    with gr.Tab("❓ Truy vấn"):
        with gr.Column():
            query_input = gr.Textbox(
                label="Câu hỏi của bạn",
                placeholder="Nhập câu hỏi...",
                lines=2
            )
            instruction_input = gr.Textbox(
                label="Instruction (tùy chọn)",
                placeholder="Tùy chỉnh cách AI trả lời...",
                lines=3
            )
            query_btn = gr.Button("🔍 Tìm kiếm", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    answer_output = gr.Textbox(
                        label="📝 Câu trả lời",
                        lines=8,
                        interactive=False
                    )
                with gr.Column(scale=1):
                    sources_output = gr.Textbox(
                        label="📚 Nguồn tài liệu",
                        lines=3,
                        interactive=False
                    )
            
            chunks_output = gr.Textbox(
                label="📄 Các đoạn văn liên quan",
                lines=10,
                interactive=False
            )
    
    # Event handlers
    upload_btn.click(
        upload_documents,
        inputs=file_input,
        outputs=[upload_progress, docs_table]
    )
    
    refresh_btn.click(
        refresh_documents_list,
        outputs=[docs_table, delete_dropdown]
    )
    
    delete_btn.click(
        delete_documents,
        inputs=delete_dropdown,
        outputs=[delete_result, docs_table]
    )
    
    query_btn.click(
        query_knowledge_base,
        inputs=[query_input, instruction_input],
        outputs=[answer_output, chunks_output, sources_output]
    )

    # with gr.Row():
    #     api_key_input = gr.Textbox(label="Gemini API Key", type="password")
    #     save_api_btn = gr.Button("Lưu API Key")
    #     api_key_status = gr.Textbox(label="Trạng thái API Key", interactive=False)
    # save_api_btn.click(set_api_key, inputs=api_key_input, outputs=api_key_status)
    
    # Auto-refresh documents list on startup
    demo.load(get_documents_table, outputs=docs_table)
    demo.load(check_server_status, outputs=status_output)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )