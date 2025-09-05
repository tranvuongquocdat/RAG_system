import gradio as gr
import requests
import time
import json
from datetime import datetime

# Server URL
SERVER_URL = "http://localhost:8000"

# Query modes with descriptions
QUERY_MODES = {
    "hybrid": {
        "name": "🔀 Hybrid (Khuyến nghị)",
        "description": "Kết hợp cả tìm kiếm cục bộ và toàn cục, cho kết quả tốt nhất"
    },
    "mix": {
        "name": "🎯 Mix",
        "description": "Tích hợp knowledge graph và vector retrieval"
    },
    "local": {
        "name": "📍 Local",
        "description": "Tập trung vào thông tin phụ thuộc ngữ cảnh cụ thể"
    },
    "global": {
        "name": "🌐 Global", 
        "description": "Sử dụng kiến thức toàn cục, phù hợp cho câu hỏi tổng quan"
    },
    "naive": {
        "name": "⚡ Naive",
        "description": "Tìm kiếm cơ bản, nhanh nhưng kém chính xác"
    }
}

# Client functions
def add_pdf(file, progress=gr.Progress()):
    if file is None:
        return "❌ Không có file nào được upload", get_kb_status_display()
    
    progress(0, desc="🔄 Đang chuẩn bị upload...")
    
    try:
        progress(0.3, desc="📤 Đang upload file...")
        with open(file.name, 'rb') as f:
            response = requests.post(f"{SERVER_URL}/add_pdf", files={"file": f})
        
        progress(0.8, desc="🔄 Đang xử lý và thêm vào Knowledge Base...")
        
        if response.status_code == 200:
            progress(1.0, desc="✅ Hoàn thành!")
            return f"✅ {response.json().get('message', 'File đã được thêm thành công!')}", get_kb_status_display()
        else:
            return f"❌ Lỗi: {response.json().get('detail', 'Unknown error')}", get_kb_status_display()
            
    except Exception as e:
        return f"❌ Lỗi khi thêm file: {str(e)}", get_kb_status_display()

def get_kb_status_display():
    """Get comprehensive KB status for display"""
    try:
        # Get files list
        files_response = requests.get(f"{SERVER_URL}/list_files")
        status_response = requests.get(f"{SERVER_URL}/kb_status")
        
        if files_response.status_code == 200 and status_response.status_code == 200:
            files_data = files_response.json()
            status_data = status_response.json()
            
            kb_files = files_data.get("kb_files", [])
            summary = files_data.get("summary", {})
            
            # Format display
            if not kb_files:
                return "📋 **Knowledge Base hiện đang trống**\n\nChưa có tài liệu nào được thêm vào."
            
            display = f"📊 **Trạng thái Knowledge Base**\n\n"
            display += f"- 📄 Tổng số tài liệu: **{summary.get('total_documents', 0)}**\n"
            display += f"- 🧩 Tổng số chunks: **{summary.get('total_chunks', 0)}**\n"
            display += f"- 📈 Trạng thái: **{status_data.get('status', 'unknown').upper()}**\n\n"
            
            display += "📑 **Chi tiết tài liệu:**\n\n"
            
            for i, file_info in enumerate(kb_files, 1):
                filename = file_info.get('filename', 'Unknown')
                chunks = file_info.get('chunks_count', 0)
                size = file_info.get('content_length', 0)
                created = file_info.get('created_at', '')
                status = file_info.get('status', 'unknown')
                
                # Format created date
                try:
                    if created:
                        created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        created_str = created_dt.strftime("%d/%m/%Y %H:%M")
                    else:
                        created_str = "N/A"
                except:
                    created_str = "N/A"
                
                status_emoji = "✅" if status == "processed" else "⏳"
                
                display += f"{i}. **{filename}**\n"
                display += f"   - {status_emoji} Trạng thái: {status}\n"
                display += f"   - 🧩 Chunks: {chunks}\n"
                display += f"   - 📏 Kích thước: {size:,} ký tự\n"
                display += f"   - 📅 Thêm vào: {created_str}\n\n"
            
            return display
        else:
            return "❌ Không thể tải trạng thái Knowledge Base"
            
    except Exception as e:
        return f"❌ Lỗi khi tải trạng thái: {str(e)}"

def refresh_kb_status():
    return get_kb_status_display()

def get_simple_file_list():
    """Get simple list of filenames for dropdown"""
    try:
        response = requests.get(f"{SERVER_URL}/list_files")
        if response.status_code == 200:
            kb_files = response.json().get("kb_files", [])
            return [f.get('filename', '') for f in kb_files if f.get('filename')]
        return []
    except:
        return []

def refresh_file_dropdown():
    files = get_simple_file_list()
    return gr.Dropdown(choices=files, value=None)

def delete_file(selected, progress=gr.Progress()):
    if not selected:
        return "❌ Chưa chọn file nào để xóa", get_kb_status_display()
    
    progress(0, desc="🗑️ Đang xóa file...")
    
    try:
        progress(0.5, desc="🔄 Đang xóa và cập nhật Knowledge Base...")
        response = requests.delete(f"{SERVER_URL}/delete_file/{selected}")
        
        if response.status_code == 200:
            progress(1.0, desc="✅ Xóa thành công!")
            return f"✅ {response.json().get('message', 'File đã được xóa thành công!')}", get_kb_status_display()
        else:
            return f"❌ Lỗi: {response.json().get('detail', 'Unknown error')}", get_kb_status_display()
            
    except Exception as e:
        return f"❌ Lỗi khi xóa file: {str(e)}", get_kb_status_display()

def get_mode_info(mode):
    """Get detailed information about selected mode"""
    if mode in QUERY_MODES:
        return f"**{QUERY_MODES[mode]['name']}**\n\n{QUERY_MODES[mode]['description']}"
    return ""

def query_kb(q, mode, instruction, progress=gr.Progress()):
    if not q.strip():
        return "❌ Vui lòng nhập câu hỏi", ""
    
    progress(0, desc="🔍 Đang tìm kiếm...")
    
    try:
        progress(0.5, desc="🤖 Đang xử lý câu trả lời...")
        
        payload = {"query": q, "mode": mode}
        if instruction.strip():
            payload["instruction"] = instruction.strip()
            
        response = requests.post(f"{SERVER_URL}/query", json=payload)
        
        if response.status_code == 200:
            progress(1.0, desc="✅ Hoàn thành!")
            data = response.json()
            result = data.get("result", "Không có kết quả")
            mode_used = data.get("mode_used", mode)
            sources = data.get("sources", [])
            query_time = data.get("query_time", "")
            
            # Format result with metadata
            mode_info = f"🔧 **Mode:** {QUERY_MODES.get(mode_used, {}).get('name', mode_used)}"
            
            sources_info = ""
            if sources:
                sources_info = f"\n📚 **Nguồn tham khảo:** {', '.join(sources)}"
            
            instruction_info = f"\n📝 **Instruction:** {instruction}" if instruction.strip() else ""
            
            time_info = ""
            if query_time:
                try:
                    query_dt = datetime.fromisoformat(query_time)
                    time_info = f"\n⏰ **Thời gian truy vấn:** {query_dt.strftime('%H:%M:%S %d/%m/%Y')}"
                except:
                    pass
            
            formatted_result = f"{mode_info}{sources_info}{instruction_info}{time_info}\n\n📋 **Kết quả:**\n\n{result}"
            
            return formatted_result, f"Truy vấn thành công với mode: {mode_used}"
        else:
            return f"❌ Lỗi: {response.json().get('detail', 'Unknown error')}", "Có lỗi xảy ra"
            
    except Exception as e:
        return f"❌ Lỗi khi truy vấn: {str(e)}", "Có lỗi xảy ra"

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.gr-button {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    border: none;
    color: white;
    font-weight: bold;
    transition: all 0.3s ease;
}

.gr-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.kb-status {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid #0ea5e9;
    font-family: 'Monaco', 'Consolas', monospace;
    line-height: 1.6;
}

.mode-info {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #2196f3;
    margin: 10px 0;
}

.instruction-box {
    background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #ff9800;
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css, title="🤖 LightRAG Knowledge Base") as demo:
    gr.Markdown("""
    # 🤖 LightRAG Knowledge Base với Gemini AI
    
    **Hướng dẫn sử dụng:**
    - 📤 Upload file PDF để thêm vào Knowledge Base
    - 🗑️ Xóa file không cần thiết  
    - 🔍 Đặt câu hỏi với các mode khác nhau để tìm kiếm thông tin
    - 📝 Thêm instruction để hướng dẫn AI trả lời theo cách bạn muốn
    """)
    
    # File Management Section
    with gr.Tab("📁 Quản lý Knowledge Base"):
        gr.Markdown("### 📤 Thêm File PDF")
        
        with gr.Row():
            with gr.Column(scale=3):
                pdf_upload = gr.File(
                    label="Chọn file PDF", 
                    file_types=[".pdf"]
                )
            with gr.Column(scale=1):
                add_btn = gr.Button("➕ Thêm vào KB", variant="primary", size="lg")
        
        add_output = gr.Textbox(
            label="📊 Trạng thái xử lý", 
            lines=2,
            interactive=False
        )
        
        gr.Markdown("### 📋 Trạng thái Knowledge Base")
        
        with gr.Row():
            with gr.Column(scale=3):
                kb_status_display = gr.Markdown(
                    get_kb_status_display(),
                    elem_classes=["kb-status"]
                )
            with gr.Column(scale=1):
                refresh_status_btn = gr.Button("🔄 Làm mới trạng thái", variant="secondary")
        
        gr.Markdown("### 🗑️ Xóa File")
        
        with gr.Row():
            files_dropdown = gr.Dropdown(
                label="Chọn file cần xóa", 
                choices=get_simple_file_list(),
                interactive=True
            )
            with gr.Column():
                refresh_dropdown_btn = gr.Button("🔄 Làm mới danh sách", variant="secondary")
                delete_btn = gr.Button("🗑️ Xóa file đã chọn", variant="stop")
        
        delete_output = gr.Textbox(
            label="📊 Trạng thái xóa", 
            lines=2,
            interactive=False
        )
    
    # Query Section
    with gr.Tab("🔍 Tìm kiếm Thông minh"):
        gr.Markdown("### 💬 Đặt câu hỏi cho Knowledge Base")
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Câu hỏi của bạn", 
                    placeholder="Ví dụ: Tóm tắt nội dung chính của tài liệu...",
                    lines=3
                )
                
                instruction_input = gr.Textbox(
                    label="📝 Instruction (Tùy chọn)",
                    placeholder="Ví dụ: Trả lời bằng tiếng Việt, tóm tắt thành 3 điểm chính...",
                    lines=2,
                    elem_classes=["instruction-box"]
                )
                
            with gr.Column(scale=1):
                mode_dropdown = gr.Dropdown(
                    label="🔧 Chọn Mode Tìm kiếm",
                    choices=[(info["name"], key) for key, info in QUERY_MODES.items()],
                    value="hybrid",
                    interactive=True
                )
                
                mode_info_display = gr.Markdown(
                    get_mode_info("hybrid"),
                    elem_classes=["mode-info"]
                )
        
        gr.Markdown("""
        ### 📚 **Giải thích các Mode:**
        
        - **🔀 Hybrid**: Tốt nhất cho hầu hết câu hỏi, kết hợp nhiều phương pháp tìm kiếm
        - **🎯 Mix**: Cân bằng giữa knowledge graph và vector search
        - **📍 Local**: Tốt cho câu hỏi cụ thể về một chủ đề nhất định
        - **🌐 Global**: Phù hợp cho câu hỏi tổng quan, xu hướng chung
        - **⚡ Naive**: Tìm kiếm nhanh nhưng đơn giản, ít chính xác
        
        ### 💡 **Mẹo sử dụng Instruction:**
        - "Trả lời bằng tiếng Việt với bullet points"
        - "Tóm tắt thành 3 điểm chính kèm nguồn"
        - "Giải thích chi tiết với ví dụ cụ thể"
        - "So sánh và đối chiếu các quan điểm"
        """)
        
        query_btn = gr.Button("🚀 Tìm kiếm", variant="primary", size="lg")
        
        query_status = gr.Textbox(
            label="📊 Trạng thái",
            lines=1,
            interactive=False
        )
        
        query_output = gr.Textbox(
            label="📋 Kết quả", 
            lines=20,
            interactive=False,
            show_copy_button=True
        )
    
    # Event handlers
    add_btn.click(
        add_pdf, 
        inputs=[pdf_upload], 
        outputs=[add_output, kb_status_display],
        show_progress=True
    )
    
    refresh_status_btn.click(
        refresh_kb_status,
        outputs=[kb_status_display]
    )
    
    refresh_dropdown_btn.click(
        refresh_file_dropdown,
        outputs=[files_dropdown]
    )
    
    delete_btn.click(
        delete_file, 
        inputs=[files_dropdown], 
        outputs=[delete_output, kb_status_display],
        show_progress=True
    )
    
    # Update mode info when mode changes
    mode_dropdown.change(
        get_mode_info,
        inputs=[mode_dropdown],
        outputs=[mode_info_display]
    )
    
    query_btn.click(
        query_kb, 
        inputs=[query_input, mode_dropdown, instruction_input], 
        outputs=[query_output, query_status],
        show_progress=True
    )
    
    # Auto-refresh status on startup
    demo.load(refresh_kb_status, outputs=[kb_status_display])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=1234, share=False)