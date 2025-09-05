import gradio as gr
import requests
import time

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
        return "❌ Không có file nào được upload", gr.Dropdown(choices=[])
    
    progress(0, desc="🔄 Đang chuẩn bị upload...")
    
    try:
        progress(0.2, desc="📤 Đang upload file...")
        with open(file.name, 'rb') as f:
            response = requests.post(f"{SERVER_URL}/add_pdf", files={"file": f})
        
        progress(0.8, desc="🔄 Đang xử lý và xây dựng Knowledge Base...")
        
        if response.status_code == 200:
            progress(1.0, desc="✅ Hoàn thành!")
            updated_files = list_files()
            return f"✅ {response.json().get('message', 'File đã được thêm thành công!')}", gr.Dropdown(choices=updated_files)
        else:
            return f"❌ Lỗi: {response.json().get('detail', 'Unknown error')}", gr.Dropdown(choices=[])
            
    except Exception as e:
        return f"❌ Lỗi khi thêm file: {str(e)}", gr.Dropdown(choices=[])

def list_files():
    try:
        response = requests.get(f"{SERVER_URL}/list_files")
        if response.status_code == 200:
            files = response.json().get("files", [])
            return files
        else:
            return []
    except Exception as e:
        print(f"Error listing files: {str(e)}")
        return []

def refresh_file_list():
    files = list_files()
    return gr.Dropdown(choices=files, value=None)

def delete_file(selected, progress=gr.Progress()):
    if not selected:
        return "❌ Chưa chọn file nào để xóa", gr.Dropdown(choices=[])
    
    progress(0, desc="🗑️ Đang xóa file...")
    
    try:
        progress(0.5, desc="🔄 Đang xóa và cập nhật Knowledge Base...")
        response = requests.delete(f"{SERVER_URL}/delete_file/{selected}")
        
        if response.status_code == 200:
            progress(1.0, desc="✅ Xóa thành công!")
            updated_files = list_files()
            return f"✅ {response.json().get('message', 'File đã được xóa thành công!')}", gr.Dropdown(choices=updated_files, value=None)
        else:
            return f"❌ Lỗi: {response.json().get('detail', 'Unknown error')}", gr.Dropdown(choices=[])
            
    except Exception as e:
        return f"❌ Lỗi khi xóa file: {str(e)}", gr.Dropdown(choices=[])

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
            
            # Format result with mode info
            mode_info = f"🔧 **Mode đã sử dụng:** {QUERY_MODES.get(mode_used, {}).get('name', mode_used)}"
            instruction_info = f"\n📝 **Instruction:** {instruction}" if instruction.strip() else ""
            
            formatted_result = f"{mode_info}{instruction_info}\n\n📋 **Kết quả:**\n\n{result}"
            
            return formatted_result, f"Đã truy vấn thành công với mode: {mode_used}"
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

.upload-area {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
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
    with gr.Tab("📁 Quản lý File"):
        gr.Markdown("### 📤 Thêm File PDF")
        
        with gr.Row():
            with gr.Column(scale=3):
                pdf_upload = gr.File(
                    label="Chọn file PDF", 
                    file_types=[".pdf"],
                    elem_classes=["upload-area"]
                )
            with gr.Column(scale=1):
                add_btn = gr.Button("➕ Thêm vào KB", variant="primary", size="lg")
        
        add_output = gr.Textbox(
            label="📊 Trạng thái xử lý", 
            lines=2,
            interactive=False
        )
        
        gr.Markdown("### 📋 Danh sách File trong KB")
        
        with gr.Row():
            files_dropdown = gr.Dropdown(
                label="Files hiện có", 
                choices=list_files(),
                interactive=True
            )
            with gr.Column():
                refresh_btn = gr.Button("🔄 Làm mới", variant="secondary")
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
        - "Trả lời bằng tiếng Việt"
        - "Tóm tắt thành bullet points"
        - "Giải thích như cho người mới bắt đầu"
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
        outputs=[add_output, files_dropdown],
        show_progress=True
    )
    
    refresh_btn.click(
        refresh_file_list,
        outputs=files_dropdown
    )
    
    delete_btn.click(
        delete_file, 
        inputs=[files_dropdown], 
        outputs=[delete_output, files_dropdown],
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
    
    # Auto-refresh file list on startup
    demo.load(refresh_file_list, outputs=files_dropdown)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=1234, share=False)