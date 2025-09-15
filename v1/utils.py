"""
Utility functions for RAG system
Các hàm hỗ trợ cho hệ thống RAG (logging, data cleaning, file handling)
"""

import os
import re
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import unicodedata
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Thiết lập logging cho hệ thống
    
    Args:
        log_level: Mức độ log (DEBUG, INFO, WARNING, ERROR)
        log_file: Đường dẫn file log (optional)
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger("rag_system")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Tránh duplicate handlers
    if logger.handlers:
        return logger
    
    # Format cho log messages
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (nếu có)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def clean_text(text: str) -> str:
    """
    Làm sạch text để chuẩn bị cho embedding
    
    Args:
        text: Raw text cần làm sạch
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Loại bỏ các ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\s\.,!?;:()\-\'""]', ' ', text)
    
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    
    # Loại bỏ dòng trống
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    Trích xuất metadata từ tên file
    
    Args:
        filename: Tên file
    
    Returns:
        Dictionary chứa metadata
    """
    path = Path(filename)
    
    metadata = {
        'filename': path.name,
        'extension': path.suffix.lower(),
        'basename': path.stem,
        'size': 0,
        'created_date': None,
        'modified_date': None
    }
    
    # Lấy thông tin file nếu tồn tại
    if path.exists():
        stat = path.stat()
        metadata['size'] = stat.st_size
        metadata['created_date'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
        metadata['modified_date'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
    
    # Phân tích tên file để tìm thông tin bổ sung
    basename_lower = path.stem.lower()
    
    # Detect document type
    if any(keyword in basename_lower for keyword in ['report', 'báo cáo']):
        metadata['document_type'] = 'report'
    elif any(keyword in basename_lower for keyword in ['manual', 'hướng dẫn', 'guide']):
        metadata['document_type'] = 'manual'
    elif any(keyword in basename_lower for keyword in ['policy', 'chính sách']):
        metadata['document_type'] = 'policy'
    elif any(keyword in basename_lower for keyword in ['contract', 'hợp đồng']):
        metadata['document_type'] = 'contract'
    else:
        metadata['document_type'] = 'general'
    
    # Extract date from filename (YYYY-MM-DD or DD-MM-YYYY patterns)
    date_patterns = [
        r'(\d{4}[-_]\d{2}[-_]\d{2})',  # YYYY-MM-DD
        r'(\d{2}[-_]\d{2}[-_]\d{4})',  # DD-MM-YYYY
        r'(\d{4}[-_]\d{2})',          # YYYY-MM
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, basename_lower)
        if match:
            metadata['date_in_filename'] = match.group(1)
            break
    
    return metadata

def generate_document_id(content: str, metadata: Dict[str, Any]) -> str:
    """
    Tạo ID duy nhất cho document dựa trên content và metadata
    
    Args:
        content: Nội dung document
        metadata: Metadata của document
    
    Returns:
        Document ID (hash string)
    """
    # Kết hợp content và các metadata quan trọng
    id_string = f"{content[:500]}{metadata.get('filename', '')}{metadata.get('size', 0)}"
    
    # Tạo SHA-256 hash
    return hashlib.sha256(id_string.encode('utf-8')).hexdigest()[:16]

def chunk_text_with_overlap(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 200,
    separators: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Chia text thành các chunks với overlap
    
    Args:
        text: Text cần chia
        chunk_size: Kích thước mỗi chunk
        overlap: Số ký tự overlap giữa các chunks
        separators: Danh sách separators để ưu tiên chia (optional)
    
    Returns:
        List of chunks với metadata
    """
    if not text:
        return []
    
    if separators is None:
        separators = ['\n\n', '\n', '. ', ' ', '']
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        # Xác định end position
        end = start + chunk_size
        
        if end >= len(text):
            # Chunk cuối cùng
            chunk_text = text[start:].strip()
            if chunk_text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': len(text),
                    'length': len(chunk_text)
                })
            break
        
        # Tìm separator tốt nhất để cắt
        best_split = end
        for separator in separators:
            if separator == '':
                continue
            
            # Tìm separator gần nhất với end position
            sep_pos = text.rfind(separator, start, end)
            if sep_pos != -1:
                best_split = sep_pos + len(separator)
                break
        
        chunk_text = text[start:best_split].strip()
        if chunk_text:
            chunks.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_pos': start,
                'end_pos': best_split,
                'length': len(chunk_text)
            })
            chunk_id += 1
        
        # Di chuyển start position với overlap
        start = best_split - overlap
        if start < 0:
            start = best_split
    
    return chunks

def validate_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Kiểm tra loại file có được phép không
    
    Args:
        filename: Tên file
        allowed_extensions: Danh sách extension được phép (bao gồm dấu chấm)
    
    Returns:
        True nếu file type được phép
    """
    file_ext = Path(filename).suffix.lower()
    return file_ext in [ext.lower() for ext in allowed_extensions]

def safe_filename(filename: str) -> str:
    """
    Tạo filename an toàn (loại bỏ ký tự đặc biệt)
    
    Args:
        filename: Tên file gốc
    
    Returns:
        Safe filename
    """
    # Loại bỏ ký tự đặc biệt
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Loại bỏ khoảng trắng đầu/cuối
    safe_name = safe_name.strip()
    
    # Đảm bảo không rỗng
    if not safe_name:
        safe_name = "unnamed_file"
    
    return safe_name

def format_file_size(size_bytes: int) -> str:
    """
    Format file size thành dạng human-readable
    
    Args:
        size_bytes: Kích thước file tính bằng bytes
    
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"

def create_directory_if_not_exists(directory: Union[str, Path]) -> None:
    """
    Tạo thư mục nếu chưa tồn tại
    
    Args:
        directory: Đường dẫn thư mục
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_supported_file_types() -> Dict[str, List[str]]:
    """
    Lấy danh sách các loại file được hỗ trợ
    
    Returns:
        Dictionary mapping category to file extensions
    """
    return {
        'documents': ['.pdf', '.docx', '.doc', '.txt', '.md', '.rtf'],
        'spreadsheets': ['.xlsx', '.xls', '.csv'],
        'presentations': ['.pptx', '.ppt'],
        'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
        'code': ['.py', '.js', '.html', '.css', '.json', '.xml', '.yaml', '.yml']
    }

def is_text_meaningful(text: str, min_words: int = 3) -> bool:
    """
    Kiểm tra text có ý nghĩa không (không phải chỉ ký tự đặc biệt)
    
    Args:
        text: Text cần kiểm tra
        min_words: Số từ tối thiểu
    
    Returns:
        True nếu text có ý nghĩa
    """
    if not text or not text.strip():
        return False
    
    # Đếm số từ (chỉ tính các từ có ít nhất 2 ký tự)
    words = re.findall(r'\b\w{2,}\b', text)
    return len(words) >= min_words
