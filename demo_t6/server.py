import os
import shutil
import io
import json
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
from pypdf import PdfReader
import google.generativeai as genai
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from typing import List, Dict, Set
import asyncio

# Thêm biến toàn cục lưu Gemini API key
GEMINI_API_KEY = ""

def configure_gemini(api_key):
    global GEMINI_API_KEY
    GEMINI_API_KEY = api_key
    genai.configure(api_key=api_key)

# Cấu hình Gemini ban đầu
configure_gemini(GEMINI_API_KEY)

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

class DocumentDeletionService:
    """
    Service for safely deleting documents from LightRAG storage
    without requiring full KB rebuild
    """
    
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.backup_dir = os.path.join(working_dir, ".deletion_backup")
        self.storage_files = [
            "kv_store_doc_status.json",
            "kv_store_full_docs.json", 
            "kv_store_text_chunks.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "vdb_chunks.json",
            "vdb_entities.json", 
            "vdb_relationships.json",
            "kv_store_llm_response_cache.json",
            "graph_chunk_entity_relation.graphml"
        ]
    
    def _create_backup(self) -> str:
        """Create timestamped backup of all storage files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.backup_dir}_{timestamp}"
        
        os.makedirs(backup_path, exist_ok=True)
        
        backed_up_files = []
        try:
            for storage_file in self.storage_files:
                source_path = os.path.join(self.working_dir, storage_file)
                if os.path.exists(source_path):
                    backup_file_path = os.path.join(backup_path, storage_file)
                    shutil.copy2(source_path, backup_file_path)
                    backed_up_files.append(storage_file)
            
            # Create manifest
            manifest = {
                "timestamp": timestamp,
                "backed_up_files": backed_up_files,
                "purpose": "document_deletion_backup"
            }
            
            with open(os.path.join(backup_path, "manifest.json"), 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"✅ Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            # Cleanup partial backup
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            raise Exception(f"Backup creation failed: {str(e)}")
    
    def _rollback_from_backup(self, backup_path: str) -> None:
        """Restore files from backup in case of failure"""
        if not os.path.exists(backup_path):
            raise Exception(f"Backup path does not exist: {backup_path}")
        
        try:
            # Read manifest
            manifest_path = os.path.join(backup_path, "manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                backed_up_files = manifest.get("backed_up_files", [])
            else:
                backed_up_files = self.storage_files
            
            # Restore files
            for storage_file in backed_up_files:
                backup_file_path = os.path.join(backup_path, storage_file)
                if os.path.exists(backup_file_path):
                    target_path = os.path.join(self.working_dir, storage_file)
                    shutil.copy2(backup_file_path, target_path)
            
            print(f"✅ Rollback completed from: {backup_path}")
            
        except Exception as e:
            raise Exception(f"Rollback failed: {str(e)}")
    
    def _cleanup_backup(self, backup_path: str) -> None:
        """Remove backup after successful operation"""
        try:
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
                print(f"✅ Cleaned up backup: {backup_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not cleanup backup {backup_path}: {str(e)}")
    
    def _find_document_by_filename(self, filename: str) -> Dict:
        """Find document ID and metadata by filename"""
        doc_status_file = os.path.join(self.working_dir, "kv_store_doc_status.json")
        
        if not os.path.exists(doc_status_file):
            return {"found": False, "error": "Document status file not found"}
        
        try:
            with open(doc_status_file, 'r', encoding='utf-8') as f:
                doc_status = json.load(f)
            
            # Normalize filename for comparison
            target_filename = filename.replace('.pdf', '.txt') if filename.endswith('.pdf') else filename
            
            for doc_id, doc_info in doc_status.items():
                content_summary = doc_info.get('content_summary', '')
                
                if '--- File:' in content_summary:
                    # Extract filename from content summary
                    start = content_summary.find('--- File:') + len('--- File:')
                    end = content_summary.find('---', start)
                    if end == -1:
                        # Try alternative patterns
                        lines = content_summary.split('\n')
                        for line in lines:
                            if '--- File:' in line:
                                filename_raw = line.replace('--- File:', '').strip()
                                break
                    else:
                        filename_raw = content_summary[start:end].strip()
                    
                    # Clean extracted filename
                    if '|' in filename_raw:
                        extracted_filename = filename_raw.split('|')[0].strip()
                    else:
                        extracted_filename = filename_raw
                    
                    # Remove page metadata
                    if 'Page:' in extracted_filename:
                        extracted_filename = extracted_filename.split('Page:')[0].strip()
                    
                    # Compare normalized filenames
                    if extracted_filename == target_filename or extracted_filename == filename:
                        return {
                            "found": True,
                            "doc_id": doc_id,
                            "doc_info": doc_info,
                            "chunks_list": doc_info.get('chunks_list', []),
                            "llm_cache_entries": self._extract_llm_cache_from_chunks(doc_info.get('chunks_list', []))
                        }
            
            return {"found": False, "error": f"Document not found: {filename}"}
            
        except Exception as e:
            return {"found": False, "error": f"Error searching document: {str(e)}"}
    
    def _extract_llm_cache_from_chunks(self, chunk_ids: List[str]) -> Set[str]:
        """Extract LLM cache entries from chunks"""
        cache_entries = set()
        
        text_chunks_file = os.path.join(self.working_dir, "kv_store_text_chunks.json")
        if not os.path.exists(text_chunks_file):
            return cache_entries
        
        try:
            with open(text_chunks_file, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
            
            for chunk_id in chunk_ids:
                if chunk_id in text_chunks:
                    chunk_cache_list = text_chunks[chunk_id].get('llm_cache_list', [])
                    cache_entries.update(chunk_cache_list)
            
        except Exception as e:
            print(f"⚠️ Warning: Could not extract LLM cache entries: {str(e)}")
        
        return cache_entries
    
    def _remove_from_json_storage(self, filename: str, keys_to_remove: List[str]) -> None:
        """Remove keys from JSON storage file"""
        filepath = os.path.join(self.working_dir, filename)
        
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            removed_count = 0
            for key in keys_to_remove:
                if key in data:
                    del data[key]
                    removed_count += 1
            
            # Write back updated data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if removed_count > 0:
                print(f"✅ Removed {removed_count} entries from {filename}")
            
        except Exception as e:
            raise Exception(f"Failed to remove from {filename}: {str(e)}")
    
    def _remove_from_vector_db(self, filename: str, ids_to_remove: List[str]) -> None:
        """Remove entries from vector database JSON file"""
        filepath = os.path.join(self.working_dir, filename)
        
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                vdb_data = json.load(f)
            
            # Vector DB format: {"embedding_dim": 768, "data": [...]}
            if "data" not in vdb_data:
                return
            
            original_count = len(vdb_data["data"])
            
            # Filter out entries with matching IDs
            vdb_data["data"] = [
                entry for entry in vdb_data["data"] 
                if entry.get("__id__") not in ids_to_remove
            ]
            
            removed_count = original_count - len(vdb_data["data"])
            
            # Write back updated data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vdb_data, f, ensure_ascii=False, indent=2)
            
            if removed_count > 0:
                print(f"✅ Removed {removed_count} vectors from {filename}")
            
        except Exception as e:
            raise Exception(f"Failed to remove from vector DB {filename}: {str(e)}")
    
    def _find_entities_to_remove(self, chunk_ids: List[str]) -> Set[str]:
        """Find entities that are only connected to deleted chunks"""
        entities_to_remove = set()
        
        # Load text chunks to get chunk-entity relationships
        text_chunks_file = os.path.join(self.working_dir, "kv_store_text_chunks.json")
        if not os.path.exists(text_chunks_file):
            return entities_to_remove
        
        try:
            with open(text_chunks_file, 'r', encoding='utf-8') as f:
                text_chunks = json.load(f)
            
            # Map entities to chunks they appear in
            entity_chunk_map = {}
            
            for chunk_id, chunk_data in text_chunks.items():
                # Extract entities from chunk content using simple heuristics
                # In a real implementation, this would use NER or LLM extraction
                content = chunk_data.get('content', '')
                
                # Simple entity extraction (this is simplified)
                # Real implementation would need proper NLP processing
                entities_in_chunk = self._extract_entities_from_content(content)
                
                for entity in entities_in_chunk:
                    if entity not in entity_chunk_map:
                        entity_chunk_map[entity] = set()
                    entity_chunk_map[entity].add(chunk_id)
            
            # Find entities that only exist in chunks being deleted
            for entity, associated_chunks in entity_chunk_map.items():
                if associated_chunks.issubset(set(chunk_ids)):
                    entities_to_remove.add(entity)
            
            print(f"🔍 Found {len(entities_to_remove)} entities to remove")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not analyze entities: {str(e)}")
        
        return entities_to_remove
    
    def _extract_entities_from_content(self, content: str) -> Set[str]:
        """Simple entity extraction from content (placeholder implementation)"""
        entities = set()
        
        # This is a very simplified implementation
        # In reality, you'd want to use the same entity extraction logic
        # that LightRAG uses internally
        
        # For now, we'll use simple heuristics like capitalized words
        words = content.split()
        for word in words:
            # Simple heuristic: capitalized words longer than 2 chars
            if len(word) > 2 and word[0].isupper() and word.isalpha():
                entities.add(word.lower())
        
        return entities
    
    def _find_relations_to_remove(self, entities_to_remove: Set[str]) -> Set[str]:
        """Find relations that involve entities being removed"""
        relations_to_remove = set()
        
        relations_file = os.path.join(self.working_dir, "kv_store_full_relations.json")
        if not os.path.exists(relations_file):
            return relations_to_remove
        
        try:
            with open(relations_file, 'r', encoding='utf-8') as f:
                relations_data = json.load(f)
            
            for relation_id, relation_info in relations_data.items():
                # Check if this relation involves any entity being removed
                # Relation format may vary, adapt based on actual structure
                relation_content = relation_info.get('content', '').lower()
                
                for entity in entities_to_remove:
                    if entity in relation_content:
                        relations_to_remove.add(relation_id)
                        break
            
            print(f"🔍 Found {len(relations_to_remove)} relations to remove")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not analyze relations: {str(e)}")
        
        return relations_to_remove
    
    def _remove_related_entities_and_relations(self, chunk_ids: List[str]) -> None:
        """Remove entities and relations related to deleted chunks"""
        print("🔗 Analyzing entities and relations...")
        
        # Find entities that are only in deleted chunks
        entities_to_remove = self._find_entities_to_remove(chunk_ids)
        
        if entities_to_remove:
            # Find entity IDs from full_entities
            entity_ids_to_remove = self._find_entity_ids(entities_to_remove)
            
            # Find relations involving these entities
            relations_to_remove = self._find_relations_to_remove(entities_to_remove)
            
            # Remove entities
            if entity_ids_to_remove:
                self._remove_from_json_storage("kv_store_full_entities.json", list(entity_ids_to_remove))
                self._remove_from_vector_db("vdb_entities.json", list(entity_ids_to_remove))
            
            # Remove relations
            if relations_to_remove:
                self._remove_from_json_storage("kv_store_full_relations.json", list(relations_to_remove))
                self._remove_from_vector_db("vdb_relationships.json", list(relations_to_remove))
        
        print(f"✅ Entity/relation cleanup completed")
    
    def _find_entity_ids(self, entity_names: Set[str]) -> Set[str]:
        """Find entity IDs from entity names"""
        entity_ids = set()
        
        entities_file = os.path.join(self.working_dir, "kv_store_full_entities.json")
        if not os.path.exists(entities_file):
            return entity_ids
        
        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
            
            for entity_id, entity_info in entities_data.items():
                entity_content = entity_info.get('content', '').lower()
                
                for entity_name in entity_names:
                    if entity_name in entity_content:
                        entity_ids.add(entity_id)
                        break
            
        except Exception as e:
            print(f"⚠️ Warning: Could not find entity IDs: {str(e)}")
        
        return entity_ids
    
    def _update_graph_file(self, chunk_ids: List[str]) -> None:
        """Update GraphML file to remove deleted chunks and related nodes"""
        graph_file = os.path.join(self.working_dir, "graph_chunk_entity_relation.graphml")
        
        if not os.path.exists(graph_file):
            return
        
        try:
            # Parse GraphML
            tree = ET.parse(graph_file)
            root = tree.getroot()
            
            # Find namespace
            namespace = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
            if root.tag.startswith('{'):
                ns = root.tag.split('}')[0] + '}'
                namespace = {'graphml': ns[1:-1]}
            
            # Find graph element
            graph_elem = root.find('.//graphml:graph', namespace)
            if graph_elem is None:
                # Try without namespace
                graph_elem = root.find('.//graph')
            
            if graph_elem is not None:
                nodes_removed = 0
                edges_removed = 0
                
                # Remove nodes (chunks, entities, relations)
                nodes_to_remove = []
                for node in graph_elem.findall('.//node') + graph_elem.findall('.//graphml:node', namespace):
                    node_id = node.get('id', '')
                    
                    # Check if this node should be removed
                    if any(chunk_id in node_id for chunk_id in chunk_ids):
                        nodes_to_remove.append(node)
                
                for node in nodes_to_remove:
                    graph_elem.remove(node)
                    nodes_removed += 1
                
                # Remove edges connected to removed nodes
                edges_to_remove = []
                for edge in graph_elem.findall('.//edge') + graph_elem.findall('.//graphml:edge', namespace):
                    source = edge.get('source', '')
                    target = edge.get('target', '')
                    
                    # Check if edge connects to removed nodes
                    if any(chunk_id in source or chunk_id in target for chunk_id in chunk_ids):
                        edges_to_remove.append(edge)
                
                for edge in edges_to_remove:
                    graph_elem.remove(edge)
                    edges_removed += 1
                
                # Write updated GraphML
                tree.write(graph_file, encoding='utf-8', xml_declaration=True)
                
                print(f"✅ Updated graph: removed {nodes_removed} nodes, {edges_removed} edges")
            else:
                print("⚠️ Warning: Could not find graph element in GraphML")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not update graph file: {str(e)}")
    
    async def delete_document_selective(self, filename: str) -> Dict:
        """
        Selectively delete document without full KB rebuild
        Returns: {"success": bool, "message": str, "details": dict}
        """
        
        print(f"🚀 Starting selective deletion for: {filename}")
        backup_path = None
        
        try:
            # Step 1: Create backup
            print("📦 Creating backup...")
            backup_path = self._create_backup()
            
            # Step 2: Find document
            print("🔍 Searching for document...")
            doc_search = self._find_document_by_filename(filename)
            
            if not doc_search["found"]:
                raise Exception(doc_search["error"])
            
            doc_id = doc_search["doc_id"]
            chunk_ids = doc_search["chunks_list"] 
            llm_cache_entries = doc_search["llm_cache_entries"]
            
            print(f"📄 Found document: {doc_id}")
            print(f"🧩 Chunks to remove: {len(chunk_ids)}")
            print(f"💾 Cache entries to remove: {len(llm_cache_entries)}")
            
            # Step 3: Remove from document storage
            print("🗑️ Removing from document storage...")
            self._remove_from_json_storage("kv_store_doc_status.json", [doc_id])
            self._remove_from_json_storage("kv_store_full_docs.json", [doc_id])
            
            # Step 4: Remove chunks
            print("🗑️ Removing chunks...")
            self._remove_from_json_storage("kv_store_text_chunks.json", chunk_ids)
            self._remove_from_vector_db("vdb_chunks.json", chunk_ids)
            
            # Step 5: Remove LLM cache
            if llm_cache_entries:
                print("🗑️ Removing LLM cache entries...")
                self._remove_from_json_storage("kv_store_llm_response_cache.json", list(llm_cache_entries))
            
            # Step 6: Remove related entities/relations
            print("🔗 Processing entities and relations...")
            self._remove_related_entities_and_relations(chunk_ids)
            
            # Step 7: Update graph
            print("📊 Processing graph updates...")
            self._update_graph_file(chunk_ids)
            
            # Step 8: Cleanup backup
            print("🧹 Cleaning up...")
            self._cleanup_backup(backup_path)
            
            print(f"✅ Successfully deleted document: {filename}")
            
            return {
                "success": True,
                "message": f"Document '{filename}' deleted successfully using selective deletion",
                "details": {
                    "doc_id": doc_id,
                    "chunks_removed": len(chunk_ids),
                    "cache_entries_removed": len(llm_cache_entries),
                    "method": "selective_deletion"
                }
            }
            
        except Exception as e:
            print(f"❌ Error during selective deletion: {str(e)}")
            print(f"📋 Traceback: {traceback.format_exc()}")
            
            # Attempt rollback
            if backup_path:
                try:
                    print("🔄 Attempting rollback...")
                    self._rollback_from_backup(backup_path)
                    print("✅ Rollback completed successfully")
                except Exception as rollback_error:
                    print(f"❌ Rollback failed: {str(rollback_error)}")
                    return {
                        "success": False,
                        "message": f"Selective deletion failed and rollback failed: {str(e)} | Rollback error: {str(rollback_error)}",
                        "details": {"error": str(e), "rollback_error": str(rollback_error)}
                    }
            
            return {
                "success": False,
                "message": f"Selective deletion failed: {str(e)}",
                "details": {"error": str(e)}
            }

# Global deletion service
deletion_service = None

def get_kb_files_from_storage():
    """Get actual files stored in KB from doc_status"""
    doc_status_file = os.path.join(WORKING_DIR, "kv_store_doc_status.json")
    if not os.path.exists(doc_status_file):
        return []
    
    try:
        with open(doc_status_file, 'r', encoding='utf-8') as f:
            doc_status = json.load(f)
        
        files = []
        for doc_id, doc_info in doc_status.items():
            content_summary = doc_info.get('content_summary', '')
            # Extract filename from content_summary
            if '--- File:' in content_summary:
                start = content_summary.find('--- File:') + len('--- File:')
                end = content_summary.find('---', start)
                if end != -1:
                    filename_raw = content_summary[start:end].strip()
                    # Clean filename - remove any page metadata
                    if '|' in filename_raw:
                        filename = filename_raw.split('|')[0].strip()
                    else:
                        filename = filename_raw
                    
                    # Remove any remaining metadata markers
                    if 'Page:' in filename:
                        filename = filename.split('Page:')[0].strip()
                    
                    # Check what file actually exists in KB_TEXT_DIR
                    actual_filename = filename
                    if filename.endswith('.pdf'):
                        txt_version = filename.replace('.pdf', '.txt')
                        if os.path.exists(os.path.join(KB_TEXT_DIR, txt_version)):
                            actual_filename = txt_version
                    elif filename.endswith('.txt'):
                        if not os.path.exists(os.path.join(KB_TEXT_DIR, filename)):
                            pdf_version = filename.replace('.txt', '.pdf')
                            if os.path.exists(os.path.join(KB_TEXT_DIR, pdf_version)):
                                actual_filename = pdf_version
                    
                    files.append({
                        'filename': actual_filename,
                        'doc_id': doc_id,
                        'chunks_count': doc_info.get('chunks_count', 0),
                        'content_length': doc_info.get('content_length', 0),
                        'created_at': doc_info.get('created_at', ''),
                        'status': doc_info.get('status', 'unknown')
                    })
        return files
    except Exception as e:
        print(f"Error reading KB storage: {e}")
        return []

def extract_text_with_metadata(pdf_reader, filename):
    """Extract text with page and line metadata for better citation"""
    full_text = ""
    metadata = []
    
    for page_num, page in enumerate(pdf_reader.pages, 1):
        page_text = page.extract_text()
        if page_text.strip():
            lines = page_text.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                if line.strip():  # Skip empty lines
                    line_metadata = {
                        'text': line.strip(),
                        'source': filename,
                        'page': page_num,
                        'line': line_num
                    }
                    metadata.append(line_metadata)
            
            # Add page separator with metadata - but keep filename clean
            page_section = f"\n\n--- Page {page_num} of {filename} ---\n\n{page_text}\n"
            full_text += page_section
    
    return full_text, metadata

async def init_rag(force_rebuild=False):
    """Initialize RAG with smart rebuild logic"""
    global rag, deletion_service
    
    # Check if KB exists and has data
    kb_exists = os.path.exists(WORKING_DIR) and os.path.exists(os.path.join(WORKING_DIR, "kv_store_doc_status.json"))
    
    if force_rebuild or not kb_exists:
        print("Building/Rebuilding Knowledge Base...")
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
    else:
        print("Loading existing Knowledge Base...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gemini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=gemini_embed
        )
    )
    
    # Initialize storages
    await rag.initialize_storages()
    
    # Initialize pipeline status
    from lightrag.kg.shared_storage import initialize_pipeline_status
    await initialize_pipeline_status()
    
    # Initialize deletion service
    deletion_service = DocumentDeletionService(WORKING_DIR)
    
    # Only insert text if we're rebuilding or KB is empty
    if force_rebuild or not kb_exists:
        all_text = ""
        for filename in os.listdir(KB_TEXT_DIR):
            if filename.endswith('.txt'):
                file_path = os.path.join(KB_TEXT_DIR, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Keep the filename marker clean without extra metadata
                    all_text += f"\n\n--- File: {filename} ---\n\n{content}"
        
        if all_text:
            await rag.ainsert(all_text)
            print("Knowledge Base built successfully!")
        else:
            print("No text files found to insert")
    else:
        print("Knowledge Base loaded from existing data")

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
        
        # Extract text with metadata
        full_text, metadata = extract_text_with_metadata(reader, file.filename)
        
        # Save text file
        filename = file.filename.replace('.pdf', '.txt')
        text_file_path = os.path.join(KB_TEXT_DIR, filename)
        with open(text_file_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print("Extracted and saved text with metadata")
        
        # Add to RAG with clean filename marker
        clean_text_for_rag = f"\n\n--- File: {filename} ---\n\n{full_text}"
        
        global rag
        if rag is None:
            await init_rag()
        else:
            # Just insert the new document
            await rag.ainsert(clean_text_for_rag)
        
        return JSONResponse(content={"message": f"File '{filename}' added successfully"})
    except Exception as e:
        print(f"Error adding file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding file: {str(e)}")

@app.get("/list_files")
async def list_files():
    """Get files actually stored in KB with metadata"""
    print("Listing KB files with metadata")
    
    try:
        # Get files from actual KB storage
        kb_files = get_kb_files_from_storage()
        
        # Also get files from text directory for comparison
        text_files = []
        if os.path.exists(KB_TEXT_DIR):
            for f in os.listdir(KB_TEXT_DIR):
                if f.endswith('.txt'):
                    file_path = os.path.join(KB_TEXT_DIR, f)
                    stats = os.stat(file_path)
                    text_files.append({
                        'filename': f,
                        'size': stats.st_size,
                        'modified': datetime.fromtimestamp(stats.st_mtime).isoformat()
                    })
        
        return JSONResponse(content={
            "kb_files": kb_files,
            "text_files": text_files,
            "summary": {
                "total_documents": len(kb_files),
                "total_chunks": sum(f.get('chunks_count', 0) for f in kb_files)
            }
        })
    except Exception as e:
        print(f"Error listing files: {str(e)}")
        return JSONResponse(content={"kb_files": [], "text_files": [], "summary": {}})

@app.delete("/delete_file/{filename}")
async def delete_file(filename: str):
    print(f"Deleting file: {filename}")
    
    # Handle both .pdf and .txt extensions since files are stored as .txt
    # but might be displayed as .pdf in the frontend
    txt_filename = filename.replace('.pdf', '.txt') if filename.endswith('.pdf') else filename
    pdf_filename = filename.replace('.txt', '.pdf') if filename.endswith('.txt') else filename
    
    # Try to find the actual text file
    txt_file_path = os.path.join(KB_TEXT_DIR, txt_filename)
    pdf_file_path = os.path.join(KB_TEXT_DIR, pdf_filename)
    
    file_found = False
    actual_filename = None
    
    if os.path.exists(txt_file_path):
        os.remove(txt_file_path)
        file_found = True
        actual_filename = txt_filename
        print(f"Removed {txt_filename} from text directory")
    elif os.path.exists(pdf_file_path):
        os.remove(pdf_file_path)
        file_found = True
        actual_filename = pdf_filename
        print(f"Removed {pdf_filename} from text directory")
    
    if not file_found:
        print(f"File not found: {filename} (tried {txt_filename} and {pdf_filename})")
        available_files = os.listdir(KB_TEXT_DIR) if os.path.exists(KB_TEXT_DIR) else []
        print(f"Available files: {available_files}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    # Try selective deletion first
    global deletion_service
    if deletion_service is None:
        deletion_service = DocumentDeletionService(WORKING_DIR)
    
    try:
        print("🚀 Attempting selective deletion...")
        deletion_result = await deletion_service.delete_document_selective(actual_filename)
        
        if deletion_result["success"]:
            print("✅ Selective deletion successful!")
            return JSONResponse(content={
                "message": deletion_result["message"],
                "details": deletion_result["details"]
            })
        else:
            print(f"❌ Selective deletion failed: {deletion_result['message']}")
            # Fall back to full rebuild
            print("🔄 Falling back to full rebuild...")
            
    except Exception as e:
        print(f"❌ Selective deletion error: {str(e)}")
        print("🔄 Falling back to full rebuild...")
    
    # Fallback: Force rebuild KB to remove from storage
    try:
        global rag
        await init_rag(force_rebuild=True)
        return JSONResponse(content={"message": f"File '{actual_filename}' deleted and KB rebuilt (fallback method)"})
    except Exception as e:
        print(f"❌ Full rebuild also failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.post("/query")
async def query_kb(query: dict):
    print(f"Querying KB: {query}")
    q = query.get("query")
    mode = query.get("mode", "hybrid")
    instruction = query.get("instruction", "Trả lời tập trung vào câu hỏi, không trả lời thông tin không liên quan, nhất là chú ý đến các keyword, tên dự án, đừng để bị nhầm giữa các keyword, và cùng không kiểu hỏi requirements trả lời thêm price, trả lời chính xác tới câu hỏi của người dùng")  # Optimized shorter instruction
    
    if not q:
        raise HTTPException(status_code=400, detail="No query provided")
    
    try:
        print(f"Querying KB with mode: {mode}")
        
        # Add instruction to query
        final_query = q
        if instruction.strip():
            final_query = f"Instruction: {instruction.strip()}\n\nQuery: {q}"
        
        result = await rag.aquery(final_query, param=QueryParam(mode=mode))
        
        # Try to extract source information from result
        # This is a simple approach - LightRAG should ideally support this natively
        sources = []
        if "--- File:" in result:
            import re
            file_matches = re.findall(r'--- File: ([^|]+?)(?:\s*\||\s*---)', result)
            page_matches = re.findall(r'Page: (\d+)', result)
            sources = list(set(file_matches))  # Remove duplicates
        
        return JSONResponse(content={
            "result": result,
            "mode_used": mode,
            "sources": sources,
            "query_time": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error querying: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying: {str(e)}")

@app.get("/kb_status")
async def get_kb_status():
    """Get detailed KB status"""
    try:
        kb_files = get_kb_files_from_storage()
        
        # Get storage file sizes
        storage_info = {}
        if os.path.exists(WORKING_DIR):
            for filename in os.listdir(WORKING_DIR):
                file_path = os.path.join(WORKING_DIR, filename)
                if os.path.isfile(file_path):
                    storage_info[filename] = {
                        'size': os.path.getsize(file_path),
                        'modified': datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                    }
        
        return JSONResponse(content={
            "status": "active" if rag else "inactive",
            "documents": len(kb_files),
            "total_chunks": sum(f.get('chunks_count', 0) for f in kb_files),
            "storage_files": storage_info,
            "working_directory": WORKING_DIR
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)})

@app.post("/set_gemini_api_key")
async def set_gemini_api_key(data: dict = Body(...)):
    """
    Set or update the Gemini API key at runtime.
    Request body: { "api_key": "your-new-api-key" }
    """
    api_key = data.get("api_key")
    if not api_key or not isinstance(api_key, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'api_key'")
    try:
        configure_gemini(api_key)
        return JSONResponse(content={"message": "Gemini API key updated successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update Gemini API key: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)