
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader

class VectorDBController:
    def __init__(self, embeddings, text_splitter):
        self.embeddings = embeddings
        self.text_splitter = text_splitter
        self.vector_db = None
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.collection_name = "knowledge_base_0"
        # FIX: Lấy dimension từ embeddings thực tế
        self.embeddings_dimension = self._get_embeddings_dimension()
        self.existing_source = set()
        self.document_tracker = set()

    def _get_embeddings_dimension(self):
        """Get embeddings dimension dynamically"""
        try:
            # Test với một text ngắn để lấy dimension
            test_embedding = self.embeddings.embed_query("test")
            return len(test_embedding)
        except:
            return 768  # fallback default


    def load_vector_db(self, vector_db_name: str):
        """
        Load vector db with different sources
        """
        if vector_db_name == "in-memory":
            from langchain_core.vectorstores import InMemoryVectorStore

            self.vector_db = InMemoryVectorStore(self.embeddings)

        elif vector_db_name == "qdrant":
            from qdrant_client.models import Distance, VectorParams
            from langchain_qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

            print("Connecting to Qdrant at ", self.qdrant_host, self.qdrant_port)
            try:
                self.qdrant_client = QdrantClient(
                    host=self.qdrant_host, 
                    port=self.qdrant_port,
                    timeout=10.0
                )

                # test connection
                collections = self.qdrant_client.get_collections()
                print(f"Connected to Qdrant successfully with {len(collections.collections)} collections")

                print("Creating collection if not exists")
                # Chỉ tạo collection nếu chưa tồn tại, không xóa data cũ
                try:
                    self.qdrant_client.get_collection(self.collection_name)
                    print(f"Collection {self.collection_name} already exists")
                except Exception:
                    # Collection chưa tồn tại, tạo mới
                    self.qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.embeddings_dimension, distance=Distance.COSINE)
                    )
                    print(f"Created new collection {self.collection_name}")

                print("Initialized Qdrant vector DB")
                self.vector_db = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding=self.embeddings
                )
            except Exception as e:
                print(f"Error connecting to Qdrant: {e}")
                raise e


    def load_existing_documents(self):
        """
        Load existing documents from the vector DB
        """
        try:
            #get collection info first
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            print(f"Collection has {collection_info.points_count} total points")

            if collection_info.points_count == 0:
                print("No existing documents found")
                return "No existing documents found"

            #get all points from collection
            scroll_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=10000,
                with_payload=True,
                with_vectors=False  # Không cần vectors để tiết kiệm băng thông
            )

            self.existing_source = set()
            for point in scroll_results[0]:
                if point.payload:
                    # Kiểm tra cả hai cấu trúc payload có thể có
                    source = None
                    if "source" in point.payload:
                        source = point.payload["source"]
                    elif "metadata" in point.payload and isinstance(point.payload["metadata"], dict):
                        source = point.payload["metadata"].get("source")
                    
                    if source and source != "unknown":
                        self.existing_source.add(source)

            #update the tracker
            self.document_tracker.update(self.existing_source)
            print(f"Loaded {len(self.existing_source)} existing sources: {list(self.existing_source)}")
            return f"Loaded {len(self.existing_source)} existing sources"
        except Exception as e:
            print(f"Error loading existing documents: {e}")
            # Không raise exception để không crash app
            return f"Error loading documents: {str(e)}"
            

    def delete_documents(self, filename: str):
        """
        Delete documents from the vector DB
        """
        if not filename:
            print("Filename is required")
            return
        
        try:
            #get the document from the vector DB
            if filename not in self.document_tracker:
                print(f"Document {filename} not found")
                return
            
            #delete the document from Qdrant using filter syntax
            delete_result = self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="metadata.source", match=MatchValue(value=filename))
                    ]
                )
            )
            print(f"Deleted {delete_result} from the vector DB")

            #remove from the tracker
            self.document_tracker.discard(filename)
            return f"Document {filename} deleted successfully from the vector DB and tracker"
        except Exception as e:
            print(f"Error deleting documents: {e}")
            raise e

    def add_documents(self, file_path: str):
        """
        Add documents to the vector DB
        """
        if not file_path:
            print("File path is required")
            return
        
        try:
            filename = os.path.basename(file_path)

            #check if the file is already in the vector DB
            if filename in self.document_tracker:
                print(f"Document {filename} already exists")
                return
            
            # FIX: Thêm phần load document
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
            
            docs = loader.load()
            
            if not docs:
                print(f"No documents found in {file_path}")
                return
            
            splits = self.text_splitter.split_documents(docs)  # FIX: split_documents thay vì split_text

            if not splits:
                print(f"No text chunks created from {filename}")
                return
            
            #add source metadata to each chunk
            for split in splits:
                split.metadata["source"] = filename
                split.metadata["author"] = "unknown"

            #add the splits to the vector DB
            self.vector_db.add_documents(splits)
            
            #track the document
            self.document_tracker.add(filename)
            # print(f"Added {len(splits)} splits to the vector DB and tracked {filename}")
            return f"Document {filename} added successfully to the vector DB and tracked with {len(splits)} splits"
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise e
            


