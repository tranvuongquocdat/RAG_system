"""
LLM Model module for RAG system
Quản lý Gemini LLM model và các operations liên quan
"""

import time
from typing import List, Optional, Dict, Any, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_config
from utils import setup_logging

class LLMManager:
    """
    Quản lý LLM operations sử dụng Gemini
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Khởi tạo LLMManager
        
        Args:
            api_key: Gemini API key (optional, sẽ lấy từ config nếu không có)
            model_name: Tên model (optional, sẽ lấy từ config nếu không có)
        """
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
        
        # Sử dụng config hoặc parameters
        self.api_key = api_key or self.config.model.gemini_api_key
        self.model_name = model_name or self.config.model.gemini_model_name
        
        if not self.api_key:
            raise ValueError("Gemini API key không được cấu hình")
        
        # Khởi tạo LLM model
        self._initialize_model()
        
        # Prompt templates
        self._setup_prompts()
        
        self.logger.info(f"✅ LLMManager đã khởi tạo với model: {self.model_name}")
    
    def _initialize_model(self) -> None:
        """Khởi tạo Gemini LLM model"""
        try:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=self.api_key,
                model=self.model_name,
                temperature=self.config.model.temperature,
                max_output_tokens=self.config.model.max_tokens,
                convert_system_message_to_human=True  # Gemini requirement
            )
            
            # Test connection
            test_response = self.llm.invoke([HumanMessage(content="Hello, test connection")])
            self.logger.info("✅ Kết nối Gemini LLM thành công")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi khởi tạo Gemini LLM: {str(e)}")
            raise
    
    def _setup_prompts(self) -> None:
        """Thiết lập các prompt templates"""
        
        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một AI assistant chuyên nghiệp hỗ trợ trả lời câu hỏi dựa trên tài liệu doanh nghiệp.

Quy tắc trả lời:
1. Chỉ sử dụng thông tin từ context được cung cấp
2. Nếu không tìm thấy thông tin trong context, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu"
3. Trả lời bằng tiếng Việt một cách rõ ràng và chuyên nghiệp
4. Trích dẫn nguồn thông tin khi có thể
5. Nếu thông tin không chắc chắn, hãy nói rõ

Context từ tài liệu:
{context}"""),
            ("human", "{question}")
        ])
        
        # Summary prompt template
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn cần tóm tắt nội dung tài liệu sau một cách ngắn gọn và chính xác.
            
Yêu cầu:
1. Tóm tắt không quá 200 từ
2. Nêu các điểm chính
3. Sử dụng tiếng Việt
4. Giữ nguyên các thuật ngữ chuyên môn quan trọng"""),
            ("human", "Tóm tắt nội dung sau:\n\n{content}")
        ])
        
        # Question generation prompt
        self.question_gen_prompt = ChatPromptTemplate.from_messages([
            ("system", """Dựa trên nội dung tài liệu, hãy tạo ra 3-5 câu hỏi quan trọng mà người dùng có thể muốn hỏi.
            
Yêu cầu:
1. Câu hỏi phải liên quan trực tiếp đến nội dung
2. Đa dạng về loại câu hỏi (thông tin, phân tích, so sánh)
3. Sử dụng tiếng Việt
4. Mỗi câu hỏi trên một dòng, bắt đầu bằng "- " """),
            ("human", "Tạo câu hỏi từ nội dung:\n\n{content}")
        ])
    
    def generate_response(
        self, 
        question: str, 
        context: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Sinh câu trả lời RAG
        
        Args:
            question: Câu hỏi của người dùng
            context: Context từ retrieved documents
            system_prompt: Custom system prompt (optional)
        
        Returns:
            Câu trả lời được sinh ra
        """
        try:
            if system_prompt:
                # Sử dụng custom prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", f"Context: {context}\n\nCâu hỏi: {question}")
                ])
            else:
                # Sử dụng default RAG prompt
                prompt = self.rag_prompt
            
            # Tạo chain
            chain = prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = chain.invoke({
                "question": question,
                "context": context
            })
            
            self.logger.debug(f"✅ Đã sinh câu trả lời cho: {question[:50]}...")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi sinh câu trả lời: {str(e)}")
            return "Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại."
    
    def summarize_content(self, content: str) -> str:
        """
        Tóm tắt nội dung
        
        Args:
            content: Nội dung cần tóm tắt
        
        Returns:
            Bản tóm tắt
        """
        try:
            chain = self.summary_prompt | self.llm | StrOutputParser()
            
            summary = chain.invoke({"content": content})
            
            self.logger.debug(f"✅ Đã tóm tắt nội dung ({len(content)} chars -> {len(summary)} chars)")
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi tóm tắt: {str(e)}")
            return "Không thể tóm tắt nội dung này."
    
    def generate_questions(self, content: str) -> List[str]:
        """
        Sinh câu hỏi từ nội dung
        
        Args:
            content: Nội dung để sinh câu hỏi
        
        Returns:
            Danh sách câu hỏi
        """
        try:
            chain = self.question_gen_prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({"content": content})
            
            # Parse câu hỏi (mỗi câu hỏi bắt đầu bằng "- ")
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    questions.append(line[2:].strip())
                elif line and not line.startswith('-'):
                    # Trường hợp không có "- " prefix
                    questions.append(line)
            
            self.logger.debug(f"✅ Đã sinh {len(questions)} câu hỏi từ nội dung")
            return questions
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi sinh câu hỏi: {str(e)}")
            return []
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None
    ) -> str:
        """
        Chat completion với lịch sử hội thoại
        
        Args:
            messages: Danh sách messages [{"role": "user/assistant", "content": "..."}]
            temperature: Temperature setting (optional)
        
        Returns:
            Response từ model
        """
        try:
            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
            
            # Temporary temperature change if specified
            if temperature is not None:
                original_temp = self.llm.temperature
                self.llm.temperature = temperature
            
            response = self.llm.invoke(langchain_messages)
            
            # Restore original temperature
            if temperature is not None:
                self.llm.temperature = original_temp
            
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi chat completion: {str(e)}")
            return "Xin lỗi, tôi gặp lỗi khi xử lý tin nhắn của bạn."
    
    def evaluate_answer_quality(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Đánh giá chất lượng câu trả lời
        
        Args:
            question: Câu hỏi gốc
            answer: Câu trả lời cần đánh giá
            context: Context đã sử dụng
        
        Returns:
            Dictionary chứa điểm đánh giá
        """
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn cần đánh giá chất lượng câu trả lời dựa trên các tiêu chí sau:

1. Relevance (0-10): Câu trả lời có liên quan đến câu hỏi không?
2. Accuracy (0-10): Câu trả lời có chính xác dựa trên context không?
3. Completeness (0-10): Câu trả lời có đầy đủ thông tin không?
4. Clarity (0-10): Câu trả lời có rõ ràng, dễ hiểu không?

Trả lời theo format JSON:
{
    "relevance": <score>,
    "accuracy": <score>,
    "completeness": <score>,
    "clarity": <score>,
    "overall": <average_score>,
    "feedback": "<nhận xét chi tiết>"
}"""),
            ("human", f"""
Câu hỏi: {question}

Context: {context}

Câu trả lời: {answer}

Hãy đánh giá câu trả lời này:""")
        ])
        
        try:
            chain = evaluation_prompt | self.llm | StrOutputParser()
            response = chain.invoke({})
            
            # Parse JSON response
            import json
            evaluation = json.loads(response)
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi đánh giá câu trả lời: {str(e)}")
            return {
                "relevance": 5,
                "accuracy": 5,
                "completeness": 5,
                "clarity": 5,
                "overall": 5,
                "feedback": "Không thể đánh giá do lỗi hệ thống"
            }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Trích xuất keywords từ text
        
        Args:
            text: Text cần trích xuất keywords
        
        Returns:
            Danh sách keywords
        """
        keyword_prompt = ChatPromptTemplate.from_messages([
            ("system", """Trích xuất 5-10 từ khóa quan trọng nhất từ văn bản.
            
Yêu cầu:
1. Chỉ trả về các từ khóa, mỗi từ trên một dòng
2. Ưu tiên các thuật ngữ chuyên môn
3. Không bao gồm từ phổ biến (và, hoặc, của, etc.)
4. Giữ nguyên tiếng Việt có dấu"""),
            ("human", f"Trích xuất keywords từ:\n\n{text}")
        ])
        
        try:
            chain = keyword_prompt | self.llm | StrOutputParser()
            response = chain.invoke({})
            
            keywords = [kw.strip() for kw in response.split('\n') if kw.strip()]
            
            self.logger.debug(f"✅ Đã trích xuất {len(keywords)} keywords")
            return keywords
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi khi trích xuất keywords: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về model hiện tại
        
        Returns:
            Dictionary chứa thông tin model
        """
        return {
            "model_name": self.model_name,
            "temperature": self.config.model.temperature,
            "max_tokens": self.config.model.max_tokens,
            "provider": "Google Gemini"
        }

class ConversationManager:
    """
    Quản lý lịch sử hội thoại
    """
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self.logger = llm_manager.logger
    
    def start_conversation(self, conversation_id: str, system_prompt: Optional[str] = None) -> None:
        """
        Bắt đầu cuộc hội thoại mới
        
        Args:
            conversation_id: ID của cuộc hội thoại
            system_prompt: System prompt cho cuộc hội thoại
        """
        self.conversations[conversation_id] = []
        
        if system_prompt:
            self.conversations[conversation_id].append({
                "role": "system",
                "content": system_prompt
            })
        
        self.logger.info(f"✅ Bắt đầu hội thoại: {conversation_id}")
    
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Thêm message vào hội thoại
        
        Args:
            conversation_id: ID cuộc hội thoại
            role: Role của message (user/assistant)
            content: Nội dung message
        """
        if conversation_id not in self.conversations:
            self.start_conversation(conversation_id)
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content
        })
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Lấy lịch sử hội thoại
        
        Args:
            conversation_id: ID cuộc hội thoại
        
        Returns:
            Danh sách messages
        """
        return self.conversations.get(conversation_id, [])
    
    def chat(self, conversation_id: str, user_message: str) -> str:
        """
        Chat với context của cuộc hội thoại
        
        Args:
            conversation_id: ID cuộc hội thoại
            user_message: Message từ user
        
        Returns:
            Response từ assistant
        """
        # Thêm user message
        self.add_message(conversation_id, "user", user_message)
        
        # Lấy response
        messages = self.get_conversation(conversation_id)
        response = self.llm_manager.chat_completion(messages)
        
        # Thêm assistant response
        self.add_message(conversation_id, "assistant", response)
        
        return response

# Factory function để tạo LLMManager
def create_llm_manager(api_key: Optional[str] = None, model_name: Optional[str] = None) -> LLMManager:
    """
    Factory function để tạo LLMManager
    
    Args:
        api_key: Gemini API key (optional)
        model_name: Model name (optional)
    
    Returns:
        LLMManager instance
    """
    return LLMManager(api_key=api_key, model_name=model_name)
