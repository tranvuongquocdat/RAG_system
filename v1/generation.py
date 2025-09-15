"""
Generation module for RAG system
Module sinh tạo câu trả lời với RAG chain
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.schema import BaseRetriever

from config import get_config
from utils import setup_logging
from model import LLMManager
from retrieval import RetrievalEngine, ContextBuilder, RetrievalResult

class ResponseType(Enum):
    """Loại response"""
    DIRECT_ANSWER = "direct_answer"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    EXPLANATION = "explanation"
    NOT_FOUND = "not_found"

@dataclass
class GenerationResult:
    """
    Kết quả sinh tạo câu trả lời
    """
    answer: str
    response_type: ResponseType
    confidence_score: float
    sources: List[RetrievalResult]
    context_used: str
    processing_time: float
    metadata: Dict[str, Any]

class PromptManager:
    """
    Quản lý các prompt templates cho RAG
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
        self._setup_prompts()
    
    def _setup_prompts(self) -> None:
        """Thiết lập các prompt templates"""
        
        # Main RAG prompt
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là một AI assistant chuyên nghiệp hỗ trợ trả lời câu hỏi dựa trên tài liệu doanh nghiệp.

NGUYÊN TẮC TRẢ LỜI:
1. CHỈ sử dụng thông tin từ context được cung cấp
2. Nếu không tìm thấy thông tin trong context, hãy nói rõ "Tôi không tìm thấy thông tin này trong tài liệu được cung cấp"
3. Trả lời bằng tiếng Việt một cách rõ ràng, chuyên nghiệp và dễ hiểu
4. Trích dẫn nguồn thông tin khi có thể (tên file, loại tài liệu)
5. Nếu thông tin không chắc chắn, hãy nói rõ mức độ tin cậy
6. Cấu trúc câu trả lời có logic, dễ theo dõi
7. Sử dụng bullet points hoặc numbering khi liệt kê nhiều thông tin

CONTEXT TỪ TÀI LIỆU:
{context}

Hãy trả lời câu hỏi dựa trên context trên."""),
            ("human", "{question}")
        ])
        
        # Summary prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn cần tóm tắt thông tin từ các tài liệu được cung cấp.

YÊU CẦU:
1. Tóm tắt ngắn gọn, súc tích nhưng đầy đủ ý chính
2. Sắp xếp thông tin theo mức độ quan trọng
3. Sử dụng tiếng Việt rõ ràng
4. Nêu rõ nguồn tài liệu nếu có
5. Không thêm thông tin không có trong context

CONTEXT:
{context}"""),
            ("human", "Tóm tắt thông tin về: {topic}")
        ])
        
        # Comparison prompt
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn cần so sánh các thông tin từ tài liệu được cung cấp.

YÊU CẦU:
1. So sánh một cách khách quan dựa trên thông tin có sẵn
2. Nêu rõ điểm giống và khác biệt
3. Sử dụng bảng hoặc bullet points để trình bày rõ ràng
4. Trích dẫn nguồn cho từng thông tin
5. Nếu thiếu thông tin để so sánh, hãy nói rõ

CONTEXT:
{context}"""),
            ("human", "So sánh: {comparison_query}")
        ])
        
        # Explanation prompt
        self.explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn cần giải thích chi tiết về một chủ đề dựa trên tài liệu.

YÊU CẦU:
1. Giải thích từ cơ bản đến nâng cao
2. Sử dụng ví dụ cụ thể từ tài liệu nếu có
3. Cấu trúc giải thích logic, dễ hiểu
4. Làm rõ các thuật ngữ chuyên môn
5. Kết luận với những điểm quan trọng nhất

CONTEXT:
{context}"""),
            ("human", "Giải thích về: {topic}")
        ])
        
        # Response type classification prompt
        self.classify_prompt = ChatPromptTemplate.from_messages([
            ("system", """Phân loại loại câu hỏi để chọn cách trả lời phù hợp.

CÁC LOẠI CÂU HỎI:
- direct_answer: Câu hỏi trực tiếp cần thông tin cụ thể
- summary: Yêu cầu tóm tắt thông tin
- comparison: So sánh các thông tin
- explanation: Giải thích khái niệm, quy trình
- not_found: Không tìm thấy thông tin liên quan

Trả lời chỉ với tên loại (ví dụ: direct_answer)"""),
            ("human", "Phân loại câu hỏi: {question}")
        ])

    def get_prompt_by_type(self, response_type: ResponseType) -> ChatPromptTemplate:
        """
        Lấy prompt template theo loại response
        
        Args:
            response_type: Loại response
            
        Returns:
            ChatPromptTemplate tương ứng
        """
        prompt_map = {
            ResponseType.DIRECT_ANSWER: self.rag_prompt,
            ResponseType.SUMMARY: self.summary_prompt,
            ResponseType.COMPARISON: self.comparison_prompt,
            ResponseType.EXPLANATION: self.explanation_prompt,
            ResponseType.NOT_FOUND: self.rag_prompt
        }
        
        return prompt_map.get(response_type, self.rag_prompt)

class RAGChain:
    """
    RAG Chain chính để sinh tạo câu trả lời
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        retrieval_engine: RetrievalEngine,
        context_builder: ContextBuilder
    ):
        """
        Khởi tạo RAGChain
        
        Args:
            llm_manager: LLMManager instance
            retrieval_engine: RetrievalEngine instance
            context_builder: ContextBuilder instance
        """
        self.llm_manager = llm_manager
        self.retrieval_engine = retrieval_engine
        self.context_builder = context_builder
        self.prompt_manager = PromptManager()
        
        self.config = get_config()
        self.logger = setup_logging(self.config.app.log_level, self.config.app.log_file)
        
        self.logger.info("✅ RAGChain đã khởi tạo")
    
    def classify_question_type(self, question: str) -> ResponseType:
        """
        Phân loại loại câu hỏi
        
        Args:
            question: Câu hỏi từ user
            
        Returns:
            ResponseType
        """
        try:
            chain = self.prompt_manager.classify_prompt | self.llm_manager.llm | StrOutputParser()
            
            result = chain.invoke({"question": question})
            result = result.strip().lower()
            
            # Map string to enum
            type_map = {
                'direct_answer': ResponseType.DIRECT_ANSWER,
                'summary': ResponseType.SUMMARY,
                'comparison': ResponseType.COMPARISON,
                'explanation': ResponseType.EXPLANATION,
                'not_found': ResponseType.NOT_FOUND
            }
            
            return type_map.get(result, ResponseType.DIRECT_ANSWER)
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi classify question: {str(e)}")
            return ResponseType.DIRECT_ANSWER
    
    def generate_answer(
        self,
        question: str,
        max_sources: Optional[int] = None,
        response_type: Optional[ResponseType] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> GenerationResult:
        """
        Sinh tạo câu trả lời cho câu hỏi
        
        Args:
            question: Câu hỏi từ user
            max_sources: Số lượng sources tối đa
            response_type: Loại response (auto-detect nếu None)
            filters: Bộ lọc cho retrieval
            
        Returns:
            GenerationResult
        """
        start_time = time.time()
        
        try:
            # Auto-classify question type if not provided
            if response_type is None:
                response_type = self.classify_question_type(question)
            
            # Retrieve relevant documents
            if max_sources is None:
                max_sources = self.config.rag.top_k
            
            sources = self.retrieval_engine.retrieve_similar_documents(
                query=question,
                top_k=max_sources,
                filters=filters
            )
            
            # Check if we found relevant information
            if not sources or (sources and sources[0].score < self.config.rag.score_threshold):
                return self._generate_not_found_response(question, start_time)
            
            # Build context
            context = self.context_builder.build_context(sources)
            
            # Generate answer based on response type
            answer = self._generate_by_type(question, context, response_type)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(sources, answer)
            
            processing_time = time.time() - start_time
            
            result = GenerationResult(
                answer=answer,
                response_type=response_type,
                confidence_score=confidence_score,
                sources=sources,
                context_used=context,
                processing_time=processing_time,
                metadata={
                    'num_sources': len(sources),
                    'avg_source_score': sum(s.score for s in sources) / len(sources),
                    'question_length': len(question),
                    'answer_length': len(answer),
                    'context_length': len(context)
                }
            )
            
            self.logger.info(f"✅ Generated answer ({response_type.value}): {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi generate answer: {str(e)}")
            return self._generate_error_response(question, str(e), start_time)
    
    def _generate_by_type(
        self,
        question: str,
        context: str,
        response_type: ResponseType
    ) -> str:
        """
        Sinh tạo answer dựa trên response type
        
        Args:
            question: Câu hỏi
            context: Context từ retrieved docs
            response_type: Loại response
            
        Returns:
            Generated answer
        """
        prompt = self.prompt_manager.get_prompt_by_type(response_type)
        chain = prompt | self.llm_manager.llm | StrOutputParser()
        
        # Prepare input based on response type
        if response_type == ResponseType.SUMMARY:
            return chain.invoke({
                "context": context,
                "topic": question
            })
        elif response_type == ResponseType.COMPARISON:
            return chain.invoke({
                "context": context,
                "comparison_query": question
            })
        elif response_type == ResponseType.EXPLANATION:
            return chain.invoke({
                "context": context,
                "topic": question
            })
        else:  # DIRECT_ANSWER and others
            return chain.invoke({
                "context": context,
                "question": question
            })
    
    def _calculate_confidence_score(
        self,
        sources: List[RetrievalResult],
        answer: str
    ) -> float:
        """
        Tính confidence score cho câu trả lời
        
        Args:
            sources: Danh sách sources
            answer: Câu trả lời được sinh
            
        Returns:
            Confidence score (0-1)
        """
        if not sources:
            return 0.0
        
        # Base score from source similarity
        avg_similarity = sum(s.score for s in sources) / len(sources)
        
        # Adjust based on number of sources
        source_factor = min(len(sources) / 3, 1.0)  # Optimal around 3 sources
        
        # Adjust based on answer length (not too short, not too long)
        answer_length = len(answer)
        if 50 <= answer_length <= 500:
            length_factor = 1.0
        elif answer_length < 50:
            length_factor = 0.7
        else:
            length_factor = 0.9
        
        # Check for uncertainty phrases in answer
        uncertainty_phrases = [
            'không chắc chắn', 'có thể', 'dường như', 'không rõ',
            'không tìm thấy', 'thiếu thông tin'
        ]
        
        uncertainty_factor = 1.0
        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                uncertainty_factor = 0.6
                break
        
        # Final confidence score
        confidence = avg_similarity * source_factor * length_factor * uncertainty_factor
        return min(confidence, 1.0)
    
    def _generate_not_found_response(
        self,
        question: str,
        start_time: float
    ) -> GenerationResult:
        """
        Sinh response khi không tìm thấy thông tin
        """
        answer = f"Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi '{question}' trong cơ sở tài liệu hiện có. Bạn có thể thử:"
        answer += "\n1. Diễn đạt câu hỏi theo cách khác"
        answer += "\n2. Sử dụng từ khóa khác"
        answer += "\n3. Kiểm tra xem tài liệu có chứa thông tin này không"
        
        return GenerationResult(
            answer=answer,
            response_type=ResponseType.NOT_FOUND,
            confidence_score=0.0,
            sources=[],
            context_used="",
            processing_time=time.time() - start_time,
            metadata={'reason': 'no_relevant_documents'}
        )
    
    def _generate_error_response(
        self,
        question: str,
        error: str,
        start_time: float
    ) -> GenerationResult:
        """
        Sinh response khi có lỗi
        """
        answer = "Xin lỗi, tôi gặp lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."
        
        return GenerationResult(
            answer=answer,
            response_type=ResponseType.NOT_FOUND,
            confidence_score=0.0,
            sources=[],
            context_used="",
            processing_time=time.time() - start_time,
            metadata={'error': error}
        )
    
    def generate_followup_questions(
        self,
        question: str,
        answer: str,
        sources: List[RetrievalResult]
    ) -> List[str]:
        """
        Sinh câu hỏi follow-up
        
        Args:
            question: Câu hỏi gốc
            answer: Câu trả lời đã sinh
            sources: Sources đã sử dụng
            
        Returns:
            Danh sách câu hỏi follow-up
        """
        try:
            if not sources:
                return []
            
            # Build context from sources
            context = self.context_builder.build_context(sources[:3])
            
            followup_prompt = ChatPromptTemplate.from_messages([
                ("system", """Dựa trên câu hỏi, câu trả lời và context, hãy tạo 3 câu hỏi follow-up hữu ích mà người dùng có thể quan tâm.

YÊU CẦU:
1. Câu hỏi phải liên quan đến chủ đề chính
2. Có thể trả lời được dựa trên context hiện có
3. Đa dạng về góc độ (chi tiết hơn, mở rộng, ứng dụng)
4. Mỗi câu hỏi trên một dòng, bắt đầu bằng "- "

CONTEXT:
{context}"""),
                ("human", f"Câu hỏi gốc: {question}\nCâu trả lời: {answer}\n\nTạo câu hỏi follow-up:")
            ])
            
            chain = followup_prompt | self.llm_manager.llm | StrOutputParser()
            
            response = chain.invoke({"context": context})
            
            # Parse questions
            questions = []
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('- '):
                    questions.append(line[2:].strip())
                elif line and not line.startswith('-'):
                    questions.append(line)
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi generate followup questions: {str(e)}")
            return []
    
    def batch_generate(
        self,
        questions: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> List[GenerationResult]:
        """
        Sinh tạo answers cho multiple questions
        
        Args:
            questions: Danh sách câu hỏi
            filters: Bộ lọc chung
            
        Returns:
            Danh sách GenerationResult
        """
        results = []
        
        for question in questions:
            try:
                result = self.generate_answer(question, filters=filters)
                results.append(result)
            except Exception as e:
                self.logger.error(f"❌ Lỗi batch generate cho '{question}': {str(e)}")
                error_result = self._generate_error_response(question, str(e), time.time())
                results.append(error_result)
        
        self.logger.info(f"✅ Batch generated {len(results)} answers")
        return results

class ConversationalRAG:
    """
    RAG với khả năng duy trì context hội thoại
    """
    
    def __init__(self, rag_chain: RAGChain):
        """
        Khởi tạo ConversationalRAG
        
        Args:
            rag_chain: RAGChain instance
        """
        self.rag_chain = rag_chain
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = rag_chain.logger
    
    def start_conversation(self, conversation_id: str) -> None:
        """
        Bắt đầu cuộc hội thoại mới
        
        Args:
            conversation_id: ID cuộc hội thoại
        """
        self.conversation_history[conversation_id] = []
        self.logger.info(f"✅ Bắt đầu hội thoại: {conversation_id}")
    
    def chat(
        self,
        conversation_id: str,
        question: str,
        use_conversation_context: bool = True
    ) -> GenerationResult:
        """
        Chat với context của cuộc hội thoại
        
        Args:
            conversation_id: ID cuộc hội thoại
            question: Câu hỏi mới
            use_conversation_context: Có sử dụng context hội thoại không
            
        Returns:
            GenerationResult
        """
        if conversation_id not in self.conversation_history:
            self.start_conversation(conversation_id)
        
        # Enhance question with conversation context if needed
        enhanced_question = question
        if use_conversation_context and self.conversation_history[conversation_id]:
            enhanced_question = self._enhance_question_with_context(
                conversation_id, question
            )
        
        # Generate answer
        result = self.rag_chain.generate_answer(enhanced_question)
        
        # Store in conversation history
        self.conversation_history[conversation_id].append({
            'question': question,
            'enhanced_question': enhanced_question,
            'answer': result.answer,
            'timestamp': datetime.now().isoformat(),
            'confidence': result.confidence_score
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history[conversation_id]) > 10:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-10:]
        
        return result
    
    def _enhance_question_with_context(
        self,
        conversation_id: str,
        question: str
    ) -> str:
        """
        Enhance question với context từ hội thoại trước
        """
        history = self.conversation_history[conversation_id]
        if not history:
            return question
        
        # Get last 2 exchanges for context
        recent_history = history[-2:]
        context_parts = []
        
        for exchange in recent_history:
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"A: {exchange['answer'][:200]}...")
        
        if context_parts:
            enhanced = f"Dựa trên cuộc hội thoại trước:\n{chr(10).join(context_parts)}\n\nCâu hỏi mới: {question}"
            return enhanced
        
        return question
    
    def get_conversation_summary(self, conversation_id: str) -> str:
        """
        Lấy tóm tắt cuộc hội thoại
        
        Args:
            conversation_id: ID cuộc hội thoại
            
        Returns:
            Tóm tắt cuộc hội thoại
        """
        if conversation_id not in self.conversation_history:
            return "Không có lịch sử hội thoại"
        
        history = self.conversation_history[conversation_id]
        if not history:
            return "Cuộc hội thoại chưa bắt đầu"
        
        summary_parts = []
        for i, exchange in enumerate(history, 1):
            summary_parts.append(f"{i}. Q: {exchange['question']}")
            summary_parts.append(f"   A: {exchange['answer'][:100]}...")
        
        return "\n".join(summary_parts)

# Factory functions
def create_rag_chain(
    llm_manager: LLMManager,
    retrieval_engine: RetrievalEngine,
    context_builder: ContextBuilder
) -> RAGChain:
    """
    Factory function để tạo RAGChain
    """
    return RAGChain(llm_manager, retrieval_engine, context_builder)

def create_conversational_rag(rag_chain: RAGChain) -> ConversationalRAG:
    """
    Factory function để tạo ConversationalRAG
    """
    return ConversationalRAG(rag_chain)
