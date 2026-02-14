from typing import Dict, List
from ..rag.knowledge_base import LightweightKnowledgeBase
from ..services.llm_service import LLMService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGService:
    def __init__(self):
        self.knowledge_base = LightweightKnowledgeBase()
        self.llm_service = LLMService()
    
    def query(self, question: str) -> Dict[str, any]:
        """Process a RAG query and return grounded response."""
        logger.info(f"Processing RAG query: {question[:50]}...")
        
        # Retrieve relevant context
        context = self.knowledge_base.get_context_for_query(question)
        
        # Generate response using LLM with context
        if self.llm_service.has_openai_key:
            response = self._generate_rag_response_openai(question, context)
        else:
            response = self._generate_rag_response_mock(question, context)
        
        # Get source documents for transparency
        search_results = self.knowledge_base.search(question)
        sources = [doc['metadata'].get('title', 'Unknown') for doc, _ in search_results]
        
        return {
            'answer': response,
            'sources': list(set(sources))  # Remove duplicates
        }
    
    def _generate_rag_response_openai(self, question: str, context: str) -> str:
        """Generate RAG response using OpenAI."""
        try:
            prompt = f"""
            You are a helpful customer service assistant. Answer the customer's question using the provided context.
            If the context doesn't contain relevant information, say so politely and offer to help in other ways.
            
            Context:
            {context}
            
            Customer Question: {question}
            
            Please provide a helpful, accurate response based on the context above.
            """
            
            response = self.llm_service.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable customer service representative. Use only the provided context to answer questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error in RAG: {e}")
            return self._generate_rag_response_mock(question, context)
    
    def _generate_rag_response_mock(self, question: str, context: str) -> str:
        """Generate mock RAG response when OpenAI is not available."""
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Simple keyword matching for demo
        if 'refund' in question_lower or 'return' in question_lower:
            if 'refund policy' in context_lower:
                return "Based on our refund policy, you can return items within 30 days of purchase for a full refund. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days after we receive the returned item."
            
        elif 'shipping' in question_lower or 'delivery' in question_lower:
            if 'shipping' in context_lower:
                return "We offer several shipping options: Standard shipping (5-7 business days) is free on orders over $50, Express shipping (2-3 business days) costs $9.99, and Overnight shipping (1 business day) costs $19.99. Orders are processed within 1-2 business days."
        
        elif 'login' in question_lower or 'account' in question_lower:
            if 'login' in context_lower or 'account' in context_lower:
                return "If you're having trouble logging in, try these steps: 1) Check your email address is correct, 2) Reset your password using 'Forgot Password', 3) Clear browser cache and cookies, 4) Try a different browser. If issues persist, contact our technical support team."
        
        elif 'billing' in question_lower or 'payment' in question_lower:
            if 'billing' in context_lower or 'payment' in context_lower:
                return "We accept credit cards (Visa, MasterCard, American Express), PayPal, Apple Pay, and Google Pay. If you see an unexpected charge, check your order history or contact billing support with your order number."
        
        elif 'compatible' in question_lower or 'product' in question_lower:
            if 'compatibility' in context_lower or 'product' in context_lower:
                return "Before purchasing, please check product compatibility by reviewing system requirements and product specifications. If you're unsure, contact our product specialists. We offer a 30-day compatibility guarantee."
        
        # Fallback response
        if context and len(context.strip()) > 50:
            return f"Based on our knowledge base, here's what I found relevant to your question: {context[:200]}... For more specific assistance, please contact our support team."
        else:
            return "I don't have specific information about that in our knowledge base. Please contact our customer support team for personalized assistance, or try rephrasing your question."