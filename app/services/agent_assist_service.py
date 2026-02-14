from typing import Dict
from .ml_service import MLService
from .rag_service import RAGService
from .llm_service import LLMService
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class AgentAssistService:
    def __init__(self):
        self.ml_service = MLService()
        self.rag_service = RAGService()
        self.llm_service = LLMService()
    
    def assist_agent(self, customer_message: str) -> Dict[str, str]:
        """Provide comprehensive assistance for customer service agents."""
        logger.info(f"Processing agent assist request for message: {customer_message[:50]}...")
        
        # Predict intent and sentiment
        intent, intent_confidence = self.ml_service.predict_intent(customer_message)
        sentiment, sentiment_confidence = self.ml_service.predict_sentiment(customer_message)
        
        # Get relevant knowledge base information
        rag_response = self.rag_service.query(customer_message)
        knowledge_context = rag_response['answer']
        
        # Generate suggested reply
        suggested_reply = self._generate_suggested_reply(
            customer_message, intent, sentiment, knowledge_context
        )
        
        return {
            'detected_intent': intent,
            'sentiment': sentiment,
            'suggested_reply': suggested_reply,
            'knowledge_context': knowledge_context,
            'intent_confidence': f"{intent_confidence:.2f}",
            'sentiment_confidence': f"{sentiment_confidence:.2f}"
        }
    
    def _generate_suggested_reply(self, customer_message: str, intent: str, 
                                sentiment: str, knowledge_context: str) -> str:
        """Generate a contextually appropriate suggested reply."""
        
        # Create context for LLM
        context = f"""
        Customer Intent: {intent}
        Customer Sentiment: {sentiment}
        Relevant Knowledge: {knowledge_context}
        """
        
        # Use LLM service to generate response
        suggested_reply = self.llm_service.generate_response(customer_message, context)
        
        # Add intent-specific enhancements
        enhanced_reply = self._enhance_reply_by_intent(suggested_reply, intent, sentiment)
        
        return enhanced_reply
    
    def _enhance_reply_by_intent(self, base_reply: str, intent: str, sentiment: str) -> str:
        """Enhance the reply based on detected intent and sentiment."""
        
        # Add empathy for negative sentiment
        if sentiment == 'negative':
            empathy_prefix = "I understand your frustration, and I'm here to help. "
            if not base_reply.lower().startswith(('i understand', 'i apologize', 'i\'m sorry')):
                base_reply = empathy_prefix + base_reply
        
        # Add intent-specific enhancements
        intent_enhancements = {
            'refund_request': "\n\nI'll make sure to process this as quickly as possible for you.",
            'shipping_inquiry': "\n\nI'll provide you with the most up-to-date tracking information.",
            'technical_support': "\n\nI'm here to walk you through this step by step until it's resolved.",
            'billing_question': "\n\nI'll review your account details to provide a clear explanation.",
            'product_inquiry': "\n\nI'll help you find exactly what you're looking for."
        }
        
        enhancement = intent_enhancements.get(intent, "")
        
        return base_reply + enhancement