import openai
from typing import Dict, Any, Optional
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class LLMService:
    def __init__(self):
        self.client = None
        self.has_openai_key = bool(Config.OPENAI_API_KEY)
        
        if self.has_openai_key:
            try:
                self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Using mock responses.")
                self.has_openai_key = False
                self.client = None
        else:
            logger.info("No OpenAI API key found. Using mock responses.")
    
    def summarize_conversation(self, conversation: str) -> Dict[str, str]:
        """Summarize a customer support conversation."""
        if self.has_openai_key and self.client:
            return self._summarize_with_openai(conversation)
        else:
            return self._mock_summarization(conversation)
    
    def _summarize_with_openai(self, conversation: str) -> Dict[str, str]:
        """Summarize using OpenAI API."""
        try:
            prompt = f"""
            Analyze this customer support conversation and provide:
            1. Issue Summary: Brief description of the customer's problem
            2. Resolution: How the issue was resolved or next steps
            3. Customer Sentiment: Overall customer sentiment (positive/negative/neutral)
            
            Conversation:
            {conversation}
            
            Please format your response as JSON with keys: issue_summary, resolution, customer_sentiment
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a customer service analyst. Provide concise, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            # Parse the response (simplified - in production, use proper JSON parsing)
            content = response.choices[0].message.content
            
            # Extract information (basic parsing - could be improved)
            lines = content.split('\n')
            result = {
                'issue_summary': 'Customer inquiry processed',
                'resolution': 'Issue addressed by agent',
                'customer_sentiment': 'neutral'
            }
            
            for line in lines:
                if 'issue_summary' in line.lower():
                    result['issue_summary'] = line.split(':', 1)[-1].strip().strip('"')
                elif 'resolution' in line.lower():
                    result['resolution'] = line.split(':', 1)[-1].strip().strip('"')
                elif 'customer_sentiment' in line.lower():
                    result['customer_sentiment'] = line.split(':', 1)[-1].strip().strip('"')
            
            return result
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._mock_summarization(conversation)
    
    def _mock_summarization(self, conversation: str) -> Dict[str, str]:
        """Provide mock summarization when OpenAI is not available."""
        # Simple keyword-based analysis for demo purposes
        conv_lower = conversation.lower()
        
        # Determine issue type
        if any(word in conv_lower for word in ['refund', 'return', 'money back']):
            issue = "Customer requesting refund for product"
            resolution = "Refund processed successfully"
        elif any(word in conv_lower for word in ['shipping', 'delivery', 'package']):
            issue = "Customer inquiring about order delivery status"
            resolution = "Tracking information provided to customer"
        elif any(word in conv_lower for word in ['technical', 'app', 'website', 'login']):
            issue = "Customer experiencing technical difficulties"
            resolution = "Technical support steps provided"
        elif any(word in conv_lower for word in ['billing', 'charge', 'invoice']):
            issue = "Customer has billing-related question"
            resolution = "Billing explanation provided"
        else:
            issue = "General customer inquiry"
            resolution = "Customer inquiry addressed"
        
        # Determine sentiment
        if any(word in conv_lower for word in ['thank', 'great', 'perfect', 'excellent']):
            sentiment = "positive"
        elif any(word in conv_lower for word in ['frustrated', 'angry', 'terrible', 'awful']):
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            'issue_summary': issue,
            'resolution': resolution,
            'customer_sentiment': sentiment
        }
    
    def generate_response(self, customer_message: str, context: str = "") -> str:
        """Generate a suggested response for the agent."""
        if self.has_openai_key and self.client:
            return self._generate_with_openai(customer_message, context)
        else:
            return self._mock_response(customer_message)
    
    def _generate_with_openai(self, customer_message: str, context: str = "") -> str:
        """Generate response using OpenAI."""
        try:
            prompt = f"""
            You are a helpful customer service agent. Generate a professional, empathetic response to this customer message.
            
            Customer Message: {customer_message}
            
            Context: {context}
            
            Provide a helpful, professional response that addresses the customer's concern.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional customer service representative."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._mock_response(customer_message)
    
    def _mock_response(self, customer_message: str) -> str:
        """Generate mock response when OpenAI is not available."""
        msg_lower = customer_message.lower()
        
        if any(word in msg_lower for word in ['refund', 'return']):
            return "I understand you'd like to process a return. I can help you with that right away. Let me look up your order details."
        elif any(word in msg_lower for word in ['shipping', 'delivery', 'where']):
            return "I can help you track your order. Let me check the shipping status for you and provide an update."
        elif any(word in msg_lower for word in ['technical', 'app', 'website', 'login']):
            return "I'm sorry you're experiencing technical difficulties. Let me help you troubleshoot this issue step by step."
        elif any(word in msg_lower for word in ['billing', 'charge']):
            return "I can help clarify any billing questions you have. Let me review your account and explain the charges."
        else:
            return "Thank you for contacting us. I'm here to help with your inquiry. Could you please provide more details so I can assist you better?"