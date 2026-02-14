import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.database import DatabaseManager

# Sample data templates
INTENTS = {
    'refund_request': [
        "I want to return this product and get my money back",
        "Can I get a refund for my order?",
        "This item doesn't work, I need a refund",
        "How do I return this and get refunded?"
    ],
    'shipping_inquiry': [
        "Where is my order? It's been 2 weeks",
        "My package hasn't arrived yet",
        "Can you track my shipment?",
        "When will my order be delivered?"
    ],
    'technical_support': [
        "The app keeps crashing on my phone",
        "I can't log into my account",
        "The website won't load properly",
        "I'm having trouble with the checkout process"
    ],
    'billing_question': [
        "Why was I charged twice?",
        "I don't understand this charge on my bill",
        "Can you explain my invoice?",
        "There's an error in my billing"
    ],
    'product_inquiry': [
        "Do you have this item in stock?",
        "What are the specifications of this product?",
        "Is this compatible with my device?",
        "Can you recommend something similar?"
    ]
}

AGENT_RESPONSES = {
    'refund_request': [
        "I understand you'd like a refund. Let me process that for you right away.",
        "I can help you with the return process. You'll receive a full refund within 5-7 business days.",
        "No problem! I've initiated your refund request. You should see the credit in 3-5 days."
    ],
    'shipping_inquiry': [
        "Let me track that package for you. I can see it's currently in transit and should arrive tomorrow.",
        "Your order is on its way! The tracking shows it will be delivered by end of day today.",
        "I apologize for the delay. Your package was held up in customs but is now out for delivery."
    ],
    'technical_support': [
        "I can help you troubleshoot this issue. Let's try clearing your cache first.",
        "This sounds like a known issue. I'll send you the steps to resolve it via email.",
        "Let me walk you through the solution step by step."
    ],
    'billing_question': [
        "I can see the charge you're referring to. Let me explain what this covers.",
        "That charge is for the premium service upgrade you selected. Would you like me to review your account?",
        "I understand the confusion. This is actually a legitimate charge for your recent purchase."
    ],
    'product_inquiry': [
        "Yes, we have that item in stock! I can place the order for you right now.",
        "Let me check our inventory... Yes, it's available and compatible with your device.",
        "Based on your needs, I'd recommend this alternative product which has better reviews."
    ]
}

SENTIMENTS = ['positive', 'negative', 'neutral']

def generate_conversation(intent: str, customer_msg: str, agent_resp: str) -> str:
    """Generate a full conversation from customer message and agent response."""
    return f"Customer: {customer_msg}\nAgent: {agent_resp}"

def generate_synthetic_data(num_records: int = 1000) -> List[Dict[str, Any]]:
    """Generate synthetic customer support conversations."""
    conversations = []
    
    for _ in range(num_records):
        # Random intent
        intent = random.choice(list(INTENTS.keys()))
        
        # Random messages for this intent
        customer_message = random.choice(INTENTS[intent])
        agent_response = random.choice(AGENT_RESPONSES[intent])
        
        # Generate full conversation
        full_conversation = generate_conversation(intent, customer_message, agent_response)
        
        # Random sentiment (weighted towards positive for good customer service)
        sentiment = random.choices(
            SENTIMENTS, 
            weights=[0.6, 0.2, 0.2]  # 60% positive, 20% negative, 20% neutral
        )[0]
        
        # Random timestamp within last 30 days
        days_ago = random.randint(0, 30)
        timestamp = datetime.now() - timedelta(days=days_ago)
        
        conversation = {
            'id': str(uuid.uuid4()),
            'customer_message': customer_message,
            'agent_response': agent_response,
            'full_conversation': full_conversation,
            'intent': intent,
            'sentiment': sentiment,
            'timestamp': timestamp
        }
        
        conversations.append(conversation)
    
    return conversations

def main():
    """Generate and save synthetic data to database."""
    print("Generating synthetic customer support data...")
    
    # Generate data
    conversations = generate_synthetic_data(1000)
    
    # Save to database
    db = DatabaseManager()
    db.insert_conversations(conversations)
    
    print(f"Successfully generated and saved {len(conversations)} conversations!")
    
    # Print some stats
    intents = [c['intent'] for c in conversations]
    sentiments = [c['sentiment'] for c in conversations]
    
    print("\nIntent distribution:")
    for intent in set(intents):
        count = intents.count(intent)
        print(f"  {intent}: {count} ({count/len(conversations)*100:.1f}%)")
    
    print("\nSentiment distribution:")
    for sentiment in set(sentiments):
        count = sentiments.count(sentiment)
        print(f"  {sentiment}: {count} ({count/len(conversations)*100:.1f}%)")

if __name__ == "__main__":
    main()