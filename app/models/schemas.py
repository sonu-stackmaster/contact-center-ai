from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class ConversationRecord(BaseModel):
    id: str
    customer_message: str
    agent_response: str
    full_conversation: str
    intent: str
    sentiment: str
    timestamp: datetime

class IntentPredictionRequest(BaseModel):
    message: str

class IntentPredictionResponse(BaseModel):
    intent: str
    confidence: float

class SentimentPredictionRequest(BaseModel):
    message: str

class SentimentPredictionResponse(BaseModel):
    sentiment: str
    confidence: float

class SummarizationRequest(BaseModel):
    conversation: str

class SummarizationResponse(BaseModel):
    issue_summary: str
    resolution: str
    customer_sentiment: str

class RAGQueryRequest(BaseModel):
    question: str

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[str]

class AgentAssistRequest(BaseModel):
    customer_message: str

class AgentAssistResponse(BaseModel):
    detected_intent: str
    sentiment: str
    suggested_reply: str

class AnalyticsSummary(BaseModel):
    total_conversations: int
    intent_distribution: dict
    sentiment_distribution: dict
    daily_volume: dict