from fastapi import APIRouter, HTTPException
from ..models.schemas import (
    IntentPredictionRequest, IntentPredictionResponse,
    SentimentPredictionRequest, SentimentPredictionResponse,
    SummarizationRequest, SummarizationResponse,
    RAGQueryRequest, RAGQueryResponse,
    AgentAssistRequest, AgentAssistResponse,
    AnalyticsSummary
)
from ..services.ml_service import MLService
from ..services.llm_service import LLMService
from ..services.rag_service import RAGService
from ..services.agent_assist_service import AgentAssistService
from ..data.database import DatabaseManager
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter()

# Initialize services
ml_service = MLService()
llm_service = LLMService()
rag_service = RAGService()
agent_assist_service = AgentAssistService()
db_manager = DatabaseManager()

@router.post("/predict/intent", response_model=IntentPredictionResponse)
async def predict_intent(request: IntentPredictionRequest):
    """Predict intent for a customer message."""
    try:
        intent, confidence = ml_service.predict_intent(request.message)
        return IntentPredictionResponse(intent=intent, confidence=confidence)
    except Exception as e:
        logger.error(f"Intent prediction error: {e}")
        raise HTTPException(status_code=500, detail="Intent prediction failed")

@router.post("/predict/sentiment", response_model=SentimentPredictionResponse)
async def predict_sentiment(request: SentimentPredictionRequest):
    """Predict sentiment for a customer message."""
    try:
        sentiment, confidence = ml_service.predict_sentiment(request.message)
        return SentimentPredictionResponse(sentiment=sentiment, confidence=confidence)
    except Exception as e:
        logger.error(f"Sentiment prediction error: {e}")
        raise HTTPException(status_code=500, detail="Sentiment prediction failed")

@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_conversation(request: SummarizationRequest):
    """Summarize a customer support conversation."""
    try:
        summary = llm_service.summarize_conversation(request.conversation)
        return SummarizationResponse(
            issue_summary=summary['issue_summary'],
            resolution=summary['resolution'],
            customer_sentiment=summary['customer_sentiment']
        )
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail="Conversation summarization failed")

@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """Query the knowledge base using RAG."""
    try:
        result = rag_service.query(request.question)
        return RAGQueryResponse(
            answer=result['answer'],
            sources=result['sources']
        )
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail="RAG query failed")

@router.post("/agent-assist", response_model=AgentAssistResponse)
async def agent_assist(request: AgentAssistRequest):
    """Provide comprehensive agent assistance."""
    try:
        result = agent_assist_service.assist_agent(request.customer_message)
        return AgentAssistResponse(
            detected_intent=result['detected_intent'],
            sentiment=result['sentiment'],
            suggested_reply=result['suggested_reply']
        )
    except Exception as e:
        logger.error(f"Agent assist error: {e}")
        raise HTTPException(status_code=500, detail="Agent assistance failed")

@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary():
    """Get analytics summary from the database."""
    try:
        data = db_manager.get_analytics_data()
        return AnalyticsSummary(
            total_conversations=data['total_conversations'],
            intent_distribution=data['intent_distribution'],
            sentiment_distribution=data['sentiment_distribution'],
            daily_volume=data['daily_volume']
        )
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Analytics retrieval failed")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Contact Center AI API is running"}