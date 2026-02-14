import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./contact_center.db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model paths
    INTENT_MODEL_PATH = "models/intent_classifier.pkl"
    VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
    
    # RAG settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 3