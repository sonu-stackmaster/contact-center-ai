import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/contact_center.db")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Model paths
    INTENT_MODEL_PATH = "models/intent_classifier.pkl"
    VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
    
    # RAG settings (optimized for low-spec systems)
    CHUNK_SIZE = 300  # Smaller chunks for better performance
    CHUNK_OVERLAP = 30
    TOP_K_RESULTS = 3
    
    # CPU-optimized settings
    USE_LIGHTWEIGHT_MODELS = True
    MAX_SEQUENCE_LENGTH = 256  # Shorter sequences for faster processing
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight sentence transformer
    
    # Memory optimization
    BATCH_SIZE = 16  # Smaller batches for limited RAM
    CACHE_SIZE = 100  # Limit cache size
    
    # Performance settings
    NUM_THREADS = 2  # Conservative thread count for i5
    ENABLE_CACHING = True