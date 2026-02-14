import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from typing import Tuple, Optional
from textblob import TextBlob
import nltk
from diskcache import Cache
from ..utils.config import Config
from ..utils.logger import setup_logger
from ..data.database import DatabaseManager

logger = setup_logger(__name__)

# Download required NLTK data (lightweight)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

class IntentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.cache = Cache('cache/intent_cache', size_limit=Config.CACHE_SIZE * 1024 * 1024) if Config.ENABLE_CACHING else None
        
    def train(self, texts: list, labels: list) -> dict:
        """Train the intent classification model with CPU optimization."""
        logger.info("Training lightweight intent classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text with reduced features for performance
        self.vectorizer = TfidfVectorizer(
            max_features=2000,  # Reduced from 5000 for performance
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=2,  # Ignore rare terms
            max_df=0.95  # Ignore too common terms
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train lightweight model
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=500,  # Reduced iterations
            solver='lbfgs',  # Better for multiclass classification
            C=1.0  # Default regularization
        )
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        logger.info(f"Intent classifier trained with accuracy: {report['accuracy']:.3f}")
        
        return report
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a given text with caching."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Check cache first
        if self.cache and text in self.cache:
            return self.cache[text]
        
        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        result = (prediction, confidence)
        
        # Cache result
        if self.cache:
            self.cache[text] = result
        
        return result
    
    def save_model(self, model_path: str, vectorizer_path: str):
        """Save trained model and vectorizer."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info("Model saved successfully")
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """Load trained model and vectorizer."""
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.is_trained = True
            logger.info("Model loaded successfully")
            return True
        return False

class LightweightSentimentAnalyzer:
    """CPU-optimized sentiment analyzer using TextBlob and NLTK."""
    
    def __init__(self):
        self.cache = Cache('cache/sentiment_cache', size_limit=Config.CACHE_SIZE * 1024 * 1024) if Config.ENABLE_CACHING else None
        logger.info("Lightweight sentiment analyzer initialized")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment using TextBlob (CPU-only, fast)."""
        # Check cache first
        if self.cache and text in self.cache:
            return self.cache[text]
        
        try:
            # Use TextBlob for sentiment analysis (lightweight, no GPU needed)
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Convert polarity to sentiment label
            if polarity > 0.1:
                sentiment = 'positive'
                confidence = min(0.9, 0.6 + abs(polarity) * 0.3)
            elif polarity < -0.1:
                sentiment = 'negative'
                confidence = min(0.9, 0.6 + abs(polarity) * 0.3)
            else:
                sentiment = 'neutral'
                confidence = 0.7
            
            result = (sentiment, confidence)
            
            # Cache result
            if self.cache:
                self.cache[text] = result
            
            return result
            
        except Exception as e:
            logger.warning(f"TextBlob sentiment analysis failed: {e}. Using rule-based fallback.")
            return self._predict_rule_based(text)
    
    def _predict_rule_based(self, text: str) -> Tuple[str, float]:
        """Simple rule-based sentiment analysis as fallback."""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'happy', 'satisfied', 'wonderful', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointed', 'angry', 'frustrated', 'worst', 'useless']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive', 0.7
        elif neg_count > pos_count:
            return 'negative', 0.7
        else:
            return 'neutral', 0.6

class MLService:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.sentiment_analyzer = LightweightSentimentAnalyzer()
        self._load_or_train_models()
    
    def _load_or_train_models(self):
        """Load existing models or train new ones."""
        model_loaded = self.intent_classifier.load_model(
            Config.INTENT_MODEL_PATH, 
            Config.VECTORIZER_PATH
        )
        
        if not model_loaded:
            logger.info("No existing model found. Training new model...")
            self._train_intent_model()
    
    def _train_intent_model(self):
        """Train intent classification model using database data."""
        try:
            db = DatabaseManager()
            df = db.get_all_conversations()
            
            if len(df) == 0:
                logger.warning("No training data available. Please generate data first.")
                return
            
            texts = df['customer_message'].tolist()
            labels = df['intent'].tolist()
            
            report = self.intent_classifier.train(texts, labels)
            self.intent_classifier.save_model(Config.INTENT_MODEL_PATH, Config.VECTORIZER_PATH)
            
            logger.info("Intent model training completed")
            
        except Exception as e:
            logger.error(f"Failed to train intent model: {e}")
    
    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent for given text."""
        return self.intent_classifier.predict(text)
    
    def predict_sentiment(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for given text."""
        return self.sentiment_analyzer.predict(text)