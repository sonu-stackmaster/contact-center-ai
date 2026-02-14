import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import pipeline
import pandas as pd
from typing import Tuple, Optional
from ..utils.config import Config
from ..utils.logger import setup_logger
from ..data.database import DatabaseManager

logger = setup_logger(__name__)

class IntentClassifier:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        
    def train(self, texts: list, labels: list) -> dict:
        """Train the intent classification model."""
        logger.info("Training intent classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_vec)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        logger.info(f"Intent classifier trained with accuracy: {report['accuracy']:.3f}")
        
        return report
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict intent for a given text."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        text_vec = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vec)[0]
        probabilities = self.model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        return prediction, confidence
    
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

class SentimentAnalyzer:
    def __init__(self):
        self.pipeline = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize the sentiment analysis pipeline."""
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            logger.info("Sentiment analyzer initialized with Hugging Face model")
        except Exception as e:
            logger.warning(f"Failed to load HF model: {e}. Using rule-based fallback.")
            self.pipeline = None
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict sentiment for a given text."""
        if self.pipeline:
            return self._predict_with_model(text)
        else:
            return self._predict_rule_based(text)
    
    def _predict_with_model(self, text: str) -> Tuple[str, float]:
        """Predict using Hugging Face model."""
        results = self.pipeline(text)[0]
        
        # Convert labels to our format
        label_mapping = {
            'LABEL_0': 'negative',
            'LABEL_1': 'neutral', 
            'LABEL_2': 'positive'
        }
        
        best_result = max(results, key=lambda x: x['score'])
        sentiment = label_mapping.get(best_result['label'], best_result['label'].lower())
        confidence = best_result['score']
        
        return sentiment, confidence
    
    def _predict_rule_based(self, text: str) -> Tuple[str, float]:
        """Simple rule-based sentiment analysis as fallback."""
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'happy', 'satisfied']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointed', 'angry', 'frustrated']
        
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
        self.sentiment_analyzer = SentimentAnalyzer()
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