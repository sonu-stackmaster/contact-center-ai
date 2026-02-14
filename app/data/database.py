import sqlite3
import pandas as pd
from typing import List, Dict, Any
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DATABASE_URL.replace("sqlite:///", "")
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    customer_message TEXT NOT NULL,
                    agent_response TEXT NOT NULL,
                    full_conversation TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """)
            conn.commit()
            logger.info("Database initialized successfully")
    
    def insert_conversations(self, conversations: List[Dict[str, Any]]):
        """Insert multiple conversation records."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.DataFrame(conversations)
            df.to_sql('conversations', conn, if_exists='append', index=False)
            logger.info(f"Inserted {len(conversations)} conversations")
    
    def get_all_conversations(self) -> pd.DataFrame:
        """Retrieve all conversations as DataFrame."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM conversations", conn)
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get aggregated analytics data."""
        with sqlite3.connect(self.db_path) as conn:
            # Total conversations
            total = pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn).iloc[0]['count']
            
            # Intent distribution
            intent_dist = pd.read_sql_query(
                "SELECT intent, COUNT(*) as count FROM conversations GROUP BY intent", conn
            ).set_index('intent')['count'].to_dict()
            
            # Sentiment distribution
            sentiment_dist = pd.read_sql_query(
                "SELECT sentiment, COUNT(*) as count FROM conversations GROUP BY sentiment", conn
            ).set_index('sentiment')['count'].to_dict()
            
            # Daily volume
            daily_volume = pd.read_sql_query(
                "SELECT DATE(timestamp) as date, COUNT(*) as count FROM conversations GROUP BY DATE(timestamp)", conn
            ).set_index('date')['count'].to_dict()
            
            return {
                'total_conversations': total,
                'intent_distribution': intent_dist,
                'sentiment_distribution': sentiment_dist,
                'daily_volume': daily_volume
            }