import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
from typing import List, Dict, Any
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.get_database_url()
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        try:
            with psycopg2.connect(self.db_path) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS conversations (
                            id VARCHAR PRIMARY KEY,
                            customer_message TEXT NOT NULL,
                            agent_response TEXT NOT NULL,
                            full_conversation TEXT NOT NULL,
                            intent VARCHAR NOT NULL,
                            sentiment VARCHAR NOT NULL,
                            timestamp TIMESTAMP NOT NULL
                        )
                    """)
                    conn.commit()
                    logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def insert_conversations(self, conversations: List[Dict[str, Any]]):
        """Insert multiple conversation records."""
        try:
            with psycopg2.connect(self.db_path) as conn:
                with conn.cursor() as cursor:
                    for conv in conversations:
                        cursor.execute("""
                            INSERT INTO conversations (id, customer_message, agent_response, 
                                                     full_conversation, intent, sentiment, timestamp)
                            VALUES (%(id)s, %(customer_message)s, %(agent_response)s, 
                                   %(full_conversation)s, %(intent)s, %(sentiment)s, %(timestamp)s)
                            ON CONFLICT (id) DO NOTHING
                        """, conv)
                    conn.commit()
                    logger.info(f"Inserted {len(conversations)} conversations")
        except Exception as e:
            logger.error(f"Failed to insert conversations: {e}")
            raise
    
    def get_all_conversations(self) -> pd.DataFrame:
        """Retrieve all conversations as DataFrame."""
        try:
            with psycopg2.connect(self.db_path) as conn:
                return pd.read_sql_query("SELECT * FROM conversations", conn)
        except Exception as e:
            logger.error(f"Failed to retrieve conversations: {e}")
            return pd.DataFrame()
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get aggregated analytics data."""
        try:
            with psycopg2.connect(self.db_path) as conn:
                # Total conversations
                total_df = pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn)
                total = total_df.iloc[0]['count'] if not total_df.empty else 0
                
                # Intent distribution
                intent_df = pd.read_sql_query(
                    "SELECT intent, COUNT(*) as count FROM conversations GROUP BY intent", conn
                )
                intent_dist = intent_df.set_index('intent')['count'].to_dict() if not intent_df.empty else {}
                
                # Sentiment distribution
                sentiment_df = pd.read_sql_query(
                    "SELECT sentiment, COUNT(*) as count FROM conversations GROUP BY sentiment", conn
                )
                sentiment_dist = sentiment_df.set_index('sentiment')['count'].to_dict() if not sentiment_df.empty else {}
                
                # Daily volume
                daily_df = pd.read_sql_query(
                    "SELECT DATE(timestamp) as date, COUNT(*) as count FROM conversations GROUP BY DATE(timestamp)", conn
                )
                daily_volume = daily_df.set_index('date')['count'].to_dict() if not daily_df.empty else {}
                
                return {
                    'total_conversations': total,
                    'intent_distribution': intent_dist,
                    'sentiment_distribution': sentiment_dist,
                    'daily_volume': daily_volume
                }
        except Exception as e:
            logger.error(f"Failed to get analytics data: {e}")
            return {
                'total_conversations': 0,
                'intent_distribution': {},
                'sentiment_distribution': {},
                'daily_volume': {}
            }