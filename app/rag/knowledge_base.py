import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from diskcache import Cache
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class LightweightKnowledgeBase:
    def __init__(self):
        self.embeddings = None
        self.index = None
        self.documents = []
        self.doc_texts = []
        
        # Initialize caching
        self.cache = Cache('cache/embedding_cache', size_limit=Config.CACHE_SIZE * 1024 * 1024) if Config.ENABLE_CACHING else None
        
        # Try to initialize embeddings, fall back to TF-IDF if needed
        self._initialize_embeddings()
        
        # Initialize with sample knowledge base
        self._create_sample_knowledge_base()
    
    def _initialize_embeddings(self):
        """Initialize embeddings with fallback options."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embeddings = SentenceTransformer(
                Config.EMBEDDING_MODEL,
                device='cpu'  # Force CPU usage
            )
            # Set max sequence length for performance
            self.embeddings.max_seq_length = Config.MAX_SEQUENCE_LENGTH
            self.embedding_type = 'sentence_transformer'
            logger.info("Sentence Transformer embeddings initialized")
        except ImportError as e:
            logger.warning(f"Sentence Transformers not available: {e}. Using TF-IDF fallback.")
            self._initialize_tfidf_fallback()
        except Exception as e:
            logger.warning(f"Failed to load Sentence Transformer: {e}. Using TF-IDF fallback.")
            self._initialize_tfidf_fallback()
    
    def _initialize_tfidf_fallback(self):
        """Initialize TF-IDF as fallback for embeddings."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.embeddings = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            self.embedding_type = 'tfidf'
            logger.info("TF-IDF embeddings initialized as fallback")
        except ImportError:
            logger.error("Neither Sentence Transformers nor scikit-learn available!")
            self.embeddings = None
            self.embedding_type = 'none'
    
    def _create_sample_knowledge_base(self):
        """Create a sample knowledge base with FAQ content."""
        sample_docs = [
            {
                "title": "Refund Policy",
                "content": """
                Our refund policy allows customers to return items within 30 days of purchase for a full refund.
                Items must be in original condition with tags attached.
                Digital products are non-refundable unless there's a technical issue.
                Refunds are processed within 5-7 business days after we receive the returned item.
                Shipping costs are non-refundable unless the return is due to our error.
                To initiate a return, customers can use our online return portal or contact customer service.
                """
            },
            {
                "title": "Shipping Information",
                "content": """
                We offer several shipping options:
                - Standard shipping (5-7 business days): Free on orders over $50
                - Express shipping (2-3 business days): $9.99
                - Overnight shipping (1 business day): $19.99
                
                Orders are processed within 1-2 business days.
                Tracking information is provided via email once the order ships.
                We ship to all 50 US states and internationally to select countries.
                Delivery delays may occur during peak seasons or due to weather conditions.
                """
            },
            {
                "title": "Account and Login Issues",
                "content": """
                If you're having trouble logging into your account:
                1. Check that you're using the correct email address
                2. Try resetting your password using the 'Forgot Password' link
                3. Clear your browser cache and cookies
                4. Disable browser extensions that might interfere
                5. Try using a different browser or incognito mode
                
                If you still can't access your account, contact our technical support team.
                We can help verify your identity and restore access to your account.
                """
            },
            {
                "title": "Product Information and Compatibility",
                "content": """
                Before purchasing, please check product compatibility:
                - Review system requirements for software products
                - Check device compatibility for accessories
                - Read product descriptions carefully for specifications
                
                Our products come with detailed specifications and compatibility information.
                If you're unsure about compatibility, contact our product specialists.
                We offer a 30-day compatibility guarantee - if a product doesn't work with your system,
                you can return it for a full refund.
                """
            },
            {
                "title": "Billing and Payment Issues",
                "content": """
                We accept the following payment methods:
                - Credit cards (Visa, MasterCard, American Express)
                - PayPal
                - Apple Pay and Google Pay
                - Bank transfers for large orders
                
                If you see an unexpected charge:
                1. Check your order history in your account
                2. Review your email for order confirmations
                3. Contact billing support with your order number
                
                We never store your full credit card information for security.
                All transactions are processed securely using industry-standard encryption.
                """
            }
        ]
        
        # Convert to documents and create embeddings
        documents = []
        for doc in sample_docs:
            # Split content into smaller chunks for better performance
            chunks = self._split_text(doc["content"], Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
            for chunk in chunks:
                documents.append({
                    'content': chunk,
                    'metadata': {"title": doc["title"]}
                })
        
        self.add_documents(documents)
        logger.info(f"Knowledge base initialized with {len(documents)} document chunks")
    
    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple text splitting for CPU efficiency."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the knowledge base."""
        if not self.embeddings:
            logger.warning("No embeddings available, using keyword matching only")
            self.documents.extend(documents)
            self.doc_texts.extend([doc['content'] for doc in documents])
            return
        
        texts = [doc['content'] for doc in documents]
        
        # Create embeddings based on type
        if self.embedding_type == 'sentence_transformer':
            embeddings = self._create_sentence_embeddings(texts)
        elif self.embedding_type == 'tfidf':
            embeddings = self._create_tfidf_embeddings(texts)
        else:
            logger.warning("No embedding method available")
            self.documents.extend(documents)
            self.doc_texts.extend(texts)
            return
        
        if embeddings is not None:
            self._add_to_index(embeddings)
        
        # Store documents and texts
        self.documents.extend(documents)
        self.doc_texts.extend(texts)
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def _create_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using sentence transformers."""
        embeddings = []
        for text in texts:
            if self.cache and text in self.cache:
                embedding = self.cache[text]
            else:
                embedding = self.embeddings.encode([text], show_progress_bar=False)[0]
                if self.cache:
                    self.cache[text] = embedding
            embeddings.append(embedding)
        
        return np.array(embeddings).astype('float32')
    
    def _create_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using TF-IDF."""
        if not hasattr(self.embeddings, 'vocabulary_'):
            # First time - fit the vectorizer
            all_existing_texts = self.doc_texts + texts
            embeddings_matrix = self.embeddings.fit_transform(all_existing_texts)
            # Return only the new embeddings
            return embeddings_matrix[-len(texts):].toarray().astype('float32')
        else:
            # Transform new texts
            embeddings_matrix = self.embeddings.transform(texts)
            return embeddings_matrix.toarray().astype('float32')
    
    def _add_to_index(self, embeddings_array: np.ndarray):
        """Add embeddings to FAISS index."""
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[Dict, float]]:
        """Search for relevant documents."""
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        if self.index is None or len(self.documents) == 0:
            return self._keyword_search(query, top_k)
        
        # Create query embedding
        if self.embedding_type == 'sentence_transformer':
            query_embedding = self._get_sentence_query_embedding(query)
        elif self.embedding_type == 'tfidf':
            query_embedding = self._get_tfidf_query_embedding(query)
        else:
            return self._keyword_search(query, top_k)
        
        if query_embedding is None:
            return self._keyword_search(query, top_k)
        
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.index.search(query_array, min(top_k, len(self.documents)))
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):  # Valid index
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def _get_sentence_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using sentence transformer."""
        if self.cache and query in self.cache:
            return self.cache[query]
        
        embedding = self.embeddings.encode([query], show_progress_bar=False)[0]
        if self.cache:
            self.cache[query] = embedding
        return embedding
    
    def _get_tfidf_query_embedding(self, query: str) -> np.ndarray:
        """Get query embedding using TF-IDF."""
        if not hasattr(self.embeddings, 'vocabulary_'):
            return None
        
        query_vec = self.embeddings.transform([query])
        return query_vec.toarray()[0]
    
    def _keyword_search(self, query: str, top_k: int) -> List[Tuple[Dict, float]]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for doc in self.documents:
            content_lower = doc['content'].lower()
            content_words = set(content_lower.split())
            
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                score = overlap / len(query_words)
                results.append((doc, score))
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a query."""
        results = self.search(query)
        
        if not results:
            return "No relevant information found in knowledge base."
        
        context_parts = []
        for doc, score in results:
            context_parts.append(f"Source: {doc['metadata'].get('title', 'Unknown')}\n{doc['content']}")
        
        return "\n\n".join(context_parts)

# Alias for backward compatibility
KnowledgeBase = LightweightKnowledgeBase