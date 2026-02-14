import os
import faiss
import numpy as np
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from ..utils.config import Config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class KnowledgeBase:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.index = None
        self.documents = []
        self.doc_texts = []
        
        # Initialize with sample knowledge base
        self._create_sample_knowledge_base()
    
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
            # Split the content into chunks
            chunks = self.text_splitter.split_text(doc["content"])
            for chunk in chunks:
                documents.append(Document(
                    page_content=chunk,
                    metadata={"title": doc["title"]}
                ))
        
        self.add_documents(documents)
        logger.info(f"Knowledge base initialized with {len(documents)} document chunks")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the knowledge base."""
        self.documents.extend(documents)
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        self.doc_texts.extend(texts)
        
        # Create embeddings
        embeddings = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create or update FAISS index
        if self.index is None:
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def search(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """Search for relevant documents."""
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.embeddings.embed_query(query)
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
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a query."""
        results = self.search(query)
        
        if not results:
            return "No relevant information found in knowledge base."
        
        context_parts = []
        for doc, score in results:
            context_parts.append(f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}")
        
        return "\n\n".join(context_parts)