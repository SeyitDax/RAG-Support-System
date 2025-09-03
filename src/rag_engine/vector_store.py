"""
Production Vector Store using Pinecone

Scalable vector database implementation for storing and retrieving document embeddings
with metadata for customer support knowledge base.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import pinecone
from pinecone import Pinecone, ServerlessSpec
import structlog
from .config import config

logger = structlog.get_logger(__name__)


class PineconeVectorStore:
    """Production vector store using Pinecone for scalable similarity search."""
    
    def __init__(self):
        """Initialize Pinecone client and index."""
        self.pc = Pinecone(api_key=config.pinecone.api_key)
        self.index_name = config.pinecone.index_name
        self.dimension = config.pinecone.dimension
        
        # Initialize or connect to index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        logger.info("PineconeVectorStore initialized", 
                   index_name=self.index_name, 
                   dimension=self.dimension)
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info("Creating new Pinecone index", index_name=self.index_name)
                
                # Free tier only supports AWS us-east-1 region
                self.pc.create_index(
                    name=self.index_name,
                    vector_type="dense",  # Required for 2024 API
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Only free tier supported region
                    ),
                    deletion_protection="disabled"  # Easier management for demos
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
                    
                logger.info("Pinecone index created successfully", index_name=self.index_name)
            else:
                logger.info("Using existing Pinecone index", index_name=self.index_name)
                
        except Exception as e:
            error_msg = str(e).lower()
            
            # Provide specific guidance for common free tier errors
            if "free plan does not support" in error_msg:
                logger.error("Pinecone free tier limitation encountered", error=str(e))
                raise Exception(
                    "Pinecone Free Tier Error: Your free plan only supports AWS us-east-1 region. "
                    "This system is configured for free tier compatibility. "
                    "Please check: 1) Your Pinecone API key is correct, 2) You have a valid free account, "
                    "3) Restart this script to reload the updated configuration."
                )
            elif "invalid api key" in error_msg or "unauthorized" in error_msg:
                logger.error("Pinecone authentication failed", error=str(e))
                raise Exception(
                    "Pinecone Authentication Error: Please check your PINECONE_API_KEY in the .env file. "
                    "Get your API key from: https://app.pinecone.io/"
                )
            elif "quota" in error_msg or "limit" in error_msg:
                logger.error("Pinecone quota exceeded", error=str(e))
                raise Exception(
                    "Pinecone Quota Error: You may have exceeded your free tier limits. "
                    "Free tier allows 5 indexes. Check your Pinecone dashboard: https://app.pinecone.io/"
                )
            elif "already exists" in error_msg:
                logger.warning("Index already exists, continuing with existing index", error=str(e))
                # This is not actually an error, just use the existing index
                pass
            else:
                logger.error("Pinecone index creation failed", error=str(e))
                raise Exception(f"Pinecone Error: {str(e)}")
    
    
    def add_documents(self, 
                     embeddings: List[List[float]], 
                     texts: List[str], 
                     metadatas: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks with embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            texts: List of document chunk texts
            metadatas: List of metadata dicts for each chunk
            
        Returns:
            bool: Success status
        """
        try:
            if len(embeddings) != len(texts) or len(texts) != len(metadatas):
                raise ValueError("Embeddings, texts, and metadatas must have the same length")
            
            # Prepare vectors for upsert
            vectors = []
            for i, (embedding, text, metadata) in enumerate(zip(embeddings, texts, metadatas)):
                vector_id = f"doc_{int(time.time())}_{i}"
                
                # Add text to metadata for retrieval
                full_metadata = {
                    **metadata,
                    "text": text,
                    "timestamp": int(time.time())
                }
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": full_metadata
                })
            
            # Batch upsert to Pinecone
            batch_size = 100  # Pinecone recommendation
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info("Documents added to vector store", 
                       count=len(vectors),
                       index_name=self.index_name)
            return True
            
        except Exception as e:
            logger.error("Failed to add documents to vector store", error=str(e))
            raise
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         top_k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of top results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Query Pinecone
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            results = []
            for match in query_response.matches:
                result = {
                    "id": match.id,
                    "score": float(match.score),
                    "text": match.metadata.get("text", ""),
                    "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
                    "similarity": float(match.score)  # Cosine similarity (0-1, higher is better)
                }
                results.append(result)
            
            logger.info("Similarity search completed", 
                       query_results=len(results),
                       top_score=results[0]["score"] if results else 0)
            
            return results
            
        except Exception as e:
            logger.error("Similarity search failed", error=str(e))
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
        except Exception as e:
            logger.error("Failed to get vector store stats", error=str(e))
            return {}
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """Delete vectors by metadata filter."""
        try:
            self.index.delete(filter=filter_dict)
            logger.info("Vectors deleted by filter", filter=filter_dict)
            return True
        except Exception as e:
            logger.error("Failed to delete vectors", error=str(e), filter=filter_dict)
            return False
    
    def clear_index(self) -> bool:
        """Clear all vectors from the index."""
        try:
            self.index.delete(delete_all=True)
            logger.warning("All vectors cleared from index", index_name=self.index_name)
            return True
        except Exception as e:
            logger.error("Failed to clear index", error=str(e))
            return False