"""
Production RAG Engine with OpenAI Integration

Main RAG engine that combines document processing, vector storage, and OpenAI
for intelligent customer support with confidence-based routing.
"""

import time
from typing import Dict, List, Any, Tuple, Optional
import openai
from openai import OpenAI
import numpy as np
import structlog
from .config import config
from .document_processor import DocumentProcessor
from .vector_store import PineconeVectorStore

logger = structlog.get_logger(__name__)


class RAGEngine:
    """Production RAG engine with OpenAI embeddings and GPT-4 generation."""
    
    def __init__(self):
        """Initialize RAG engine with OpenAI client and vector store."""
        self.client = OpenAI(api_key=config.openai.api_key)
        self.document_processor = DocumentProcessor()
        self.vector_store = PineconeVectorStore()
        
        # Test API connectivity
        self._test_openai_connection()
        
        logger.info("RAGEngine initialized successfully",
                   openai_model=config.openai.model,
                   embedding_model=config.openai.embedding_model)
    
    def _test_openai_connection(self):
        """Test OpenAI API connectivity."""
        try:
            # Test with a simple embedding request
            response = self.client.embeddings.create(
                model=config.openai.embedding_model,
                input="test connection"
            )
            logger.info("OpenAI connection successful")
        except Exception as e:
            logger.error("OpenAI connection failed", error=str(e))
            raise
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest documents from file paths into the knowledge base.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Ingestion results with statistics
        """
        start_time = time.time()
        total_chunks = 0
        failed_files = []
        
        try:
            all_chunks = []
            
            # Process each file
            for file_path in file_paths:
                try:
                    chunks = self.document_processor.process_file(file_path)
                    all_chunks.extend(chunks)
                    total_chunks += len(chunks)
                except Exception as e:
                    logger.warning("Failed to process file", 
                                 file=file_path, 
                                 error=str(e))
                    failed_files.append(file_path)
            
            if not all_chunks:
                return {
                    "success": False,
                    "message": "No chunks were successfully processed",
                    "failed_files": failed_files
                }
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in all_chunks]
            metadatas = [chunk["metadata"] for chunk in all_chunks]
            
            embeddings = self._generate_embeddings(texts)
            
            # Store in vector database
            success = self.vector_store.add_documents(embeddings, texts, metadatas)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": success,
                "total_files": len(file_paths),
                "failed_files": failed_files,
                "total_chunks": total_chunks,
                "processing_time": round(processing_time, 2),
                "average_chunk_size": np.mean([len(text) for text in texts]) if texts else 0
            }
            
            logger.info("Document ingestion completed", **result)
            return result
            
        except Exception as e:
            logger.error("Document ingestion failed", error=str(e))
            return {
                "success": False,
                "message": f"Ingestion failed: {str(e)}",
                "failed_files": failed_files
            }
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            Ingestion results with statistics
        """
        start_time = time.time()
        
        try:
            # Process all files in directory
            all_chunks = self.document_processor.process_directory(directory_path)
            
            if not all_chunks:
                return {
                    "success": False,
                    "message": "No documents found or processed in directory",
                    "directory": directory_path
                }
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in all_chunks]
            metadatas = [chunk["metadata"] for chunk in all_chunks]
            
            embeddings = self._generate_embeddings(texts)
            
            # Store in vector database
            success = self.vector_store.add_documents(embeddings, texts, metadatas)
            
            processing_time = time.time() - start_time
            
            result = {
                "success": success,
                "directory": directory_path,
                "total_chunks": len(all_chunks),
                "processing_time": round(processing_time, 2),
                "unique_sources": len(set(chunk["metadata"]["source"] for chunk in all_chunks))
            }
            
            logger.info("Directory ingestion completed", **result)
            return result
            
        except Exception as e:
            logger.error("Directory ingestion failed", 
                        directory=directory_path, 
                        error=str(e))
            return {
                "success": False,
                "message": f"Directory ingestion failed: {str(e)}",
                "directory": directory_path
            }
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """
        Query the RAG system with confidence scoring and source attribution.
        
        Args:
            question: Customer question to answer
            top_k: Number of relevant chunks to retrieve (default from config)
            
        Returns:
            Response with answer, confidence score, sources, and business decision
        """
        start_time = time.time()
        
        if top_k is None:
            top_k = config.rag.similarity_top_k
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([question])[0]
            
            # Retrieve relevant chunks
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            if not search_results:
                return self._create_no_results_response(question, time.time() - start_time)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(search_results, question)
            
            # Generate response using GPT-4
            context_text = "\n\n".join([result["text"] for result in search_results])
            response_text = self._generate_response(question, context_text)
            
            # Extract sources
            sources = self._extract_sources(search_results)
            
            # Make business decision
            should_escalate = confidence < config.rag.confidence_threshold_low
            auto_response = confidence >= config.rag.confidence_threshold_high
            
            processing_time = time.time() - start_time
            
            result = {
                "response": response_text,
                "confidence": round(confidence, 3),
                "sources": sources,
                "should_escalate": should_escalate,
                "auto_response": auto_response,
                "processing_time": round(processing_time, 2),
                "retrieved_chunks": len(search_results),
                "search_results": search_results  # For debugging/analysis
            }
            
            logger.info("Query processed successfully", 
                       question_length=len(question),
                       confidence=confidence,
                       should_escalate=should_escalate,
                       processing_time=processing_time)
            
            return result
            
        except Exception as e:
            logger.error("Query processing failed", 
                        question=question[:100] + "..." if len(question) > 100 else question,
                        error=str(e))
            return self._create_error_response(str(e), time.time() - start_time)
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        try:
            # Batch process for efficiency
            response = self.client.embeddings.create(
                model=config.openai.embedding_model,
                input=texts
            )
            
            embeddings = [data.embedding for data in response.data]
            
            logger.info("Embeddings generated successfully", 
                       text_count=len(texts),
                       embedding_dimension=len(embeddings[0]) if embeddings else 0)
            
            return embeddings
            
        except Exception as e:
            logger.error("Embedding generation failed", 
                        text_count=len(texts),
                        error=str(e))
            raise
    
    def _generate_response(self, question: str, context: str) -> str:
        """Generate response using GPT-4 with context."""
        try:
            prompt = self._create_response_prompt(question, context)
            
            response = self.client.chat.completions.create(
                model=config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a helpful customer support assistant. Provide accurate, concise answers based on the provided context. If you cannot answer based on the context, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.openai.max_tokens,
                temperature=config.openai.temperature
            )
            
            response_text = response.choices[0].message.content.strip()
            
            logger.info("Response generated successfully", 
                       response_length=len(response_text))
            
            return response_text
            
        except Exception as e:
            logger.error("Response generation failed", error=str(e))
            return "I apologize, but I'm unable to generate a response at this time due to a technical issue. Please contact our support team for assistance."
    
    def _create_response_prompt(self, question: str, context: str) -> str:
        """Create prompt for response generation."""
        return f"""Based on the following customer support information, please answer the customer's question accurately and helpfully.

CONTEXT:
{context}

CUSTOMER QUESTION:
{question}

Please provide a clear, helpful answer based on the context provided. If the context doesn't contain enough information to answer the question completely, acknowledge this and suggest contacting customer support for more specific assistance."""
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]], question: str) -> float:
        """
        Calculate confidence score based on multiple factors.
        
        Args:
            search_results: List of search results with similarity scores
            question: Original question for semantic analysis
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results:
            return 0.0
        
        # Factor 1: Average similarity score (primary factor)
        avg_similarity = np.mean([result["similarity"] for result in search_results])
        similarity_weight = 0.6
        
        # Factor 2: Top result score (importance of best match)
        top_score = search_results[0]["similarity"] if search_results else 0.0
        top_score_weight = 0.25
        
        # Factor 3: Consistency between top results (how similar are top results)
        consistency = self._calculate_result_consistency(search_results[:3])
        consistency_weight = 0.15
        
        # Calculate weighted confidence
        confidence = (
            avg_similarity * similarity_weight +
            top_score * top_score_weight +
            consistency * consistency_weight
        )
        
        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))
    
    def _calculate_result_consistency(self, results: List[Dict[str, Any]]) -> float:
        """Calculate consistency score between top results."""
        if len(results) < 2:
            return 1.0
        
        scores = [result["similarity"] for result in results]
        score_std = np.std(scores)
        
        # Lower standard deviation means higher consistency
        # Normalize to 0-1 scale (assuming max std of 0.3 for similarity scores)
        consistency = max(0.0, 1.0 - (score_std / 0.3))
        
        return consistency
    
    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and format source information from search results."""
        sources = []
        seen_sources = set()
        
        for result in search_results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            
            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "document_type": metadata.get("document_type", "unknown"),
                    "relevance_score": round(result["similarity"], 3)
                })
                seen_sources.add(source)
        
        return sources
    
    def _create_no_results_response(self, question: str, processing_time: float) -> Dict[str, Any]:
        """Create response for when no relevant documents are found."""
        return {
            "response": "I don't have specific information to answer your question. Please contact our customer support team for personalized assistance.",
            "confidence": 0.0,
            "sources": [],
            "should_escalate": True,
            "auto_response": False,
            "processing_time": round(processing_time, 2),
            "retrieved_chunks": 0,
            "message": "No relevant information found in knowledge base"
        }
    
    def _create_error_response(self, error_message: str, processing_time: float) -> Dict[str, Any]:
        """Create response for error cases."""
        return {
            "response": "I'm experiencing technical difficulties and cannot process your request right now. Please try again later or contact customer support.",
            "confidence": 0.0,
            "sources": [],
            "should_escalate": True,
            "auto_response": False,
            "processing_time": round(processing_time, 2),
            "retrieved_chunks": 0,
            "error": error_message
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health information."""
        try:
            vector_stats = self.vector_store.get_stats()
            
            return {
                "vector_store": vector_stats,
                "configuration": {
                    "chunk_size": config.rag.chunk_size,
                    "chunk_overlap": config.rag.chunk_overlap,
                    "confidence_high_threshold": config.rag.confidence_threshold_high,
                    "confidence_low_threshold": config.rag.confidence_threshold_low,
                    "similarity_top_k": config.rag.similarity_top_k
                },
                "openai_models": {
                    "completion_model": config.openai.model,
                    "embedding_model": config.openai.embedding_model
                }
            }
            
        except Exception as e:
            logger.error("Failed to get system stats", error=str(e))
            return {"error": "Unable to retrieve system statistics"}