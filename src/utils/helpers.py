"""
Utility Functions for RAG Support System

Helper functions for confidence calculation, source formatting, query sanitization,
and other common operations used throughout the system.
"""

import re
import time
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger(__name__)


def calculate_confidence(search_results: List[Dict[str, Any]], 
                        query: str,
                        weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate overall confidence score for RAG query results.
    
    Args:
        search_results: List of search results with similarity scores
        query: Original query string
        weights: Custom weights for confidence factors
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not search_results:
        return 0.0
    
    # Default weights for confidence factors
    default_weights = {
        "similarity": 0.6,    # Average similarity score
        "top_result": 0.25,   # Best match quality
        "consistency": 0.15   # Result consistency
    }
    weights = weights or default_weights
    
    try:
        # Factor 1: Average similarity score
        similarities = [result.get("similarity", 0.0) for result in search_results]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Factor 2: Top result score
        top_similarity = similarities[0] if similarities else 0.0
        
        # Factor 3: Consistency between results
        consistency = calculate_result_consistency(similarities)
        
        # Calculate weighted confidence
        confidence = (
            avg_similarity * weights["similarity"] +
            top_similarity * weights["top_result"] +
            consistency * weights["consistency"]
        )
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))
        
        logger.debug("Confidence calculated",
                    avg_similarity=avg_similarity,
                    top_similarity=top_similarity,
                    consistency=consistency,
                    final_confidence=confidence)
        
        return confidence
        
    except Exception as e:
        logger.error("Confidence calculation failed", error=str(e))
        return 0.0


def calculate_result_consistency(similarities: List[float]) -> float:
    """
    Calculate consistency score based on similarity score variance.
    
    Args:
        similarities: List of similarity scores
        
    Returns:
        Consistency score between 0.0 and 1.0
    """
    if len(similarities) < 2:
        return 1.0
    
    # Calculate standard deviation
    mean_sim = sum(similarities) / len(similarities)
    variance = sum((s - mean_sim) ** 2 for s in similarities) / len(similarities)
    std_dev = variance ** 0.5
    
    # Convert to consistency score (lower std_dev = higher consistency)
    # Normalize assuming max reasonable std_dev of 0.3 for similarity scores
    consistency = max(0.0, 1.0 - (std_dev / 0.3))
    
    return consistency


def format_sources(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format and deduplicate source information from search results.
    
    Args:
        search_results: List of search results with metadata
        
    Returns:
        List of formatted, unique sources with relevance scores
    """
    sources = []
    seen_sources = set()
    
    try:
        for result in search_results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            
            # Skip duplicates
            if source in seen_sources:
                continue
            
            source_info = {
                "source": source,
                "document_type": metadata.get("document_type", "unknown"),
                "relevance_score": round(result.get("similarity", 0.0), 3),
                "chunk_count": 1  # Count how many chunks from this source
            }
            
            # Count additional chunks from same source
            for other_result in search_results:
                other_source = other_result.get("metadata", {}).get("source", "")
                if other_source == source and other_result != result:
                    source_info["chunk_count"] += 1
            
            sources.append(source_info)
            seen_sources.add(source)
        
        # Sort by relevance score (highest first)
        sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        logger.debug("Sources formatted", source_count=len(sources))
        
        return sources
        
    except Exception as e:
        logger.error("Source formatting failed", error=str(e))
        return []


def sanitize_query(query: str) -> str:
    """
    Sanitize user query for safe processing.
    
    Args:
        query: Raw user query string
        
    Returns:
        Sanitized query string
    """
    if not query or not isinstance(query, str):
        return ""
    
    try:
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove potentially harmful characters
        query = re.sub(r'[<>{}[\]|\\`]', '', query)
        
        # Limit query length
        max_length = 500
        if len(query) > max_length:
            query = query[:max_length].rsplit(' ', 1)[0] + "..."
            logger.warning("Query truncated due to length", 
                          original_length=len(query),
                          truncated_length=len(query))
        
        return query
        
    except Exception as e:
        logger.error("Query sanitization failed", error=str(e))
        return str(query)[:100] if query else ""


def validate_query_input(query: str) -> tuple[bool, str]:
    """
    Validate user query input for completeness and safety.
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Query cannot be empty"
    
    if not isinstance(query, str):
        return False, "Query must be a string"
    
    # Check minimum length
    if len(query.strip()) < 3:
        return False, "Query must be at least 3 characters long"
    
    # Check maximum length
    if len(query) > 1000:
        return False, "Query is too long (maximum 1000 characters)"
    
    # Check for only whitespace or special characters
    if not re.search(r'[a-zA-Z0-9]', query):
        return False, "Query must contain alphanumeric characters"
    
    return True, ""


def format_processing_time(seconds: float) -> str:
    """
    Format processing time for human-readable display.
    
    Args:
        seconds: Processing time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0.001:
        return "< 1ms"
    elif seconds < 1:
        return f"{int(seconds * 1000)}ms"
    else:
        return f"{seconds:.2f}s"


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for analysis and metadata.
    
    Args:
        text: Input text to analyze
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    try:
        # Simple keyword extraction (can be enhanced with NLP libraries)
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'cannot', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
            'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', text.lower())
        
        # Filter out stop words and short words
        keywords = [word for word in words 
                   if word not in stop_words and len(word) > 2]
        
        # Count word frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_keywords[:max_keywords]]
        
    except Exception as e:
        logger.error("Keyword extraction failed", error=str(e))
        return []


def create_error_response(message: str, 
                         error_type: str = "processing_error",
                         processing_time: float = 0.0) -> Dict[str, Any]:
    """
    Create standardized error response structure.
    
    Args:
        message: Error message for user
        error_type: Type of error for logging
        processing_time: Time spent processing before error
        
    Returns:
        Standardized error response dictionary
    """
    return {
        "success": False,
        "response": message,
        "confidence": 0.0,
        "sources": [],
        "should_escalate": True,
        "auto_response": False,
        "processing_time": round(processing_time, 2),
        "error_type": error_type,
        "timestamp": int(time.time())
    }


def create_success_response(response: str,
                          confidence: float,
                          sources: List[Dict[str, Any]],
                          processing_time: float,
                          additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create standardized success response structure.
    
    Args:
        response: Generated response text
        confidence: Confidence score
        sources: Source information
        processing_time: Processing time in seconds
        additional_data: Optional additional response data
        
    Returns:
        Standardized success response dictionary
    """
    from ..rag_engine.config import config
    
    response_data = {
        "success": True,
        "response": response,
        "confidence": round(confidence, 3),
        "sources": sources,
        "should_escalate": confidence < config.rag.confidence_threshold_low,
        "auto_response": confidence >= config.rag.confidence_threshold_high,
        "processing_time": round(processing_time, 2),
        "timestamp": int(time.time())
    }
    
    if additional_data:
        response_data.update(additional_data)
    
    return response_data


def log_query_metrics(query: str, 
                     response_data: Dict[str, Any],
                     user_id: Optional[str] = None) -> None:
    """
    Log query metrics for analytics and monitoring.
    
    Args:
        query: Original user query
        response_data: Response data from RAG engine
        user_id: Optional user identifier
    """
    try:
        metrics = {
            "query_length": len(query),
            "confidence": response_data.get("confidence", 0.0),
            "processing_time": response_data.get("processing_time", 0.0),
            "source_count": len(response_data.get("sources", [])),
            "should_escalate": response_data.get("should_escalate", False),
            "auto_response": response_data.get("auto_response", False),
            "user_id": user_id,
            "timestamp": int(time.time())
        }
        
        logger.info("Query metrics logged", **metrics)
        
    except Exception as e:
        logger.error("Failed to log query metrics", error=str(e))