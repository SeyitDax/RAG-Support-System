"""
API Routes for RAG Support System

Flask routes for customer support automation with confidence-based routing,
analytics, and integration endpoints for n8n workflows.
"""

import time
import uuid
from typing import Dict, Any
from flask import Blueprint, request, jsonify, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import structlog

from .models import (
    QueryRequest, QueryResponse, FeedbackRequest, FeedbackResponse,
    IngestionRequest, IngestionResponse, HealthResponse,
    AnalyticsRequest, AnalyticsResponse, SystemStatsResponse,
    ErrorResponse
)
from ..rag_engine import RAGEngine
from ..rag_engine.config import config
from ..utils.helpers import (
    sanitize_query, validate_query_input, create_error_response,
    create_success_response, log_query_metrics
)

logger = structlog.get_logger(__name__)

# Create blueprint for API routes
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "20 per minute"]
)

# Global RAG engine instance (initialized in app factory)
rag_engine: RAGEngine = None


def init_routes(engine: RAGEngine, rate_limiter: Limiter):
    """Initialize routes with RAG engine and rate limiter."""
    global rag_engine, limiter
    rag_engine = engine
    limiter = rate_limiter


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for system monitoring.
    
    Returns:
        JSON response with system health status
    """
    try:
        start_time = time.time()
        
        # Check core components
        components = {
            "api": "healthy",
            "rag_engine": "unknown",
            "vector_store": "unknown",
            "openai": "unknown"
        }
        
        # Test RAG engine if available
        if rag_engine:
            try:
                stats = rag_engine.get_system_stats()
                components["rag_engine"] = "healthy"
                components["vector_store"] = "healthy" if stats.get("vector_store") else "error"
                components["openai"] = "healthy"
            except Exception as e:
                components["rag_engine"] = "error"
                logger.warning("RAG engine health check failed", error=str(e))
        
        response = HealthResponse(
            status="healthy" if all(status == "healthy" for status in components.values()) else "degraded",
            timestamp=int(time.time()),
            uptime=time.time() - start_time,
            components=components
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        error_response = ErrorResponse(
            error="Health check failed",
            error_type="system_error",
            timestamp=int(time.time())
        )
        return jsonify(error_response.dict()), 500


@api_bp.route('/query', methods=['POST'])
@limiter.limit("30 per minute")
def process_query():
    """
    Process customer query with confidence-based routing.
    
    Returns:
        JSON response with answer, confidence score, and routing decision
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate request data
        if not request.is_json:
            return jsonify(create_error_response(
                "Request must be JSON",
                "validation_error",
                time.time() - start_time
            )), 400
        
        data = request.get_json()
        if not data:
            return jsonify(create_error_response(
                "Empty request body",
                "validation_error", 
                time.time() - start_time
            )), 400
        
        # Parse and validate request
        try:
            query_request = QueryRequest(**data)
        except Exception as e:
            return jsonify(create_error_response(
                f"Invalid request format: {str(e)}",
                "validation_error",
                time.time() - start_time
            )), 400
        
        # Validate query content
        is_valid, error_msg = validate_query_input(query_request.question)
        if not is_valid:
            return jsonify(create_error_response(
                error_msg,
                "validation_error",
                time.time() - start_time
            )), 400
        
        # Sanitize query
        sanitized_query = sanitize_query(query_request.question)
        
        # Check if RAG engine is available
        if not rag_engine:
            return jsonify(create_error_response(
                "RAG engine not available",
                "system_error",
                time.time() - start_time
            )), 503
        
        # Process query through RAG engine
        result = rag_engine.query(
            question=sanitized_query,
            top_k=query_request.top_k
        )
        
        # Format response
        response_data = create_success_response(
            response=result["response"],
            confidence=result["confidence"],
            sources=result.get("sources", []),
            processing_time=result.get("processing_time", time.time() - start_time),
            additional_data={
                "retrieved_chunks": result.get("retrieved_chunks", 0),
                "request_id": request_id
            }
        )
        
        # Log metrics
        log_query_metrics(
            query=sanitized_query,
            response_data=response_data,
            user_id=query_request.user_id
        )
        
        # Determine HTTP status based on confidence
        status_code = 200
        if result.get("should_escalate", False):
            status_code = 202  # Accepted but requires human intervention
        
        logger.info("Query processed successfully",
                   request_id=request_id,
                   confidence=result["confidence"],
                   processing_time=result.get("processing_time", 0),
                   should_escalate=result.get("should_escalate", False))
        
        return jsonify(response_data), status_code
        
    except Exception as e:
        logger.error("Query processing failed", 
                    request_id=request_id,
                    error=str(e))
        
        error_response = create_error_response(
            "An error occurred while processing your query. Please try again later.",
            "processing_error",
            time.time() - start_time
        )
        error_response["request_id"] = request_id
        
        return jsonify(error_response), 500


@api_bp.route('/feedback', methods=['POST'])
@limiter.limit("10 per minute")
def submit_feedback():
    """
    Submit feedback on query responses.
    
    Returns:
        JSON response confirming feedback submission
    """
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "Empty request body"
            }), 400
        
        # Parse and validate feedback
        try:
            feedback_request = FeedbackRequest(**data)
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Invalid feedback format: {str(e)}"
            }), 400
        
        # Generate feedback ID
        feedback_id = str(uuid.uuid4())
        
        # Log feedback (in production, save to database)
        logger.info("Feedback received",
                   feedback_id=feedback_id,
                   query_id=feedback_request.query_id,
                   rating=feedback_request.rating,
                   helpful=feedback_request.helpful,
                   accurate=feedback_request.accurate,
                   user_id=feedback_request.user_id)
        
        response = FeedbackResponse(
            success=True,
            message="Feedback received successfully",
            feedback_id=feedback_id
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error("Feedback submission failed", error=str(e))
        return jsonify({
            "success": False,
            "message": "Failed to submit feedback"
        }), 500


@api_bp.route('/ingest', methods=['POST'])
@limiter.limit("5 per minute")
def ingest_documents():
    """
    Ingest documents into the knowledge base.
    
    Returns:
        JSON response with ingestion results
    """
    start_time = time.time()
    
    try:
        if not request.is_json:
            return jsonify({
                "success": False,
                "message": "Request must be JSON"
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "Empty request body"
            }), 400
        
        # Parse and validate ingestion request
        try:
            ingest_request = IngestionRequest(**data)
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Invalid request format: {str(e)}"
            }), 400
        
        # Check if RAG engine is available
        if not rag_engine:
            return jsonify({
                "success": False,
                "message": "RAG engine not available"
            }), 503
        
        # Process ingestion based on request type
        result = None
        
        if ingest_request.file_paths:
            result = rag_engine.ingest_documents(ingest_request.file_paths)
        elif ingest_request.directory_path:
            result = rag_engine.ingest_directory(ingest_request.directory_path)
        elif ingest_request.text_content:
            # Process text content through document processor
            chunks = rag_engine.document_processor.process_text_content(
                text=ingest_request.text_content,
                source_name=ingest_request.source_name,
                document_type=ingest_request.document_type
            )
            
            if chunks:
                texts = [chunk["text"] for chunk in chunks]
                metadatas = [chunk["metadata"] for chunk in chunks]
                embeddings = rag_engine._generate_embeddings(texts)
                success = rag_engine.vector_store.add_documents(embeddings, texts, metadatas)
                
                result = {
                    "success": success,
                    "total_chunks": len(chunks),
                    "processing_time": time.time() - start_time,
                    "source_name": ingest_request.source_name
                }
            else:
                result = {
                    "success": False,
                    "message": "No chunks generated from text content"
                }
        
        if not result:
            return jsonify({
                "success": False,
                "message": "No valid ingestion source provided"
            }), 400
        
        # Format response
        response = IngestionResponse(
            success=result.get("success", False),
            message=result.get("message", "Ingestion completed" if result.get("success") else "Ingestion failed"),
            total_chunks=result.get("total_chunks"),
            processing_time=result.get("processing_time"),
            failed_files=result.get("failed_files", []),
            stats=result
        )
        
        logger.info("Document ingestion completed",
                   success=result.get("success", False),
                   total_chunks=result.get("total_chunks", 0),
                   processing_time=result.get("processing_time", 0))
        
        return jsonify(response.dict()), 200 if result.get("success") else 400
        
    except Exception as e:
        logger.error("Document ingestion failed", error=str(e))
        return jsonify({
            "success": False,
            "message": "Document ingestion failed due to internal error"
        }), 500


@api_bp.route('/analytics', methods=['GET'])
# @limiter.limit("20 per minute")  # Temporarily disabled due to memory reference issue
def get_analytics():
    """
    Get system analytics and metrics.
    
    Returns:
        JSON response with analytics data
    """
    try:
        # Parse query parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        metric_types = request.args.getlist('metrics')
        user_id = request.args.get('user_id')
        
        # In a production system, this would query the database
        # For now, return mock analytics data
        mock_metrics = [
            {
                "metric_name": "total_queries",
                "value": 1250.0,
                "timestamp": int(time.time()),
                "metadata": {"period": "24h"}
            },
            {
                "metric_name": "average_confidence",
                "value": 0.782,
                "timestamp": int(time.time()),
                "metadata": {"period": "24h"}
            },
            {
                "metric_name": "automation_rate",
                "value": 0.64,
                "timestamp": int(time.time()),
                "metadata": {"period": "24h"}
            },
            {
                "metric_name": "average_processing_time",
                "value": 1.45,
                "timestamp": int(time.time()),
                "metadata": {"unit": "seconds", "period": "24h"}
            }
        ]
        
        response = AnalyticsResponse(
            success=True,
            metrics=mock_metrics,
            summary={
                "total_queries": 1250,
                "successful_queries": 1198,
                "escalated_queries": 52,
                "average_confidence": 0.782,
                "automation_rate": 0.64
            }
        )
        
        logger.info("Analytics data retrieved",
                   metric_count=len(mock_metrics),
                   start_date=start_date,
                   end_date=end_date)
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error("Analytics retrieval failed", error=str(e))
        return jsonify({
            "success": False,
            "error": "Failed to retrieve analytics data"
        }), 500


@api_bp.route('/system/stats', methods=['GET'])
# @limiter.limit("10 per minute")  # Temporarily disabled due to memory reference issue
def get_system_stats():
    """
    Get system statistics and configuration.
    
    Returns:
        JSON response with system stats
    """
    try:
        if not rag_engine:
            return jsonify({
                "error": "RAG engine not available"
            }), 503
        
        stats = rag_engine.get_system_stats()
        
        response = SystemStatsResponse(
            vector_store=stats.get("vector_store", {}),
            configuration=stats.get("configuration", {}),
            performance={
                "uptime": time.time(),  # Simplified uptime
                "memory_usage": "Unknown",  # Would implement actual monitoring
                "active_connections": 1
            }
        )
        
        return jsonify(response.dict()), 200
        
    except Exception as e:
        logger.error("System stats retrieval failed", error=str(e))
        return jsonify({
            "error": "Failed to retrieve system statistics"
        }), 500


@api_bp.route('/system/config', methods=['GET'])
def get_system_config():
    """
    Get system configuration (non-sensitive values only).
    
    Returns:
        JSON response with public configuration
    """
    try:
        public_config = {
            "rag": {
                "chunk_size": config.rag.chunk_size,
                "chunk_overlap": config.rag.chunk_overlap,
                "confidence_threshold_high": config.rag.confidence_threshold_high,
                "confidence_threshold_low": config.rag.confidence_threshold_low,
                "similarity_top_k": config.rag.similarity_top_k
            },
            "business": {
                "auto_response_enabled": config.business.auto_response_enabled,
                "escalation_enabled": config.business.escalation_enabled
            },
            "openai": {
                "model": config.openai.model,
                "embedding_model": config.openai.embedding_model,
                "max_tokens": config.openai.max_tokens,
                "temperature": config.openai.temperature
            }
        }
        
        return jsonify(public_config), 200
        
    except Exception as e:
        logger.error("Config retrieval failed", error=str(e))
        return jsonify({
            "error": "Failed to retrieve configuration"
        }), 500


# Error handlers
@api_bp.errorhandler(429)
def rate_limit_exceeded(error):
    """Handle rate limit exceeded errors."""
    return jsonify({
        "success": False,
        "error": "Rate limit exceeded",
        "error_type": "rate_limit",
        "message": "Too many requests. Please try again later."
    }), 429


@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "error_type": "not_found",
        "message": "The requested endpoint does not exist."
    }), 404


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error("Internal server error", error=str(error))
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "error_type": "server_error",
        "message": "An unexpected error occurred. Please try again later."
    }), 500