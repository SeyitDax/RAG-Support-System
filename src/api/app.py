"""
Flask Application Factory for RAG Support System

Creates and configures the Flask application with all necessary components
for production deployment of the customer support automation system.
"""

import time
import os
from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import structlog

from .routes import api_bp, init_routes
from ..rag_engine import RAGEngine
from ..rag_engine.config import config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global application state
app_start_time = time.time()
rag_engine_instance = None


def create_app(testing: bool = False) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        testing: Whether to configure for testing
        
    Returns:
        Configured Flask application
    """
    global rag_engine_instance
    
    app = Flask(__name__)
    
    # Configure Flask settings
    configure_app(app, testing)
    
    # Initialize CORS (allow file:// URLs for demo interface)
    CORS(app, 
         origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080", "null"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
         supports_credentials=False)
    
    # Initialize rate limiter with stable configuration (temporarily simplified)
    try:
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=["1000 per hour", "100 per minute"],
            storage_uri="memory://",  # Explicit in-memory storage
            headers_enabled=True  # Enable rate limit headers
        )
    except Exception as e:
        logger.warning("Failed to initialize rate limiter", error=str(e))
        # Create a dummy limiter that doesn't actually limit
        limiter = None
    
    # Initialize RAG engine (skip in testing mode to avoid API calls)
    if not testing:
        try:
            logger.info("Initializing RAG engine...")
            rag_engine_instance = RAGEngine()
            logger.info("RAG engine initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize RAG engine", error=str(e))
            # In production, you might want to fail fast here
            # For demo purposes, we'll continue without RAG engine
            rag_engine_instance = None
    
    # Initialize routes with dependencies (handle None limiter gracefully)
    init_routes(rag_engine_instance, limiter)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register application hooks
    register_hooks(app)
    
    # Add health check at root level
    @app.route('/')
    def root():
        """Root endpoint with basic system information."""
        return jsonify({
            "service": "RAG Customer Support System",
            "version": "1.0.0",
            "status": "running",
            "uptime": round(time.time() - app_start_time, 2),
            "endpoints": {
                "health": "/api/health",
                "query": "/api/query",
                "feedback": "/api/feedback",
                "ingest": "/api/ingest",
                "analytics": "/api/analytics",
                "stats": "/api/system/stats",
                "config": "/api/system/config"
            }
        })
    
    logger.info("Flask application created successfully", 
               testing=testing,
               rag_engine_available=rag_engine_instance is not None)
    
    return app


def configure_app(app: Flask, testing: bool = False):
    """Configure Flask application settings."""
    
    if testing:
        app.config.update({
            'TESTING': True,
            'SECRET_KEY': 'test-secret-key',
            'DEBUG': False,
            'WTF_CSRF_ENABLED': False
        })
    else:
        app.config.update({
            'SECRET_KEY': config.flask.secret_key,
            'DEBUG': config.flask.debug,
            'ENV': config.flask.env,
            'JSON_SORT_KEYS': False,
            'JSONIFY_PRETTYPRINT_REGULAR': True,
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max request size
        })


def register_error_handlers(app: Flask):
    """Register global error handlers."""
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({
            "success": False,
            "error": "Not Found",
            "error_type": "not_found",
            "message": "The requested resource was not found.",
            "timestamp": int(time.time())
        }), 404
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 errors."""
        return jsonify({
            "success": False,
            "error": "Bad Request",
            "error_type": "bad_request", 
            "message": "The request was invalid or malformed.",
            "timestamp": int(time.time())
        }), 400
    
    @app.errorhandler(413)
    def request_too_large(error):
        """Handle request too large errors."""
        return jsonify({
            "success": False,
            "error": "Request Too Large",
            "error_type": "request_too_large",
            "message": "The request payload is too large. Maximum size is 16MB.",
            "timestamp": int(time.time())
        }), 413
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 errors."""
        logger.error("Internal server error occurred", error=str(error))
        return jsonify({
            "success": False,
            "error": "Internal Server Error",
            "error_type": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": int(time.time())
        }), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle service unavailable errors."""
        return jsonify({
            "success": False,
            "error": "Service Unavailable", 
            "error_type": "service_unavailable",
            "message": "The service is temporarily unavailable. Please try again later.",
            "timestamp": int(time.time())
        }), 503


def register_hooks(app: Flask):
    """Register application hooks for logging and monitoring."""
    
    @app.before_request
    def before_request():
        """Log incoming requests."""
        from flask import request
        logger.info("Request started",
                   method=request.method,
                   path=request.path,
                   remote_addr=request.remote_addr,
                   user_agent=request.headers.get('User-Agent', 'Unknown')[:100])
    
    @app.after_request
    def after_request(response):
        """Log completed requests."""
        from flask import request
        logger.info("Request completed",
                   method=request.method,
                   path=request.path,
                   status_code=response.status_code,
                   content_length=response.content_length)
        return response
    
    @app.teardown_appcontext
    def teardown_appcontext(error):
        """Handle application context teardown."""
        if error:
            logger.error("Application context error", error=str(error))


def get_rag_engine():
    """Get the global RAG engine instance."""
    return rag_engine_instance


def health_check_detailed():
    """Perform detailed health check of all components."""
    health_status = {
        "timestamp": int(time.time()),
        "uptime": round(time.time() - app_start_time, 2),
        "components": {
            "api": "healthy",
            "rag_engine": "unknown",
            "vector_store": "unknown", 
            "openai": "unknown"
        },
        "metrics": {
            "memory_usage": "unknown",
            "active_connections": "unknown"
        }
    }
    
    # Check RAG engine
    if rag_engine_instance:
        try:
            stats = rag_engine_instance.get_system_stats()
            health_status["components"]["rag_engine"] = "healthy"
            
            # Check vector store
            vector_stats = stats.get("vector_store", {})
            if vector_stats and vector_stats.get("total_vectors") is not None:
                health_status["components"]["vector_store"] = "healthy"
                health_status["metrics"]["vector_count"] = vector_stats.get("total_vectors", 0)
            else:
                health_status["components"]["vector_store"] = "error"
            
            # OpenAI is healthy if RAG engine initialized successfully
            health_status["components"]["openai"] = "healthy"
            
        except Exception as e:
            logger.warning("Health check component failed", component="rag_engine", error=str(e))
            health_status["components"]["rag_engine"] = "error"
            health_status["components"]["vector_store"] = "error"
            health_status["components"]["openai"] = "error"
    else:
        health_status["components"]["rag_engine"] = "not_initialized"
    
    # Overall status
    component_statuses = list(health_status["components"].values())
    if all(status == "healthy" for status in component_statuses):
        health_status["status"] = "healthy"
    elif any(status == "healthy" for status in component_statuses):
        health_status["status"] = "degraded"
    else:
        health_status["status"] = "unhealthy"
    
    return health_status


if __name__ == '__main__':
    """Run the application in development mode."""
    app = create_app(testing=False)
    
    logger.info("Starting RAG Support System API server",
               host=config.flask.host,
               port=config.flask.port,
               debug=config.flask.debug)
    
    app.run(
        host=config.flask.host,
        port=config.flask.port,
        debug=config.flask.debug,
        threaded=True
    )