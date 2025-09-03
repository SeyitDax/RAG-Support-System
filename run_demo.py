#!/usr/bin/env python3
"""
Demo Startup Script for RAG Customer Support System

This script handles the complete setup and initialization of the demo system,
including knowledge base ingestion and API server startup.
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.rag_engine import RAGEngine, DocumentProcessor
    from src.rag_engine.config import config
    from src.api.app import create_app
    import structlog
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you have installed all dependencies:")
    print("pip install -r requirements.txt")
    sys.exit(1)

logger = structlog.get_logger(__name__)

def print_banner():
    """Print startup banner."""
    banner = """
    ===============================================================
                   RAG Customer Support System                  
                        Demo Startup                           
    ===============================================================
     AI-Powered Customer Support with Confidence-Based Routing  
     OpenAI GPT-4 + Pinecone + Flask + n8n Integration        
    ===============================================================
    """
    print(banner)

def check_environment() -> Dict[str, bool]:
    """Check if required environment variables are set."""
    print("Checking environment configuration...")
    
    required_vars = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY"
    ]
    
    env_status = {}
    for var in required_vars:
        value = os.getenv(var)
        is_set = bool(value and value != f"your-{var.lower().replace('_', '-')}-here")
        env_status[var] = is_set
        
        status = "[OK]" if is_set else "[MISSING]"
        print(f"  {status} {var}: {'Set' if is_set else 'Missing'}")
    
    return env_status

def create_env_file_if_missing():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("[OK] .env file created. Please edit it with your API keys.")
        return False
    elif not env_file.exists():
        print("[ERROR] No .env file found and no .env.example template available.")
        return False
    
    return True

def initialize_knowledge_base() -> bool:
    """Initialize the knowledge base with documents."""
    print("Initializing knowledge base...")
    
    try:
        # Check if knowledge base directory exists
        kb_path = Path("data/knowledge_base")
        if not kb_path.exists() or not any(kb_path.glob("*.md")):
            print("[ERROR] Knowledge base directory not found or empty.")
            print(f"Expected path: {kb_path.absolute()}")
            return False
        
        print("  Creating RAG engine...")
        rag_engine = RAGEngine()
        
        print("  Processing knowledge base documents...")
        result = rag_engine.ingest_directory(str(kb_path))
        
        if result.get("success", False):
            total_chunks = result.get("total_chunks", 0)
            processing_time = result.get("processing_time", 0)
            print(f"  [OK] Successfully ingested {total_chunks} chunks in {processing_time:.2f}s")
            return True
        else:
            print(f"  [ERROR] Failed to ingest knowledge base: {result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Error initializing knowledge base: {e}")
        return False

def start_api_server():
    """Start the Flask API server."""
    print("Starting API server...")
    
    try:
        app = create_app()
        
        host = config.flask.host
        port = config.flask.port
        
        print(f"  Server starting on http://{host}:{port}")
        print(f"  API endpoints available at http://{host}:{port}/api/")
        print(f"  Health check: http://{host}:{port}/api/health")
        
        # Start server
        app.run(
            host=host,
            port=port,
            debug=config.flask.debug,
            threaded=True
        )
        
    except Exception as e:
        print(f"  [ERROR] Failed to start API server: {e}")
        return False

def open_demo_interface():
    """Open the demo interface in the browser."""
    print("Opening demo interface...")
    
    frontend_path = Path("frontend/index.html")
    if not frontend_path.exists():
        print("[ERROR] Demo interface not found at frontend/index.html")
        return False
    
    try:
        # Open in browser
        webbrowser.open(f"file://{frontend_path.absolute()}")
        print("  [OK] Demo interface opened in browser")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not auto-open browser: {e}")
        print(f"  Manually open: file://{frontend_path.absolute()}")
        return True

def print_usage_info():
    """Print usage information and next steps."""
    print("""
System Ready! Here's what you can do:

Demo Interface:
  - Chat interface with real-time confidence scoring
  - Try example queries or ask your own questions
  - View source attribution and processing times

API Endpoints:
  - POST /api/query - Process customer questions
  - GET /api/health - System health check  
  - GET /api/analytics - Performance metrics
  - GET /api/system/stats - System statistics

Test Queries to Try:
  - "How do I return a defective product?"
  - "What shipping options do you offer?"
  - "Is the iPhone 15 compatible with MagSafe?"
  - "What is the meaning of life?" (out-of-scope test)

Business Logic:
  - High confidence (>80%): Auto-response [OK]
  - Medium confidence (60-80%): Needs review [REVIEW]
  - Low confidence (<60%): Escalate to human [ESCALATE]

System Components:
  - OpenAI GPT-4 for response generation
  - Pinecone vector database for document search
  - Flask REST API with comprehensive endpoints
  - Professional frontend with real-time features

To Stop: Press Ctrl+C
    """)

def main():
    """Main demo startup sequence."""
    print_banner()
    
    # Check environment
    create_env_file_if_missing()
    env_status = check_environment()
    
    if not all(env_status.values()):
        print("\n[ERROR] Environment setup incomplete!")
        print("Please set your API keys in the .env file:")
        print("  1. Get OpenAI API key from https://platform.openai.com/api-keys")
        print("  2. Get Pinecone API key from https://www.pinecone.io/")
        print("  3. Edit .env file with your keys")
        print("  4. Run this script again")
        sys.exit(1)
    
    print("[OK] Environment configuration complete!")
    
    # Initialize knowledge base
    kb_success = initialize_knowledge_base()
    if not kb_success:
        print("\n[WARNING] Knowledge base initialization failed!")
        print("The API will still start, but responses may be limited.")
        print("")
        print("Common solutions:")
        print("1. Check if your Pinecone API key is correct")
        print("2. Ensure you have a valid Pinecone free tier account")
        print("3. Restart this script completely to reload the updated configuration")
        print("4. You can try ingesting documents later via the /api/ingest endpoint")
        print("")
        input("Press Enter to continue anyway...")
    
    # Open demo interface (non-blocking)
    open_demo_interface()
    
    # Print usage info
    print_usage_info()
    
    # Start API server (blocking)
    try:
        start_api_server()
    except KeyboardInterrupt:
        print("\n\nShutting down RAG Customer Support System...")
        print("Thank you for trying the demo!")
        sys.exit(0)

if __name__ == "__main__":
    main()