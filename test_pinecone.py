#!/usr/bin/env python3
"""
Pinecone Connection Test Script

This script tests Pinecone API connectivity with detailed error reporting
to help diagnose connection issues.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pinecone import Pinecone, ServerlessSpec
    from dotenv import load_dotenv
    from src.rag_engine.config import config
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

def test_pinecone_connection():
    """Test Pinecone API connection with detailed diagnostics."""
    
    print("=" * 60)
    print("             Pinecone Connection Diagnostic")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    print("[OK] Environment variables loaded")
    
    # Check API key
    api_key = config.pinecone.api_key
    if not api_key:
        print("[ERROR] No Pinecone API key found in environment")
        print("Please set PINECONE_API_KEY in your .env file")
        return False
    
    # Validate API key format (Pinecone keys typically start with specific patterns)
    if len(api_key) < 20:
        print(f"[ERROR] API key seems too short: {len(api_key)} characters")
        print("Pinecone API keys are typically longer")
        return False
    
    print(f"[OK] API key format looks valid: {api_key[:8]}...{api_key[-4:]} ({len(api_key)} chars)")
    
    # Test connection
    print(f"Testing connection...")
    print(f"Index name: {config.pinecone.index_name}")
    print(f"Dimension: {config.pinecone.dimension}")
    
    try:
        pc = Pinecone(api_key=api_key)
        print("[OK] Pinecone client initialized")
        
        # Test by listing indexes (doesn't create anything)
        print("Fetching existing indexes...")
        indexes = pc.list_indexes()
        
        print("[OK] Successfully connected to Pinecone!")
        
        if indexes:
            print(f"  Found {len(indexes)} existing indexes:")
            for idx in indexes:
                print(f"    - {idx.name} ({idx.dimension}D, {idx.metric})")
        else:
            print("  No existing indexes found (this is normal for new accounts)")
        
        # Check if our target index exists
        target_exists = any(idx.name == config.pinecone.index_name for idx in indexes)
        if target_exists:
            print(f"[OK] Target index '{config.pinecone.index_name}' already exists")
        else:
            print(f"[INFO] Target index '{config.pinecone.index_name}' will be created when needed")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Pinecone API Error: {str(e)}")
        
        error_msg = str(e).lower()
        print("\nDiagnostic suggestions:")
        
        if "api key" in error_msg or "unauthorized" in error_msg:
            print("- Your Pinecone API key appears to be invalid")
            print("- Get your API key from: https://app.pinecone.io/")
            print("- Make sure there are no extra spaces in your .env file")
            
        elif "free plan" in error_msg or "quota" in error_msg:
            print("- Your Pinecone free plan may have limitations")
            print("- Check your usage: https://app.pinecone.io/")
            print("- Free tier allows 5 indexes maximum")
            
        elif "connection" in error_msg or "network" in error_msg:
            print("- Network connection issue")
            print("- Check your internet connection")
            print("- Verify firewall isn't blocking Pinecone API")
            
        else:
            print(f"- Unexpected error: {str(e)}")
            print("- Check Pinecone status: https://status.pinecone.io/")
        
        return False

def main():
    """Run Pinecone connection test."""
    success = test_pinecone_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("Pinecone connection test PASSED!")
        print("The vector database should work properly.")
    else:
        print("Pinecone connection test FAILED!")
        print("Please fix the issues above before running the demo.")
    print("=" * 60)

if __name__ == "__main__":
    main()