#!/usr/bin/env python3
"""
OpenAI Connection Test Script

This script tests OpenAI API connectivity with detailed error reporting
to help diagnose connection issues.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    from src.rag_engine.config import config
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    print("Please run: pip install -r requirements.txt")
    sys.exit(1)

def test_openai_connection():
    """Test OpenAI API connection with detailed diagnostics."""
    
    print("=" * 60)
    print("             OpenAI Connection Diagnostic")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    print("[OK] Environment variables loaded")
    
    # Check API key
    api_key = config.openai.api_key
    if not api_key:
        print("[ERROR] No OpenAI API key found in environment")
        print("Please set OPENAI_API_KEY in your .env file")
        return False
    
    # Validate API key format
    if not (api_key.startswith('sk-proj-') or api_key.startswith('sk-')):
        print(f"[ERROR] Invalid API key format: {api_key[:20]}...")
        print("OpenAI API keys should start with 'sk-proj-' or 'sk-'")
        return False
    
    print(f"[OK] API key format valid: {api_key[:20]}...")
    
    # Test connection
    print(f"Testing connection with model: {config.openai.embedding_model}")
    
    try:
        client = OpenAI(api_key=api_key)
        print("[OK] OpenAI client initialized")
        
        # Test with minimal embedding request
        print("Making test embedding request...")
        response = client.embeddings.create(
            model=config.openai.embedding_model,
            input="test"
        )
        
        print("[OK] Embedding request successful!")
        print(f"  Response type: {type(response)}")
        print(f"  Embedding dimension: {len(response.data[0].embedding)}")
        print(f"  Model used: {response.model}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] OpenAI API Error: {str(e)}")
        
        error_msg = str(e).lower()
        print("\nDiagnostic suggestions:")
        
        if "api key" in error_msg or "unauthorized" in error_msg:
            print("- Your API key appears to be invalid or expired")
            print("- Get a new API key from: https://platform.openai.com/api-keys")
            print("- Make sure there are no extra spaces in your .env file")
            
        elif "quota" in error_msg or "billing" in error_msg:
            print("- Your OpenAI account may be out of credits")
            print("- Check billing: https://platform.openai.com/account/billing")
            print("- Add a payment method if you haven't already")
            
        elif "rate" in error_msg or "limit" in error_msg:
            print("- You're being rate limited by OpenAI")
            print("- Wait a few minutes and try again")
            print("- Consider using a different API key if available")
            
        elif "connection" in error_msg or "network" in error_msg:
            print("- Network connection issue")
            print("- Check your internet connection")
            print("- Verify firewall isn't blocking OpenAI API (*.openai.com)")
            
        elif "model" in error_msg:
            print(f"- Model '{config.openai.embedding_model}' may not be accessible")
            print("- Try using 'text-embedding-ada-002' instead")
            print("- Check if you have access to the model")
            
        else:
            print(f"- Unexpected error: {str(e)}")
            print("- Try again in a few minutes")
            print("- Check OpenAI status: https://status.openai.com/")
        
        return False

def main():
    """Run OpenAI connection test."""
    success = test_openai_connection()
    
    print("\n" + "=" * 60)
    if success:
        print("OpenAI connection test PASSED!")
        print("The RAG system should work properly.")
    else:
        print("OpenAI connection test FAILED!")
        print("Please fix the issues above before running the demo.")
    print("=" * 60)

if __name__ == "__main__":
    main()