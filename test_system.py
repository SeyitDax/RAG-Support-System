#!/usr/bin/env python3
"""
Complete System Test Script

Tests all components of the RAG support system to identify issues
before running the main demo.
"""

import os
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_step(step, status="", details=""):
    """Print test step with status."""
    if status == "PASS":
        print(f"[PASS] {step}")
    elif status == "FAIL":
        print(f"[FAIL] {step}")
    elif status == "WARN":
        print(f"[WARN] {step}")
    else:
        print(f"  {step}")
    
    if details:
        print(f"    {details}")

def test_imports():
    """Test all required imports."""
    print_header("Testing Package Imports")
    
    failed_imports = []
    
    try:
        from dotenv import load_dotenv
        print_step("dotenv", "PASS")
    except ImportError as e:
        print_step("dotenv", "FAIL", str(e))
        failed_imports.append("python-dotenv")
    
    try:
        from openai import OpenAI
        print_step("OpenAI", "PASS")
    except ImportError as e:
        print_step("OpenAI", "FAIL", str(e))
        failed_imports.append("openai")
    
    try:
        from pinecone import Pinecone, ServerlessSpec
        print_step("Pinecone", "PASS")
    except ImportError as e:
        print_step("Pinecone", "FAIL", str(e))
        failed_imports.append("pinecone")
    
    try:
        from src.rag_engine.config import config
        print_step("RAG Engine Config", "PASS")
    except ImportError as e:
        print_step("RAG Engine Config", "FAIL", str(e))
        failed_imports.append("rag-engine-config")
    
    if failed_imports:
        print(f"\n⚠ Failed imports detected. Please run:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def test_configuration():
    """Test configuration and API keys."""
    print_header("Testing Configuration")
    
    try:
        from dotenv import load_dotenv
        from src.rag_engine.config import config
        
        load_dotenv()
        print_step("Environment file loaded", "PASS")
        
        # Test OpenAI config
        if config.openai.api_key and config.openai.api_key != "sk-your-openai-api-key-here":
            print_step(f"OpenAI API key configured: {config.openai.api_key[:20]}...", "PASS")
        else:
            print_step("OpenAI API key missing or placeholder", "FAIL", 
                      "Set OPENAI_API_KEY in .env file")
            
        # Test Pinecone config
        if config.pinecone.api_key and config.pinecone.api_key != "your-pinecone-api-key-here":
            print_step(f"Pinecone API key configured: {config.pinecone.api_key[:8]}...", "PASS")
        else:
            print_step("Pinecone API key missing or placeholder", "FAIL",
                      "Set PINECONE_API_KEY in .env file")
        
        print_step(f"OpenAI model: {config.openai.model}", "PASS")
        print_step(f"Embedding model: {config.openai.embedding_model}", "PASS")
        print_step(f"Pinecone index: {config.pinecone.index_name}", "PASS")
        
        return True
        
    except Exception as e:
        print_step("Configuration test failed", "FAIL", str(e))
        return False

def test_openai():
    """Test OpenAI API connectivity."""
    print_header("Testing OpenAI Connection")
    
    try:
        from openai import OpenAI
        from src.rag_engine.config import config
        
        if not config.openai.api_key or config.openai.api_key == "sk-your-openai-api-key-here":
            print_step("Skipping OpenAI test - API key not configured", "WARN")
            return False
            
        client = OpenAI(api_key=config.openai.api_key)
        print_step("OpenAI client initialized", "PASS")
        
        # Test embedding
        print_step("Testing embedding generation...", "")
        response = client.embeddings.create(
            model=config.openai.embedding_model,
            input="test embedding"
        )
        
        print_step("Embedding request successful", "PASS", 
                  f"Dimension: {len(response.data[0].embedding)}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print_step("OpenAI connection failed", "FAIL", error_msg)
        
        if "api key" in error_msg.lower():
            print_step("Diagnosis: Invalid or expired API key", "WARN",
                      "Get new key from https://platform.openai.com/api-keys")
        elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
            print_step("Diagnosis: Billing or quota issue", "WARN",
                      "Check https://platform.openai.com/account/billing")
        
        return False

def test_pinecone():
    """Test Pinecone API connectivity."""
    print_header("Testing Pinecone Connection")
    
    try:
        from pinecone import Pinecone
        from src.rag_engine.config import config
        
        if not config.pinecone.api_key or config.pinecone.api_key == "your-pinecone-api-key-here":
            print_step("Skipping Pinecone test - API key not configured", "WARN")
            return False
            
        pc = Pinecone(api_key=config.pinecone.api_key)
        print_step("Pinecone client initialized", "PASS")
        
        # Test listing indexes
        print_step("Fetching existing indexes...", "")
        indexes = pc.list_indexes()
        
        print_step("Pinecone connection successful", "PASS", 
                  f"Found {len(indexes)} existing indexes")
        
        # Check for target index
        target_exists = any(idx.name == config.pinecone.index_name for idx in indexes)
        if target_exists:
            print_step(f"Target index '{config.pinecone.index_name}' exists", "PASS")
        else:
            print_step(f"Target index '{config.pinecone.index_name}' will be created", "PASS")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print_step("Pinecone connection failed", "FAIL", error_msg)
        
        if "api key" in error_msg.lower():
            print_step("Diagnosis: Invalid API key", "WARN",
                      "Get key from https://app.pinecone.io/")
        elif "free plan" in error_msg.lower():
            print_step("Diagnosis: Free plan limitation", "WARN",
                      "Check usage at https://app.pinecone.io/")
        
        return False

def test_knowledge_base():
    """Test knowledge base files."""
    print_header("Testing Knowledge Base")
    
    kb_path = Path("data/knowledge_base")
    
    if not kb_path.exists():
        print_step("Knowledge base directory missing", "FAIL", str(kb_path))
        return False
    
    print_step("Knowledge base directory exists", "PASS")
    
    md_files = list(kb_path.glob("*.md"))
    if not md_files:
        print_step("No markdown files found", "FAIL", "Expected FAQ, return policy, etc.")
        return False
    
    print_step(f"Found {len(md_files)} knowledge base files", "PASS")
    
    for md_file in md_files:
        size = md_file.stat().st_size
        if size > 0:
            print_step(f"{md_file.name}: {size:,} bytes", "PASS")
        else:
            print_step(f"{md_file.name}: Empty file", "WARN")
    
    return True

def main():
    """Run complete system test."""
    print("RAG Support System - Complete Diagnostic Test")
    
    # Track test results
    results = {
        "imports": test_imports(),
        "config": test_configuration(),
        "openai": test_openai(),
        "pinecone": test_pinecone(),
        "knowledge_base": test_knowledge_base()
    }
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print_step(f"{test_name.replace('_', ' ').title()}", status)
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print_step("ALL TESTS PASSED! The system should work correctly.", "PASS")
        print_step("You can now run: python run_demo.py", "")
    else:
        print_step("Some tests failed. Please fix the issues above.", "FAIL")
        print_step("Run this test again after making fixes: python test_system.py", "")
    
    print("=" * 70)

if __name__ == "__main__":
    main()