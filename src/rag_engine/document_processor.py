"""
Document Processing Pipeline

Handles document ingestion, text chunking, metadata extraction, and preprocessing
for the RAG knowledge base system.
"""

import os
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import structlog
from .config import config

logger = structlog.get_logger(__name__)


class DocumentProcessor:
    """Process documents for RAG ingestion with chunking and metadata extraction."""
    
    def __init__(self):
        """Initialize document processor with configuration."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        logger.info("DocumentProcessor initialized", 
                   chunk_size=config.rag.chunk_size,
                   chunk_overlap=config.rag.chunk_overlap)
    
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single file into chunks with metadata.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of processed chunks with text and metadata
        """
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read file content
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata
            base_metadata = self._extract_file_metadata(file_path_obj)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create processed chunks with metadata
            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_id": self._generate_chunk_id(file_path, i),
                    "character_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                }
                
                processed_chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": chunk_metadata
                })
            
            logger.info("File processed successfully", 
                       file=file_path, 
                       chunks_created=len(processed_chunks))
            
            return processed_chunks
            
        except Exception as e:
            logger.error("Failed to process file", file=file_path, error=str(e))
            raise
    
    def process_directory(self, directory_path: str, 
                         file_extensions: List[str] = ['.md', '.txt']) -> List[Dict[str, Any]]:
        """
        Process all files in a directory into chunks.
        
        Args:
            directory_path: Path to directory containing files
            file_extensions: List of file extensions to process
            
        Returns:
            List of all processed chunks from all files
        """
        try:
            directory_path_obj = Path(directory_path)
            
            if not directory_path_obj.exists():
                raise FileNotFoundError(f"Directory not found: {directory_path}")
            
            all_chunks = []
            processed_files = 0
            
            # Process all files with matching extensions
            for file_path in directory_path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                    try:
                        file_chunks = self.process_file(str(file_path))
                        all_chunks.extend(file_chunks)
                        processed_files += 1
                    except Exception as e:
                        logger.warning("Failed to process file", 
                                     file=str(file_path), 
                                     error=str(e))
            
            logger.info("Directory processing completed", 
                       directory=directory_path,
                       files_processed=processed_files,
                       total_chunks=len(all_chunks))
            
            return all_chunks
            
        except Exception as e:
            logger.error("Failed to process directory", 
                        directory=directory_path, 
                        error=str(e))
            raise
    
    def process_text_content(self, text: str, 
                           source_name: str = "manual_input",
                           document_type: str = "text") -> List[Dict[str, Any]]:
        """
        Process raw text content into chunks.
        
        Args:
            text: Raw text content to process
            source_name: Name identifier for the source
            document_type: Type of document (e.g., 'faq', 'policy', 'manual')
            
        Returns:
            List of processed chunks with metadata
        """
        try:
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create base metadata
            base_metadata = {
                "source": source_name,
                "document_type": document_type,
                "file_extension": "txt",
                "processing_timestamp": self._get_timestamp()
            }
            
            # Create processed chunks
            processed_chunks = []
            for i, chunk_text in enumerate(chunks):
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_id": self._generate_chunk_id(source_name, i),
                    "character_count": len(chunk_text),
                    "word_count": len(chunk_text.split())
                }
                
                processed_chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": chunk_metadata
                })
            
            logger.info("Text content processed successfully", 
                       source=source_name,
                       chunks_created=len(processed_chunks))
            
            return processed_chunks
            
        except Exception as e:
            logger.error("Failed to process text content", 
                        source=source_name,
                        error=str(e))
            raise
    
    def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from file path and properties."""
        stat = file_path.stat()
        
        # Determine document type from file path/name
        document_type = self._infer_document_type(file_path)
        
        return {
            "source": file_path.name,
            "full_path": str(file_path),
            "document_type": document_type,
            "file_extension": file_path.suffix.lower(),
            "file_size": stat.st_size,
            "created_timestamp": int(stat.st_ctime),
            "modified_timestamp": int(stat.st_mtime),
            "processing_timestamp": self._get_timestamp()
        }
    
    def _infer_document_type(self, file_path: Path) -> str:
        """Infer document type from file name and path."""
        name_lower = file_path.name.lower()
        path_lower = str(file_path).lower()
        
        # Check for common document types
        if any(keyword in name_lower for keyword in ['faq', 'question', 'q&a']):
            return "faq"
        elif any(keyword in name_lower for keyword in ['policy', 'terms', 'condition']):
            return "policy"
        elif any(keyword in name_lower for keyword in ['return', 'refund', 'exchange']):
            return "return_policy"
        elif any(keyword in name_lower for keyword in ['shipping', 'delivery', 'ship']):
            return "shipping_info"
        elif any(keyword in name_lower for keyword in ['product', 'catalog', 'item']):
            return "product_info"
        elif any(keyword in name_lower for keyword in ['support', 'help', 'guide']):
            return "support_guide"
        else:
            return "general"
    
    def _generate_chunk_id(self, source: str, chunk_index: int) -> str:
        """Generate unique chunk identifier."""
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        timestamp = self._get_timestamp()
        return f"{source_hash}_{timestamp}_{chunk_index}"
    
    def _get_timestamp(self) -> int:
        """Get current timestamp."""
        import time
        return int(time.time())
    
    def validate_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate processed chunks for quality and completeness.
        
        Args:
            chunks: List of processed chunks to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if not chunks:
            issues.append("No chunks provided")
            return False, issues
        
        for i, chunk in enumerate(chunks):
            # Check required fields
            if "text" not in chunk:
                issues.append(f"Chunk {i}: Missing 'text' field")
            elif not chunk["text"].strip():
                issues.append(f"Chunk {i}: Empty text content")
            
            if "metadata" not in chunk:
                issues.append(f"Chunk {i}: Missing 'metadata' field")
            else:
                metadata = chunk["metadata"]
                required_fields = ["source", "chunk_index", "chunk_id"]
                for field in required_fields:
                    if field not in metadata:
                        issues.append(f"Chunk {i}: Missing metadata field '{field}'")
            
            # Check text quality
            if "text" in chunk:
                text = chunk["text"]
                if len(text) < 10:
                    issues.append(f"Chunk {i}: Text too short (less than 10 characters)")
                elif len(text) > config.rag.chunk_size * 2:
                    issues.append(f"Chunk {i}: Text too long (exceeds 2x chunk size)")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning("Chunk validation failed", 
                          issues_count=len(issues),
                          total_chunks=len(chunks))
        else:
            logger.info("Chunk validation passed", 
                       total_chunks=len(chunks))
        
        return is_valid, issues