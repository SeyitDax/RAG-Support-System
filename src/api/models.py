"""
Data Models for API Requests and Responses

Pydantic models for request/response validation and serialization
in the RAG support system API.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class QueryRequest(BaseModel):
    """Model for customer query requests."""
    
    question: str = Field(..., min_length=1, max_length=1000, description="Customer question")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Number of results to retrieve")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    session_id: Optional[str] = Field(default=None, description="Optional session identifier")
    
    @field_validator("question")
    @classmethod
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or only whitespace")
        return v.strip()


class SourceInfo(BaseModel):
    """Model for source attribution information."""
    
    source: str = Field(..., description="Source document name")
    document_type: str = Field(..., description="Type of source document")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    chunk_count: Optional[int] = Field(default=1, description="Number of chunks from this source")


class QueryResponse(BaseModel):
    """Model for query response data."""
    
    success: bool = Field(..., description="Whether the query was successful")
    response: str = Field(..., description="Generated response text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents used")
    should_escalate: bool = Field(..., description="Whether to escalate to human agent")
    auto_response: bool = Field(..., description="Whether response can be sent automatically")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    timestamp: int = Field(..., description="Unix timestamp")
    retrieved_chunks: Optional[int] = Field(default=None, description="Number of chunks retrieved")
    error_type: Optional[str] = Field(default=None, description="Error type if failed")


class FeedbackRequest(BaseModel):
    """Model for user feedback on responses."""
    
    query_id: str = Field(..., description="ID of the original query")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_text: Optional[str] = Field(default=None, max_length=1000, description="Optional feedback text")
    user_id: Optional[str] = Field(default=None, description="Optional user identifier")
    helpful: Optional[bool] = Field(default=None, description="Whether response was helpful")
    accurate: Optional[bool] = Field(default=None, description="Whether response was accurate")


class FeedbackResponse(BaseModel):
    """Model for feedback submission response."""
    
    success: bool = Field(..., description="Whether feedback was recorded")
    message: str = Field(..., description="Response message")
    feedback_id: Optional[str] = Field(default=None, description="Generated feedback ID")


class IngestionRequest(BaseModel):
    """Model for document ingestion requests."""
    
    file_paths: Optional[List[str]] = Field(default=None, description="List of file paths to ingest")
    directory_path: Optional[str] = Field(default=None, description="Directory path to ingest")
    text_content: Optional[str] = Field(default=None, description="Raw text content to ingest")
    source_name: Optional[str] = Field(default="manual_input", description="Source name for text content")
    document_type: Optional[str] = Field(default="general", description="Document type classification")
    
    @model_validator(mode='after')
    def validate_at_least_one_source(self):
        # Check that at least one source is provided
        sources = [
            self.file_paths,
            self.directory_path, 
            self.text_content
        ]
        if not any(source for source in sources):
            raise ValueError("Must provide at least one of: file_paths, directory_path, or text_content")
        return self


class IngestionResponse(BaseModel):
    """Model for document ingestion response."""
    
    success: bool = Field(..., description="Whether ingestion was successful")
    message: str = Field(..., description="Status message")
    total_chunks: Optional[int] = Field(default=None, description="Number of chunks processed")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    failed_files: Optional[List[str]] = Field(default_factory=list, description="Files that failed to process")
    stats: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional statistics")


class HealthResponse(BaseModel):
    """Model for system health check response."""
    
    status: str = Field(..., description="System status")
    timestamp: int = Field(..., description="Check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    uptime: Optional[float] = Field(default=None, description="System uptime in seconds")
    components: Optional[Dict[str, str]] = Field(default_factory=dict, description="Component status")


class AnalyticsRequest(BaseModel):
    """Model for analytics data requests."""
    
    start_date: Optional[datetime] = Field(default=None, description="Start date for analytics")
    end_date: Optional[datetime] = Field(default=None, description="End date for analytics")
    metric_types: Optional[List[str]] = Field(default=None, description="Specific metrics to retrieve")
    user_id: Optional[str] = Field(default=None, description="Filter by user ID")
    
    @model_validator(mode='after')
    def validate_date_range(self):
        if self.end_date and self.start_date and self.end_date < self.start_date:
            raise ValueError("End date must be after start date")
        return self


class MetricData(BaseModel):
    """Model for individual metric data points."""
    
    metric_name: str = Field(..., description="Name of the metric")
    value: float = Field(..., description="Metric value")
    timestamp: int = Field(..., description="Timestamp of measurement")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metric metadata")


class AnalyticsResponse(BaseModel):
    """Model for analytics data response."""
    
    success: bool = Field(..., description="Whether analytics retrieval was successful")
    metrics: List[MetricData] = Field(default_factory=list, description="Retrieved metrics")
    summary: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Summary statistics")
    time_range: Optional[Dict[str, datetime]] = Field(default_factory=dict, description="Actual time range of data")


class SystemStatsResponse(BaseModel):
    """Model for system statistics response."""
    
    vector_store: Dict[str, Any] = Field(default_factory=dict, description="Vector store statistics")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="System configuration")
    performance: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Performance metrics")
    uptime: Optional[float] = Field(default=None, description="System uptime in seconds")


class ErrorResponse(BaseModel):
    """Model for error responses."""
    
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: int = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for debugging")


# Configuration models

class APIConfig(BaseModel):
    """Model for API configuration settings."""
    
    max_request_size: int = Field(default=1048576, description="Maximum request size in bytes")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per minute")
    rate_limit_per_hour: int = Field(default=1000, description="Rate limit per hour") 
    enable_cors: bool = Field(default=True, description="Enable CORS")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    log_level: str = Field(default="INFO", description="Logging level")


class BusinessRules(BaseModel):
    """Model for business logic configuration."""
    
    confidence_threshold_high: float = Field(default=0.8, ge=0.0, le=1.0, description="High confidence threshold")
    confidence_threshold_low: float = Field(default=0.6, ge=0.0, le=1.0, description="Low confidence threshold")
    auto_response_enabled: bool = Field(default=True, description="Enable automatic responses")
    escalation_enabled: bool = Field(default=True, description="Enable human escalation")
    max_query_length: int = Field(default=1000, description="Maximum query length")
    
    @model_validator(mode='after')
    def validate_threshold_order(self):
        if self.confidence_threshold_low >= self.confidence_threshold_high:
            raise ValueError("Low confidence threshold must be less than high threshold")
        return self