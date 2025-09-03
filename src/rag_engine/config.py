"""
Configuration Management for RAG Support System

Handles environment variables, settings validation, and configuration defaults
for all system components.
"""

import os
from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration settings."""
    
    api_key: str = Field(default="", env="OPENAI_API_KEY")
    model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-ada-002", env="OPENAI_EMBEDDING_MODEL")
    max_tokens: int = Field(default=1000, env="OPENAI_MAX_TOKENS")
    temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v):
        # Allow empty during import, validate when actually used
        if v and v == "sk-your-openai-api-key-here":
            raise ValueError("OpenAI API key placeholder must be replaced with real key")
        return v
    
    def validate_for_use(self):
        """Validate config is ready for actual use."""
        if not self.api_key or self.api_key == "sk-your-openai-api-key-here":
            raise ValueError("OpenAI API key must be provided")


class PineconeConfig(BaseSettings):
    """Pinecone vector database configuration settings."""
    
    api_key: str = Field(default="", env="PINECONE_API_KEY")
    index_name: str = Field(default="rag-support-system", env="PINECONE_INDEX_NAME")
    dimension: int = Field(default=1536, env="PINECONE_DIMENSION")
    
    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v):
        # Allow empty during import, validate when actually used
        if v and v == "your-pinecone-api-key-here":
            raise ValueError("Pinecone API key placeholder must be replaced with real key")
        return v
    
    def validate_for_use(self):
        """Validate config is ready for actual use."""
        if not self.api_key or self.api_key == "your-pinecone-api-key-here":
            raise ValueError("Pinecone API key must be provided")


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    url: Optional[str] = Field(default=None, env="DATABASE_URL")
    host: str = Field(default="localhost", env="POSTGRES_HOST")
    port: int = Field(default=5432, env="POSTGRES_PORT")
    user: str = Field(default="rag_user", env="POSTGRES_USER")
    password: str = Field(default="", env="POSTGRES_PASSWORD")
    database: str = Field(default="rag_support_system", env="POSTGRES_DB")
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        if self.url:
            return self.url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class RAGConfig(BaseSettings):
    """RAG engine configuration settings."""
    
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    confidence_threshold_high: float = Field(default=0.8, env="CONFIDENCE_THRESHOLD_HIGH")
    confidence_threshold_low: float = Field(default=0.6, env="CONFIDENCE_THRESHOLD_LOW")
    max_search_results: int = Field(default=5, env="MAX_SEARCH_RESULTS")
    similarity_top_k: int = Field(default=3, env="SIMILARITY_TOP_K")
    
    @field_validator("confidence_threshold_high")
    @classmethod
    def validate_high_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("confidence_threshold_low")
    @classmethod
    def validate_low_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        return v
    
    @model_validator(mode='after')
    def validate_threshold_order(self):
        if self.confidence_threshold_low >= self.confidence_threshold_high:
            raise ValueError("Low confidence threshold must be less than high threshold")
        return self


class FlaskConfig(BaseSettings):
    """Flask application configuration settings."""
    
    env: str = Field(default="development", env="FLASK_ENV")
    debug: bool = Field(default=True, env="FLASK_DEBUG")
    secret_key: str = Field(default="dev-secret-key", env="SECRET_KEY")
    host: str = Field(default="0.0.0.0", env="FLASK_HOST")
    port: int = Field(default=8000, env="FLASK_PORT")


class BusinessConfig(BaseSettings):
    """Business logic configuration settings."""
    
    auto_response_enabled: bool = Field(default=True, env="AUTO_RESPONSE_ENABLED")
    escalation_enabled: bool = Field(default=True, env="ESCALATION_ENABLED")
    conversation_history_enabled: bool = Field(default=True, env="CONVERSATION_HISTORY_ENABLED")


class Config:
    """Main configuration class that combines all settings."""
    
    def __init__(self):
        self.openai = OpenAIConfig()
        self.pinecone = PineconeConfig()
        self.database = DatabaseConfig()
        self.rag = RAGConfig()
        self.flask = FlaskConfig()
        self.business = BusinessConfig()
    
    def validate(self) -> bool:
        """Validate all configuration settings for actual use."""
        try:
            # Test that all required settings are properly configured
            self.openai.validate_for_use()
            self.pinecone.validate_for_use()
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global configuration instance
config = Config()