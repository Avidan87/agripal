"""
⚙️ AgriPal Configuration Management
Centralized settings and environment variable management.
"""
import os
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # App Configuration
    APP_NAME: str = "AgriPal"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Auto-detect production environment
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Auto-detect if running on Render
        if os.getenv("RENDER") or "render.com" in os.getenv("HOST", ""):
            self.ENVIRONMENT = "production"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALLOWED_ORIGINS: str = "*"  # Changed to string for environment variable compatibility
    TRUSTED_HOSTS: str = "*.onrender.com,*.render.com,localhost,127.0.0.1"  # Fixed for Render deployment
    
    # Database Configuration
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agripal"
    DATABASE_ECHO: bool = False
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0
    
    # Object Storage (MinIO/S3)
    STORAGE_ENDPOINT: str = "http://localhost:9000"
    STORAGE_ACCESS_KEY: str = "agripal"
    STORAGE_SECRET_KEY: str = "agripal123"
    STORAGE_BUCKET: str = "agripal-storage"
    STORAGE_SECURE: bool = False
    
    # Vector Database (ChromaDB)
    CHROMA_DB_PATH: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "agricultural_knowledge"
    ENABLE_CHROMADB: bool = True
    USE_CHROMADB: bool = True
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.7
    # Optional lower-cost model for general chat/intent
    GENERAL_CHAT_MODEL: str = "gpt-4o-mini"
    
    # Hugging Face Configuration (for local generation)
    HF_TOKEN: str = ""  # Hugging Face token for dataset access
    HUGGINGFACE_MODEL: str = "meta-llama/Llama-2-7b-chat-hf"
    USE_LOCAL_GENERATION: bool = False
    
    # Weather API
    # Read WEATHER_API_KEY from env var WEATHER_API for compatibility with existing .env
    WEATHER_API_KEY: str = Field(default="", validation_alias="WEATHER_API")
    WEATHER_API_URL: str = "https://api.openweathermap.org/data/2.5"
    
    # Geocoding Services (OpenCage)
    OPENCAGE_API_KEY: str = ""
    OPENCAGE_API_URL: str = "https://api.opencagedata.com/geocode/v1"
    
    # Email Configuration (SendGrid)
    SENDGRID_API_KEY: str = ""
    SENDGRID_FROM_EMAIL: str = "noreply@agripal.com"
    SENDGRID_FROM_NAME: str = "AgriPal AI Assistant"
    FROM_EMAIL: str = "noreply@agripal.com"
    
    # File Upload Limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg", "image/bmp"]
    
    # Agent Configuration
    PERCEPTION_AGENT_ENABLED: bool = True
    KNOWLEDGE_AGENT_ENABLED: bool = True
    EMAIL_AGENT_ENABLED: bool = True
    
    # Session Configuration
    SESSION_TIMEOUT_MINUTES: int = 60
    MAX_SESSION_MESSAGES: int = 100
    
    # Authentication & Authorization
    ENABLE_AUTH: bool = False  # Set to True in production
    JWT_SECRET_KEY: str = "your-jwt-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 10
    RATE_LIMIT_PER_HOUR: int = 100
    ENABLE_RATE_LIMITING: bool = True
    
    # Monitoring & Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Background Tasks
    ENABLE_BACKGROUND_TASKS: bool = True
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env file

@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Using lru_cache to avoid re-reading environment variables
    """
    return Settings()

# Global settings instance
settings = get_settings()

# Environment-specific configurations
def get_database_url() -> str:
    """Get database URL with proper formatting"""
    if settings.ENVIRONMENT == "test":
        return settings.DATABASE_URL.replace("/agripal", "/agripal_test")
    return settings.DATABASE_URL

def is_production() -> bool:
    """Check if running in production environment"""
    return settings.ENVIRONMENT.lower() == "production"

def is_development() -> bool:
    """Check if running in development environment"""
    return settings.ENVIRONMENT.lower() == "development"

# Validation functions
def validate_openai_config() -> bool:
    """Validate OpenAI configuration"""
    if not settings.OPENAI_API_KEY:
        return False
    return True

def validate_storage_config() -> bool:
    """Validate storage configuration"""
    required_fields = [
        settings.STORAGE_ENDPOINT,
        settings.STORAGE_ACCESS_KEY,
        settings.STORAGE_SECRET_KEY,
        settings.STORAGE_BUCKET
    ]
    return all(required_fields)

def validate_email_config() -> bool:
    """Validate email configuration"""
    return bool(settings.SENDGRID_API_KEY)

# Export commonly used settings
__all__ = [
    "settings",
    "get_settings",
    "get_database_url",
    "is_production",
    "is_development",
    "validate_openai_config",
    "validate_storage_config",
    "validate_email_config"
]


