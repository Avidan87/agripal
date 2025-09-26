"""
📊 AgriPal Data Models
Pydantic models for request/response validation and data structures.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import uuid

# Enums
class AgentType(str, Enum):
    """Types of AI agents in the system"""
    PERCEPTION = "perception"
    KNOWLEDGE = "knowledge"
    EMAIL = "email"

class MessageType(str, Enum):
    """Types of chat messages"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class AnalysisStatus(str, Enum):
    """Status of analysis requests"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class CropType(str, Enum):
    """Supported crop types"""
    RICE = "rice"
    WHEAT = "wheat"
    CORN = "corn"
    SOYBEANS = "soybeans"
    COTTON = "cotton"
    TOMATOES = "tomatoes"
    POTATOES = "potatoes"
    GENERAL = "general"

class SeverityLevel(str, Enum):
    """Severity levels for crop issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Base Models
class BaseAgriPalModel(BaseModel):
    """Base model with common fields"""
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)

# Request Models
class AnalysisRequest(BaseModel):
    """Request model for crop analysis"""
    message: str = Field(..., min_length=1, max_length=2000, description="User's agricultural query")
    crop_type: Optional[CropType] = Field(None, description="Type of crop being analyzed")
    location: Optional[str] = Field(None, description="Farm location (city, state)")
    image_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional image metadata")
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Message cannot be empty")
        return v.strip()

class SessionCreateRequest(BaseModel):
    """Request to create a new session"""
    user_email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    user_name: Optional[str] = Field(None, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    farm_details: Optional[Dict[str, Any]] = Field(None)

# Response Models
class ImageAnalysisResult(BaseModel):
    """Results from perception agent image analysis"""
    detected_issues: List[str] = Field(default_factory=list)
    crop_health_score: float = Field(..., ge=0.0, le=100.0)
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)
    severity: SeverityLevel = Field(default=SeverityLevel.LOW)
    analysis_text: Optional[str] = Field(None, description="Formatted analysis text from GPT-4o for direct display")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeSearchResult(BaseModel):
    """Results from knowledge agent RAG search"""
    relevant_documents: List[Dict[str, Any]] = Field(default_factory=list)
    search_query: str = Field(...)
    confidence_scores: List[float] = Field(default_factory=list)
    source_types: List[str] = Field(default_factory=list)
    contextual_advice: str = Field(...)

class WeatherContext(BaseModel):
    """Weather information for contextual recommendations"""
    current_temperature: float = Field(...)
    humidity: float = Field(..., ge=0.0, le=100.0)
    precipitation: float = Field(..., ge=0.0)
    forecast_days: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class AnalysisResponse(BaseModel):
    """Main response from crop analysis"""
    session_id: str = Field(...)
    response_text: str = Field(...)
    image_analysis: Optional[ImageAnalysisResult] = None
    knowledge_results: Optional[KnowledgeSearchResult] = None
    weather_context: Optional[WeatherContext] = None
    recommendations: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int = Field(..., ge=0)
    status: AnalysisStatus = Field(default=AnalysisStatus.COMPLETED)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = Field(...)
    timestamp: datetime = Field(...)
    checks: Dict[str, bool] = Field(...)
    version: str = Field(...)
    error: Optional[str] = None

# Agent Communication Models
class AgentMessage(BaseModel):
    """Message format for inter-agent communication"""
    agent_type: AgentType = Field(...)
    session_id: str = Field(...)
    content: Dict[str, Any] = Field(...)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentResponse(BaseModel):
    """Response format from agents"""
    agent_type: AgentType = Field(...)
    session_id: str = Field(...)
    success: bool = Field(...)
    result: Dict[str, Any] = Field(...)
    error: Optional[str] = None
    processing_time_ms: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Session Management Models
class UserSession(BaseModel):
    """User session information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    location: Optional[str] = None
    farm_details: Optional[Dict[str, Any]] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = Field(default=0)
    context: Dict[str, Any] = Field(default_factory=dict)
    is_admin: bool = Field(default=False)
    
    def is_session_complete(self) -> bool:
        """Check if session should be considered complete"""
        return self.message_count >= 5  # Example threshold

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()

class SessionMessage(BaseModel):
    """Individual message within a session"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(...)
    message_type: MessageType = Field(...)
    content: str = Field(...)
    image_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Email Report Models
class EmailReportData(BaseModel):
    """Data for email report generation"""
    session_id: str = Field(...)
    user_email: str = Field(...)
    user_name: Optional[str] = None
    session_summary: str = Field(...)
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    images_analyzed: int = Field(default=0)
    weather_data: Optional[WeatherContext] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)

class EmailSendResult(BaseModel):
    """Result of email sending operation"""
    success: bool = Field(...)
    message_id: Optional[str] = None
    recipient: str = Field(...)
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Database Models (for ORM integration)
class UserProfile(BaseAgriPalModel):
    """User profile information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str = Field(...)
    name: Optional[str] = None
    location: Optional[Dict[str, Any]] = None
    farm_details: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    is_active: bool = Field(default=True)

class ChatSession(BaseAgriPalModel):
    """Chat session record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(...)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    session_data: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None
    total_messages: int = Field(default=0)

class AnalysisReport(BaseAgriPalModel):
    """Analysis report record"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = Field(...)
    image_url: Optional[str] = None
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)

# Validation helpers
def validate_session_id(session_id: str) -> bool:
    """Validate session ID format"""
    try:
        uuid.UUID(session_id)
        return True
    except ValueError:
        return False

def validate_image_size(file_size: int, max_size: int = 10 * 1024 * 1024) -> bool:
    """Validate image file size"""
    return 0 < file_size <= max_size

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate geographic coordinates"""
    return -90 <= lat <= 90 and -180 <= lon <= 180

# Export all models
__all__ = [
    # Enums
    "AgentType", "MessageType", "AnalysisStatus", "CropType", "SeverityLevel",
    
    # Request Models
    "AnalysisRequest", "SessionCreateRequest",
    
    # Response Models
    "ImageAnalysisResult", "KnowledgeSearchResult", "WeatherContext", 
    "AnalysisResponse", "HealthCheckResponse",
    
    # Agent Models
    "AgentMessage", "AgentResponse",
    
    # Session Models
    "UserSession", "SessionMessage",
    
    # Email Models
    "EmailReportData", "EmailSendResult",
    
    # Database Models
    "UserProfile", "ChatSession", "AnalysisReport",
    
    # Validators
    "validate_session_id", "validate_image_size", "validate_coordinates"
]

