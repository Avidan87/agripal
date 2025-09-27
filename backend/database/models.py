"""
🗄️ AgriPal Database Models
SQLAlchemy models for the AgriPal agricultural assistant system.
"""
from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime, 
    ForeignKey, JSON, BigInteger, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func, text
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """Farmer user profiles with location and farm details"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255))
    location = Column(JSON)  # {"country": "USA", "state": "California", "city": "Fresno"}
    farm_details = Column(JSON)  # {"farm_size": "100 acres", "crop_types": ["wheat", "corn"]}
    preferences = Column(JSON)  # {"language": "en", "notifications": true, "units": "metric"}
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    updated_at = Column(DateTime(timezone=True), server_default=text('NOW()'), onupdate=text('NOW()'))
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', name='{self.name}')>"

class ChatSession(Base):
    """Individual consultation sessions between farmers and AI"""
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), server_default=text('NOW()'), index=True)
    ended_at = Column(DateTime(timezone=True), index=True)
    session_data = Column(JSON)  # Additional session metadata
    summary = Column(Text)  # AI-generated session summary
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    updated_at = Column(DateTime(timezone=True), server_default=text('NOW()'), onupdate=text('NOW()'))
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("SessionMessage", back_populates="session", cascade="all, delete-orphan")
    analysis_reports = relationship("AnalysisReport", back_populates="session", cascade="all, delete-orphan")
    rag_analytics = relationship("RAGQueryAnalytics", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, user_id={self.user_id}, started_at='{self.started_at}')>"

class SessionMessage(Base):
    """Individual messages within chat sessions"""
    __tablename__ = "session_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    message_type = Column(String(50), nullable=False, index=True)  # 'user', 'assistant', 'system'
    content = Column(Text, nullable=False)
    message_metadata = Column(JSON)  # Message-specific metadata
    timestamp = Column(DateTime(timezone=True), server_default=text('NOW()'), index=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self):
        return f"<SessionMessage(id={self.id}, session_id={self.session_id}, type='{self.message_type}')>"

class AnalysisReport(Base):
    """AI analysis results from image and text processing"""
    __tablename__ = "analysis_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    image_url = Column(String(500))  # URL to stored image
    analysis_results = Column(JSON)  # AI analysis results
    recommendations = Column(JSON)  # Generated recommendations
    confidence_score = Column(Float)  # AI confidence score (0.0 - 1.0)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'), index=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="analysis_reports")
    
    def __repr__(self):
        return f"<AnalysisReport(id={self.id}, session_id={self.session_id}, confidence={self.confidence_score})>"

class KnowledgeDocument(Base):
    """Agricultural knowledge base documents and metadata"""
    __tablename__ = "knowledge_documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(255), nullable=False, index=True)  # 'FAO', 'Research Paper', 'Extension Guide'
    title = Column(String(500))
    document_type = Column(String(100), index=True)  # 'fao_manual', 'research_paper', 'extension_guide'
    crop_type = Column(String(100), index=True)  # 'wheat', 'corn', 'rice', etc.
    region = Column(String(100), index=True)  # 'global', 'USA', 'India', etc.
    language = Column(String(10), default='en')
    ingestion_date = Column(DateTime(timezone=True), server_default=text('NOW()'), index=True)
    chunks_count = Column(Integer)  # Number of text chunks created
    file_size = Column(BigInteger)  # File size in bytes
    file_hash = Column(String(64), index=True)  # SHA-256 hash for duplicate detection
    document_metadata = Column(JSON)  # Additional document metadata
    
    # Relationships
    ingestion_logs = relationship("IngestionLog", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<KnowledgeDocument(id={self.id}, source='{self.source}', title='{self.title}')>"

class IngestionLog(Base):
    """Logs of document ingestion and processing"""
    __tablename__ = "ingestion_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    source = Column(String(255))
    status = Column(String(50), index=True)  # 'success', 'failed', 'partial'
    ingested_at = Column(DateTime(timezone=True), server_default=text('NOW()'), index=True)
    chunks_created = Column(Integer)
    error_message = Column(Text)
    processing_time_ms = Column(Integer)
    
    # Relationships
    document = relationship("KnowledgeDocument", back_populates="ingestion_logs")
    
    def __repr__(self):
        return f"<IngestionLog(id={self.id}, document_id={self.document_id}, status='{self.status}')>"

class RAGQueryAnalytics(Base):
    """Analytics for RAG query performance and quality"""
    __tablename__ = "rag_query_analytics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    query_text = Column(Text)
    query_embedding_model = Column(String(100))  # 'text-embedding-3-large'
    retrieved_chunks = Column(JSON)  # Store retrieved document chunks
    response_quality_score = Column(Float, index=True)  # User feedback or auto-evaluation
    processing_time_ms = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'), index=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="rag_analytics")
    
    def __repr__(self):
        return f"<RAGQueryAnalytics(id={self.id}, session_id={self.session_id}, quality_score={self.response_quality_score})>"

class ChromaDBSyncStatus(Base):
    """Status tracking for ChromaDB vector database synchronization"""
    __tablename__ = "chromadb_sync_status"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_name = Column(String(100), index=True)
    last_sync_at = Column(DateTime(timezone=True), index=True)
    total_objects = Column(Integer)
    sync_status = Column(String(50), index=True)  # 'pending', 'syncing', 'completed', 'failed'
    error_details = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=text('NOW()'))
    updated_at = Column(DateTime(timezone=True), server_default=text('NOW()'), onupdate=text('NOW()'))
    
    def __repr__(self):
        return f"<ChromaDBSyncStatus(id={self.id}, collection='{self.collection_name}', status='{self.sync_status}')>"

# Additional indexes for performance
# Note: JSON columns cannot have regular B-tree indexes
# GIN indexes for JSON require gin_trgm_ops extension in PostgreSQL
# For now, removing JSON indexes to avoid extension dependencies
Index('idx_analysis_reports_confidence', AnalysisReport.confidence_score)
Index('idx_rag_query_analytics_quality_score', RAGQueryAnalytics.response_quality_score)
