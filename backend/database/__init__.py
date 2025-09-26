"""
🗄️ AgriPal Database Package
Database models, connections, and services for the AgriPal system.
"""

from .models import (
    User,
    ChatSession,
    SessionMessage,
    AnalysisReport,
    KnowledgeDocument,
    IngestionLog,
    RAGQueryAnalytics,
    ChromaDBSyncStatus
)

from .connection import DatabaseManager, get_database_manager
from .services import (
    UserService,
    SessionService,
    KnowledgeService,
    AnalyticsService
)

__all__ = [
    # Models
    "User",
    "ChatSession", 
    "SessionMessage",
    "AnalysisReport",
    "KnowledgeDocument",
    "IngestionLog",
    "RAGQueryAnalytics",
    "ChromaDBSyncStatus",
    
    # Database Management
    "DatabaseManager",
    "get_database_manager",
    
    # Services
    "UserService",
    "SessionService", 
    "KnowledgeService",
    "AnalyticsService"
]
