"""
🗄️ AgriPal Database Services
High-level database operations and business logic for the AgriPal system.
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from .models import (
    User, ChatSession, SessionMessage, AnalysisReport,
    KnowledgeDocument, IngestionLog, RAGQueryAnalytics, ChromaDBSyncStatus
)
from .connection import DatabaseManager

logger = logging.getLogger(__name__)

class UserService:
    """Service for user-related database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_user(
        self, 
        email: str, 
        name: Optional[str] = None,
        location: Optional[Dict[str, Any]] = None,
        farm_details: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> User:
        """Create a new user"""
        async with self.db_manager.get_session() as session:
            user = User(
                email=email,
                name=name,
                location=location,
                farm_details=farm_details,
                preferences=preferences
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            logger.info(f"✅ Created user: {user.email}")
            return user
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            return result.scalar_one_or_none()
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.email == email)
            )
            return result.scalar_one_or_none()
    
    async def update_user(
        self, 
        user_id: UUID, 
        **updates
    ) -> Optional[User]:
        """Update user information"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                for key, value in updates.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                
                await session.commit()
                await session.refresh(user)
                logger.info(f"✅ Updated user: {user.email}")
                return user
            
            return None
    
    async def delete_user(self, user_id: UUID) -> bool:
        """Delete user and all associated data"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if user:
                await session.delete(user)
                await session.commit()
                logger.info(f"✅ Deleted user: {user.email}")
                return True
            
            return False

class SessionService:
    """Service for chat session-related database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_session(
        self, 
        user_id: UUID,
        session_data: Optional[Dict[str, Any]] = None
    ) -> ChatSession:
        """Create a new chat session"""
        async with self.db_manager.get_session() as session:
            chat_session = ChatSession(
                user_id=user_id,
                session_data=session_data
            )
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            logger.info(f"✅ Created chat session: {chat_session.id}")
            return chat_session
    
    async def get_session_by_id(self, session_id: UUID) -> Optional[ChatSession]:
        """Get chat session by ID with messages"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(ChatSession)
                .options(selectinload(ChatSession.messages))
                .where(ChatSession.id == session_id)
            )
            return result.scalar_one_or_none()
    
    async def get_user_sessions(
        self, 
        user_id: UUID, 
        limit: int = 50,
        offset: int = 0
    ) -> List[ChatSession]:
        """Get user's chat sessions"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.started_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()
    
    async def add_message(
        self,
        session_id: UUID,
        message_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionMessage:
        """Add message to chat session"""
        async with self.db_manager.get_session() as session:
            message = SessionMessage(
                session_id=session_id,
                message_type=message_type,
                content=content,
                message_metadata=metadata
            )
            session.add(message)
            await session.commit()
            await session.refresh(message)
            logger.info(f"✅ Added message to session: {session_id}")
            return message

    async def get_recent_messages(
        self,
        session_id: UUID,
        limit: int = 10
    ) -> List[SessionMessage]:
        """Return most recent messages for a session (descending time)."""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(SessionMessage)
                .where(SessionMessage.session_id == session_id)
                .order_by(SessionMessage.timestamp.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def end_session(
        self, 
        session_id: UUID, 
        summary: Optional[str] = None
    ) -> Optional[ChatSession]:
        """End a chat session"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            chat_session = result.scalar_one_or_none()
            
            if chat_session:
                chat_session.ended_at = datetime.utcnow()
                if summary:
                    chat_session.summary = summary
                
                await session.commit()
                await session.refresh(chat_session)
                logger.info(f"✅ Ended session: {session_id}")
                return chat_session
            
            return None
    
    async def create_analysis_report(
        self,
        session_id: UUID,
        image_url: Optional[str] = None,
        analysis_results: Optional[Dict[str, Any]] = None,
        recommendations: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None
    ) -> AnalysisReport:
        """Create analysis report for session"""
        async with self.db_manager.get_session() as session:
            report = AnalysisReport(
                session_id=session_id,
                image_url=image_url,
                analysis_results=analysis_results,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            session.add(report)
            await session.commit()
            await session.refresh(report)
            logger.info(f"✅ Created analysis report: {report.id}")
            return report

class KnowledgeService:
    """Service for knowledge base-related database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_document(
        self,
        source: str,
        title: Optional[str] = None,
        document_type: Optional[str] = None,
        crop_type: Optional[str] = None,
        region: Optional[str] = None,
        language: str = 'en',
        chunks_count: Optional[int] = None,
        file_size: Optional[int] = None,
        file_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> KnowledgeDocument:
        """Create a new knowledge document"""
        async with self.db_manager.get_session() as session:
            document = KnowledgeDocument(
                source=source,
                title=title,
                document_type=document_type,
                crop_type=crop_type,
                region=region,
                language=language,
                chunks_count=chunks_count,
                file_size=file_size,
                file_hash=file_hash,
                document_metadata=metadata
            )
            session.add(document)
            await session.commit()
            await session.refresh(document)
            logger.info(f"✅ Created knowledge document: {document.id}")
            return document
    
    async def get_documents_by_crop_type(
        self, 
        crop_type: str,
        limit: int = 50
    ) -> List[KnowledgeDocument]:
        """Get documents by crop type"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(KnowledgeDocument)
                .where(KnowledgeDocument.crop_type == crop_type)
                .order_by(KnowledgeDocument.ingestion_date.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def get_documents_by_region(
        self, 
        region: str,
        limit: int = 50
    ) -> List[KnowledgeDocument]:
        """Get documents by region"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(KnowledgeDocument)
                .where(KnowledgeDocument.region == region)
                .order_by(KnowledgeDocument.ingestion_date.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def search_documents(
        self,
        query: str,
        crop_type: Optional[str] = None,
        region: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 20
    ) -> List[KnowledgeDocument]:
        """Search documents with filters"""
        async with self.db_manager.get_session() as session:
            stmt = select(KnowledgeDocument)
            
            # Add text search
            if query:
                stmt = stmt.where(
                    or_(
                        KnowledgeDocument.title.ilike(f"%{query}%"),
                        KnowledgeDocument.source.ilike(f"%{query}%")
                    )
                )
            
            # Add filters
            if crop_type:
                stmt = stmt.where(KnowledgeDocument.crop_type == crop_type)
            if region:
                stmt = stmt.where(KnowledgeDocument.region == region)
            if document_type:
                stmt = stmt.where(KnowledgeDocument.document_type == document_type)
            
            stmt = stmt.order_by(KnowledgeDocument.ingestion_date.desc()).limit(limit)
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def log_ingestion(
        self,
        document_id: UUID,
        source: str,
        status: str,
        chunks_created: Optional[int] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None
    ) -> IngestionLog:
        """Log document ingestion"""
        async with self.db_manager.get_session() as session:
            log = IngestionLog(
                document_id=document_id,
                source=source,
                status=status,
                chunks_created=chunks_created,
                error_message=error_message,
                processing_time_ms=processing_time_ms
            )
            session.add(log)
            await session.commit()
            await session.refresh(log)
            logger.info(f"✅ Logged ingestion: {log.id}")
            return log

class AnalyticsService:
    """Service for analytics-related database operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def log_rag_query(
        self,
        session_id: UUID,
        query_text: str,
        query_embedding_model: str,
        retrieved_chunks: Optional[Dict[str, Any]] = None,
        response_quality_score: Optional[float] = None,
        processing_time_ms: Optional[int] = None
    ) -> RAGQueryAnalytics:
        """Log RAG query analytics"""
        async with self.db_manager.get_session() as session:
            analytics = RAGQueryAnalytics(
                session_id=session_id,
                query_text=query_text,
                query_embedding_model=query_embedding_model,
                retrieved_chunks=retrieved_chunks,
                response_quality_score=response_quality_score,
                processing_time_ms=processing_time_ms
            )
            session.add(analytics)
            await session.commit()
            await session.refresh(analytics)
            logger.info(f"✅ Logged RAG query analytics: {analytics.id}")
            return analytics
    
    async def get_session_analytics(
        self, 
        session_id: UUID
    ) -> List[RAGQueryAnalytics]:
        """Get analytics for a session"""
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                select(RAGQueryAnalytics)
                .where(RAGQueryAnalytics.session_id == session_id)
                .order_by(RAGQueryAnalytics.created_at.desc())
            )
            return result.scalars().all()
    
    async def get_quality_metrics(
        self,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get quality metrics for the last N days"""
        async with self.db_manager.get_session() as session:
            # Calculate date threshold
            threshold_date = datetime.utcnow() - timedelta(days=days)
            
            # Get average quality score
            result = await session.execute(
                select(func.avg(RAGQueryAnalytics.response_quality_score))
                .where(
                    and_(
                        RAGQueryAnalytics.created_at >= threshold_date,
                        RAGQueryAnalytics.response_quality_score.isnot(None)
                    )
                )
            )
            avg_quality = result.scalar() or 0.0
            
            # Get total queries
            result = await session.execute(
                select(func.count(RAGQueryAnalytics.id))
                .where(RAGQueryAnalytics.created_at >= threshold_date)
            )
            total_queries = result.scalar() or 0
            
            # Get average processing time
            result = await session.execute(
                select(func.avg(RAGQueryAnalytics.processing_time_ms))
                .where(
                    and_(
                        RAGQueryAnalytics.created_at >= threshold_date,
                        RAGQueryAnalytics.processing_time_ms.isnot(None)
                    )
                )
            )
            avg_processing_time = result.scalar() or 0.0
            
            return {
                "avg_quality_score": round(avg_quality, 3),
                "total_queries": total_queries,
                "avg_processing_time_ms": round(avg_processing_time, 2),
                "period_days": days
            }
    
    async def update_chromadb_sync_status(
        self,
        collection_name: str,
        sync_status: str,
        total_objects: Optional[int] = None,
        error_details: Optional[str] = None
    ) -> ChromaDBSyncStatus:
        """Update ChromaDB vector database sync status"""
        async with self.db_manager.get_session() as session:
            # Check if record exists
            result = await session.execute(
                select(ChromaDBSyncStatus)
                .where(ChromaDBSyncStatus.collection_name == collection_name)
            )
            sync_status_record = result.scalar_one_or_none()
            
            if sync_status_record:
                # Update existing record
                sync_status_record.sync_status = sync_status
                sync_status_record.last_sync_at = datetime.utcnow()
                if total_objects is not None:
                    sync_status_record.total_objects = total_objects
                if error_details is not None:
                    sync_status_record.error_details = error_details
            else:
                # Create new record
                sync_status_record = ChromaDBSyncStatus(
                    collection_name=collection_name,
                    sync_status=sync_status,
                    total_objects=total_objects,
                    error_details=error_details
                )
                session.add(sync_status_record)
            
            await session.commit()
            await session.refresh(sync_status_record)
            logger.info(f"✅ Updated ChromaDB sync status: {collection_name}")
            return sync_status_record
