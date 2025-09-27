"""
🗄️ AgriPal Database Connection Management
Async database connection pooling and management for PostgreSQL.
"""
import asyncio
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool

from ..config import settings

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError:
    logger.warning("⚠️ asyncpg not available, database functionality may be limited")
    asyncpg = None

class DatabaseManager:
    """Async database connection manager with connection pooling"""
    
    def __init__(self):
        self.database_url = settings.DATABASE_URL
        self.engine = None
        self.session_factory = None
        self._connection_pool = None
        self._is_initialized = False
        self._initialization_failed = False
    
    async def initialize(self):
        """Initialize database connection and session factory"""
        try:
            logger.info("🔄 Initializing database connection...")
            
            # Handle SSL configuration for production environments like Render
            connect_args = {}
            if settings.ENVIRONMENT == "production":
                connect_args = {
                    "ssl": "require",
                    "server_settings": {
                        "jit": "off"
                    }
                }
            
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.database_url,
                echo=settings.DATABASE_ECHO,
                poolclass=NullPool,  # Use asyncpg's connection pooling
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections every hour
                connect_args=connect_args
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            await self._test_connection()
            
            # Create tables if they don't exist (for production deployments)
            await self._ensure_tables_exist()
            
            self._is_initialized = True
            logger.info("✅ Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connection: {str(e)}")
            self._initialization_failed = True
            raise
    
    async def _test_connection(self):
        """Test database connection"""
        try:
            async with self.engine.begin() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection test passed")
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {str(e)}")
            raise
    
    async def _ensure_tables_exist(self):
        """Ensure database tables exist (create if missing)"""
        try:
            from .models import Base
            async with self.engine.begin() as conn:
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
            logger.info("✅ Database tables verified/created")
        except Exception as e:
            logger.warning(f"⚠️ Table creation check failed: {str(e)}")
            # Don't raise - this might fail in some environments but database can still work
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup"""
        if not self._is_initialized:
            await self.initialize()
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Database session error: {str(e)}")
            raise
        finally:
            await session.close()
    
    async def execute_raw_sql(self, query: str, *args):
        """Execute raw SQL query"""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            async with self.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text(query), args)
                return result
        except Exception as e:
            logger.error(f"❌ Raw SQL execution failed: {str(e)}")
            raise
    
    async def fetch_one(self, query: str, *args):
        """Fetch one row from database"""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            async with self.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text(query), args)
                return result.fetchone()
        except Exception as e:
            logger.error(f"❌ Fetch one failed: {str(e)}")
            raise
    
    async def fetch_all(self, query: str, *args):
        """Fetch all rows from database"""
        if not self._is_initialized:
            await self.initialize()
        
        try:
            async with self.engine.begin() as conn:
                from sqlalchemy import text
                result = await conn.execute(text(query), args)
                return result.fetchall()
        except Exception as e:
            logger.error(f"❌ Fetch all failed: {str(e)}")
            raise
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            if not self._is_initialized:
                return False
            
            async with self.engine.begin() as conn:
                from sqlalchemy import text
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"❌ Database health check failed: {str(e)}")
            return False
    
    def is_available(self) -> bool:
        """Check if database is available and initialized"""
        return self._is_initialized and not self._initialization_failed
    
    async def close(self):
        """Close database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
            self._is_initialized = False
            logger.info("🔌 Database connections closed")
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {str(e)}")

# Global database manager instance
_database_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _database_manager
    
    if _database_manager is None:
        _database_manager = DatabaseManager()
        try:
            await _database_manager.initialize()
        except Exception as e:
            logger.warning(f"⚠️ Database initialization failed, will use fallback storage: {str(e)}")
            # Mark initialization as failed but don't raise - allow fallback
            _database_manager._initialization_failed = True
            _database_manager._is_initialized = False
    
    return _database_manager

async def close_database():
    """Close global database connections"""
    global _database_manager
    
    if _database_manager:
        await _database_manager.close()
        _database_manager = None

# Dependency for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions"""
    db_manager = await get_database_manager()
    async with db_manager.get_session() as session:
        yield session
