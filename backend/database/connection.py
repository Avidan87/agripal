"""
🗄️ AgriPal Database Connection Management
Async database connection pooling and management for PostgreSQL.
"""
import asyncio
import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from ..config import settings

logger = logging.getLogger(__name__)

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
                # For Render PostgreSQL, SSL is required
                # Ensure SSL is properly configured for Render
                if "sslmode" not in self.database_url and "ssl" not in self.database_url:
                    if "?" in self.database_url:
                        self.database_url += "&sslmode=require"
                    else:
                        self.database_url += "?sslmode=require"
                
                # Configure SSL settings for psycopg - Render requires SSL
                connect_args = {
                    "sslmode": "require",  # Force SSL connection for Render
                    "server_settings": {
                        "jit": "off",
                        "application_name": "agripal_backend"
                    }
                }
                
                # Render-specific SSL configuration
                if "render.com" in self.database_url or "onrender.com" in self.database_url:
                    logger.info("🔒 Configuring SSL for Render PostgreSQL connection")
                    # Use simple SSL configuration that works with Render
                    connect_args = {
                        "sslmode": "require",  # Render PostgreSQL requires SSL
                        "server_settings": {
                            "jit": "off",
                            "application_name": "agripal_backend"
                        }
                    }
            
            # Log the database URL for debugging (without password)
            safe_url = self.database_url
            if "://" in safe_url and "@" in safe_url:
                # Hide password in logs
                parts = safe_url.split("://")
                if len(parts) == 2:
                    protocol, rest = parts
                    if "@" in rest:
                        user_pass, host_db = rest.split("@", 1)
                        if ":" in user_pass:
                            user, _ = user_pass.split(":", 1)
                            safe_url = f"{protocol}://{user}:***@{host_db}"
            logger.info(f"🔗 Database URL: {safe_url}")
            
            # Ensure the database URL uses the correct async driver
            if not self.database_url.startswith("postgresql+psycopg://"):
                if self.database_url.startswith("postgresql://"):
                    # Replace postgresql:// with postgresql+psycopg:// for async driver
                    self.database_url = self.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
                elif self.database_url.startswith("postgresql+asyncpg://"):
                    # Replace asyncpg with psycopg for better compatibility
                    self.database_url = self.database_url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
            
            # Create async engine with connection pooling optimized for Render free plan
            self.engine = create_async_engine(
                self.database_url,
                echo=settings.DATABASE_ECHO,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes (Render free plan timeout)
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
            logger.error(f"🔍 Connection details: URL={safe_url}, Environment={settings.ENVIRONMENT}")
            self._initialization_failed = True
            # Don't raise - allow fallback to in-memory storage
            logger.warning("⚠️ Database initialization failed, will use fallback storage")
    
    async def _test_connection(self):
        """Test database connection with retry logic for Render free plan"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with self.engine.begin() as conn:
                    from sqlalchemy import text
                    await conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection test passed")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Database connection test failed (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"❌ Database connection test failed after {max_retries} attempts: {str(e)}")
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
