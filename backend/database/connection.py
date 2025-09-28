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

# Optional Supabase client import
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Async database connection manager with connection pooling"""
    
    def __init__(self):
        # Prioritize Supabase database URL for production, with better environment detection
        if settings.ENVIRONMENT == "production" and settings.SUPABASE_DATABASE_URL:
            self.database_url = settings.SUPABASE_DATABASE_URL
            logger.info("🔗 Using Supabase database for production")
        elif settings.SUPABASE_DATABASE_URL:
            self.database_url = settings.SUPABASE_DATABASE_URL
            logger.info("🔗 Using Supabase database (configured)")
        else:
            self.database_url = settings.DATABASE_URL
            logger.warning("⚠️ Using local database - SUPABASE_DATABASE_URL not configured")
        
        self.engine = None
        self.session_factory = None
        self._connection_pool = None
        self._is_initialized = False
        self._initialization_failed = False
        self._supabase_client = None
    
    async def initialize(self):
        """Initialize database connection and session factory"""
        try:
            logger.info("🔄 Initializing database connection...")
            
            # Log which database we're connecting to
            if "supabase.co" in self.database_url:
                logger.info("🔗 Connecting to Supabase database")
            else:
                logger.info("🔗 Connecting to local database")
            
            # Handle SSL configuration for production environments (Railway + Supabase)
            connect_args = {}
            if settings.ENVIRONMENT == "production":
                # For Supabase PostgreSQL, SSL is required
                # Ensure SSL is properly configured for Supabase
                if "sslmode" not in self.database_url and "ssl" not in self.database_url:
                    if "?" in self.database_url:
                        self.database_url += "&sslmode=require"
                    else:
                        self.database_url += "?sslmode=require"
                
                # Configure SSL settings for psycopg - Supabase requires SSL
                connect_args = {
                    "sslmode": "require"  # Force SSL connection for Supabase
                }
                
                # Supabase-specific SSL configuration
                if "supabase.co" in self.database_url:
                    logger.info("🔒 Configuring SSL for Supabase PostgreSQL connection")
                    # Use SSL configuration optimized for Supabase
                    connect_args = {
                        "sslmode": "require"  # Supabase PostgreSQL requires SSL
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
            
            # Create async engine with connection pooling optimized for Supabase
            self.engine = create_async_engine(
                self.database_url,
                echo=settings.DATABASE_ECHO,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=300,    # Recycle connections every 5 minutes
                pool_size=5,        # Supabase allows more connections
                max_overflow=10,     # Overflow connections for better performance
                pool_timeout=30,    # Connection timeout
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
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                async with self.engine.begin() as conn:
                    from sqlalchemy import text
                    await conn.execute(text("SELECT 1"))
                logger.info("✅ Database connection test passed")
                return
            except Exception as e:
                error_msg = str(e).lower()
                if "connection was closed" in error_msg or "connection refused" in error_msg:
                    logger.warning(f"⚠️ Database connection closed/refused (attempt {attempt + 1}/{max_retries}): {str(e)}")
                elif "timeout" in error_msg:
                    logger.warning(f"⚠️ Database connection timeout (attempt {attempt + 1}/{max_retries}): {str(e)}")
                else:
                    logger.warning(f"⚠️ Database connection test failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 1.5, 10)  # Gradual backoff, max 10s
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
    
    def get_supabase_client(self) -> Optional[Client]:
        """Get Supabase client for additional features"""
        if not SUPABASE_AVAILABLE:
            logger.warning("⚠️ Supabase client not available - install with: pip install supabase")
            return None
        
        if not self._supabase_client and settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
            try:
                self._supabase_client = create_client(
                    settings.SUPABASE_URL,
                    settings.SUPABASE_ANON_KEY
                )
                logger.info("✅ Supabase client initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Supabase client: {str(e)}")
                return None
        
        return self._supabase_client
    
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
