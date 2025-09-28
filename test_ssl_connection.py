#!/usr/bin/env python3
"""
🔧 SSL Connection Test for Render PostgreSQL
Test script to verify SSL/TLS connection to Render PostgreSQL database.
"""
import asyncio
import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_ssl_connection():
    """Test SSL connection to Render PostgreSQL"""
    try:
        import asyncpg
        
        # Get database URL
        database_url = settings.DATABASE_URL
        logger.info(f"🔗 Testing connection to: {database_url.split('@')[1] if '@' in database_url else 'hidden'}")
        
        # Convert to asyncpg format
        asyncpg_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
        
        # Add SSL mode for Render
        if settings.ENVIRONMENT == "production" and ("render.com" in asyncpg_url or "onrender.com" in asyncpg_url):
            if "?" in asyncpg_url:
                asyncpg_url += "&sslmode=require"
            else:
                asyncpg_url += "?sslmode=require"
            logger.info("🔒 Added sslmode=require for Render PostgreSQL")
        
        # Test connection with SSL
        logger.info("🔄 Attempting SSL connection...")
        
        # Simple SSL configuration
        ssl_config = "require" if settings.ENVIRONMENT == "production" else None
        
        conn = await asyncpg.connect(
            asyncpg_url,
            ssl=ssl_config
        )
        
        # Test query
        result = await conn.fetchval("SELECT 1")
        logger.info(f"✅ SSL connection successful! Test query result: {result}")
        
        await conn.close()
        logger.info("🔌 Connection closed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SSL connection failed: {str(e)}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting SSL connection test...")
    logger.info(f"🌍 Environment: {settings.ENVIRONMENT}")
    logger.info(f"🔗 Database URL: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'hidden'}")
    
    success = await test_ssl_connection()
    
    if success:
        logger.info("🎉 SSL connection test PASSED!")
        sys.exit(0)
    else:
        logger.error("💥 SSL connection test FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
