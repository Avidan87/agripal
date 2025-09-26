#!/usr/bin/env python3
"""
🚀 AgriPal Database Setup Script
Complete database setup and initialization for AgriPal.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from scripts.init_db import DatabaseInitializer
from scripts.test_db import test_database
from config import settings
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def setup_database():
    """Complete database setup process"""
    try:
        logger.info("🚀 Starting AgriPal database setup...")
        
        # Check if .env file exists
        env_file = Path(__file__).parent.parent / ".env"
        if not env_file.exists():
            logger.warning("⚠️ .env file not found. Please copy env.example to .env and configure your settings.")
            logger.info("📝 You can run: cp env.example .env")
            return False
        
        # Check database URL
        if not settings.DATABASE_URL or "localhost" in settings.DATABASE_URL:
            logger.info("📋 Database URL: " + settings.DATABASE_URL)
            logger.info("💡 Make sure PostgreSQL is running on your system")
        
        # Step 1: Initialize database schema
        logger.info("📋 Step 1: Initializing database schema...")
        initializer = DatabaseInitializer()
        await initializer.initialize()
        
        # Step 2: Test database operations
        logger.info("📋 Step 2: Testing database operations...")
        await test_database()
        
        # Step 3: Display setup summary
        logger.info("📋 Step 3: Setup summary...")
        logger.info("✅ Database schema created successfully")
        logger.info("✅ All tables and indexes created")
        logger.info("✅ Sample data inserted")
        logger.info("✅ Database operations tested")
        
        logger.info("🎉 AgriPal database setup completed successfully!")
        logger.info("🚀 You can now start the AgriPal backend server")
        logger.info("💡 Run: cd backend && python main.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {str(e)}")
        logger.error("💡 Please check your PostgreSQL connection and try again")
        return False

async def main():
    """Main setup function"""
    success = await setup_database()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
