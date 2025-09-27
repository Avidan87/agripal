#!/usr/bin/env python3
"""
🔍 AgriPal Database Status Checker
Simple script to check if the database is available and healthy.
"""
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.database.connection import get_database_manager
from backend.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def check_database_status():
    """Check database status and provide detailed information"""
    print("🔍 Checking AgriPal Database Status...")
    print(f"📍 Database URL: {settings.DATABASE_URL}")
    print(f"🌍 Environment: {settings.ENVIRONMENT}")
    print("-" * 50)
    
    try:
        # Get database manager
        db_manager = await get_database_manager()
        
        # Check basic availability
        is_available = db_manager.is_available()
        is_initialized = db_manager._is_initialized
        initialization_failed = db_manager._initialization_failed
        
        print(f"✅ Database Manager Created: Yes")
        print(f"🔧 Initialized: {is_initialized}")
        print(f"❌ Initialization Failed: {initialization_failed}")
        print(f"📊 Available: {is_available}")
        
        if is_available:
            # Test connection health
            print("\n🏥 Testing Database Connection Health...")
            try:
                health_status = await db_manager.health_check()
                if health_status:
                    print("✅ Database Connection: Healthy")
                    print("🎉 Database is fully operational!")
                else:
                    print("❌ Database Connection: Unhealthy")
            except Exception as health_error:
                print(f"❌ Database Health Check Failed: {str(health_error)}")
        else:
            print("\n❌ Database is not available")
            if initialization_failed:
                print("💡 Database initialization failed during startup")
            if not is_initialized:
                print("💡 Database was never initialized")
        
        # Test a simple query if available
        if is_available:
            print("\n🧪 Testing Database Query...")
            try:
                result = await db_manager.fetch_one("SELECT 1 as test")
                if result:
                    print("✅ Database Query Test: Success")
                    print(f"📊 Query Result: {result}")
                else:
                    print("❌ Database Query Test: Failed (no result)")
            except Exception as query_error:
                print(f"❌ Database Query Test Failed: {str(query_error)}")
        
    except Exception as e:
        print(f"❌ Database Status Check Failed: {str(e)}")
        return False
    
    return is_available

async def main():
    """Main function"""
    print("🌾 AgriPal Database Status Checker")
    print("=" * 50)
    
    try:
        is_healthy = await check_database_status()
        
        print("\n" + "=" * 50)
        if is_healthy:
            print("🎉 RESULT: Database is available and healthy!")
            print("💡 You can now use database-dependent features.")
        else:
            print("⚠️  RESULT: Database is not available")
            print("💡 The application will use fallback storage (in-memory).")
            print("💡 To fix this:")
            print("   1. Start PostgreSQL: docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres")
            print("   2. Create database: createdb -h localhost -U postgres agripal")
            print("   3. Or update DATABASE_URL in your .env file")
        
    except KeyboardInterrupt:
        print("\n👋 Database check interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
