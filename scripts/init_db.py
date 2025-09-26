#!/usr/bin/env python3
"""
🗄️ AgriPal Database Initialization Script
Creates PostgreSQL database schema and initializes the AgriPal system.
"""
import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from typing import Optional
import logging

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent / "backend"))

# Import after path modification
from config import settings

logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Database initialization and setup manager"""
    
    def __init__(self):
        self.database_url = settings.DATABASE_URL
        self.connection: Optional[asyncpg.Connection] = None
    
    async def initialize(self):
        """Initialize the complete database setup"""
        try:
            logger.info("🚀 Starting AgriPal database initialization...")
            
            # Connect to database
            await self._connect()
            
            # Create schema
            await self._create_schema()
            
            # Create tables
            await self._create_tables()
            
            # Create indexes
            await self._create_indexes()
            
            # Insert initial data
            await self._insert_initial_data()
            
            # Verify setup
            await self._verify_setup()
            
            logger.info("✅ Database initialization completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {str(e)}")
            raise
        finally:
            await self._disconnect()
    
    async def _connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = await asyncpg.connect(self.database_url)
            logger.info("✅ Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {str(e)}")
            raise
    
    async def _disconnect(self):
        """Disconnect from database"""
        if self.connection:
            await self.connection.close()
            logger.info("🔌 Disconnected from database")
    
    async def _create_schema(self):
        """Create database schema and extensions"""
        try:
            # Enable required extensions
            extensions = [
                "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";",
                "CREATE EXTENSION IF NOT EXISTS \"pgcrypto\";",
                "CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";",  # For text search
            ]
            
            for extension in extensions:
                await self.connection.execute(extension)
            
            logger.info("✅ Database extensions enabled")
            
        except Exception as e:
            logger.error(f"❌ Failed to create schema: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create all database tables"""
        try:
            # Read and execute schema file
            schema_file = Path(__file__).parent / "schema.sql"
            if schema_file.exists():
                schema_sql = schema_file.read_text()
                await self.connection.execute(schema_sql)
                logger.info("✅ Database tables created")
            else:
                logger.warning("⚠️ Schema file not found, creating tables manually")
                await self._create_tables_manual()
                
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {str(e)}")
            raise
    
    async def _create_tables_manual(self):
        """Create tables manually if schema file is not available"""
        tables = [
            # Users table
            """
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255),
                location JSONB,
                farm_details JSONB,
                preferences JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """,
            
            # Chat sessions table
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                started_at TIMESTAMP DEFAULT NOW(),
                ended_at TIMESTAMP,
                session_data JSONB,
                summary TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """,
            
            # Session messages table
            """
            CREATE TABLE IF NOT EXISTS session_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
                message_type VARCHAR(50) NOT NULL, -- 'user', 'assistant', 'system'
                content TEXT NOT NULL,
                metadata JSONB,
                timestamp TIMESTAMP DEFAULT NOW()
            );
            """,
            
            # Analysis reports table
            """
            CREATE TABLE IF NOT EXISTS analysis_reports (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
                image_url VARCHAR(500),
                analysis_results JSONB,
                recommendations JSONB,
                confidence_score FLOAT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """,
            
            # Knowledge documents table
            """
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source VARCHAR(255) NOT NULL,
                title VARCHAR(500),
                document_type VARCHAR(100), -- 'fao_manual', 'research_paper', 'extension_guide'
                crop_type VARCHAR(100),
                region VARCHAR(100),
                language VARCHAR(10) DEFAULT 'en',
                ingestion_date TIMESTAMP DEFAULT NOW(),
                chunks_count INTEGER,
                file_size BIGINT,
                file_hash VARCHAR(64), -- For duplicate detection
                metadata JSONB
            );
            """,
            
            # Ingestion logs table
            """
            CREATE TABLE IF NOT EXISTS ingestion_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                source VARCHAR(255),
                status VARCHAR(50), -- 'success', 'failed', 'partial'
                ingested_at TIMESTAMP DEFAULT NOW(),
                chunks_created INTEGER,
                error_message TEXT,
                processing_time_ms INTEGER
            );
            """,
            
            # RAG query analytics table
            """
            CREATE TABLE IF NOT EXISTS rag_query_analytics (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
                query_text TEXT,
                query_embedding_model VARCHAR(100),
                retrieved_chunks JSONB, -- Store retrieved document chunks
                response_quality_score FLOAT, -- User feedback or auto-eval
                processing_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """,
            
            # Vector sync status table (optional for future ChromaDB integration)
            """
            CREATE TABLE IF NOT EXISTS vector_sync_status (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                collection_name VARCHAR(100),
                last_sync_at TIMESTAMP,
                total_objects INTEGER,
                sync_status VARCHAR(50),
                error_details TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """
        ]
        
        for table_sql in tables:
            await self.connection.execute(table_sql)
        
        logger.info("✅ All database tables created manually")
    
    async def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            indexes = [
                # Users indexes
                "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
                "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);",
                
                # Chat sessions indexes
                "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);",
                "CREATE INDEX IF NOT EXISTS idx_chat_sessions_started_at ON chat_sessions(started_at);",
                "CREATE INDEX IF NOT EXISTS idx_chat_sessions_ended_at ON chat_sessions(ended_at);",
                
                # Session messages indexes
                "CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_session_messages_timestamp ON session_messages(timestamp);",
                "CREATE INDEX IF NOT EXISTS idx_session_messages_type ON session_messages(message_type);",
                
                # Analysis reports indexes
                "CREATE INDEX IF NOT EXISTS idx_analysis_reports_session_id ON analysis_reports(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_analysis_reports_created_at ON analysis_reports(created_at);",
                
                # Knowledge documents indexes
                "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_source ON knowledge_documents(source);",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_crop_type ON knowledge_documents(crop_type);",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_region ON knowledge_documents(region);",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_document_type ON knowledge_documents(document_type);",
                "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_ingestion_date ON knowledge_documents(ingestion_date);",
                
                # Ingestion logs indexes
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_document_id ON ingestion_logs(document_id);",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_status ON ingestion_logs(status);",
                "CREATE INDEX IF NOT EXISTS idx_ingestion_logs_ingested_at ON ingestion_logs(ingested_at);",
                
                # RAG query analytics indexes
                "CREATE INDEX IF NOT EXISTS idx_rag_query_analytics_session_id ON rag_query_analytics(session_id);",
                "CREATE INDEX IF NOT EXISTS idx_rag_query_analytics_created_at ON rag_query_analytics(created_at);",
                
            # Vector sync status indexes
            "CREATE INDEX IF NOT EXISTS idx_vector_sync_status_collection_name ON vector_sync_status(collection_name);",
            "CREATE INDEX IF NOT EXISTS idx_vector_sync_status_sync_status ON vector_sync_status(sync_status);",
            ]
            
            for index_sql in indexes:
                await self.connection.execute(index_sql)
            
            logger.info("✅ Database indexes created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indexes: {str(e)}")
            raise
    
    async def _insert_initial_data(self):
        """Insert initial data and sample records"""
        try:
            # Insert sample user
            await self.connection.execute("""
                INSERT INTO users (email, name, location, farm_details, preferences)
                VALUES (
                    'demo@agripal.com',
                    'Demo Farmer',
                    '{"country": "USA", "state": "California", "city": "Fresno"}',
                    '{"farm_size": "100 acres", "crop_types": ["wheat", "corn"], "farming_method": "conventional"}',
                    '{"language": "en", "notifications": true, "units": "metric"}'
                ) ON CONFLICT (email) DO NOTHING;
            """)
            
            # Insert sample knowledge document
            await self.connection.execute("""
                INSERT INTO knowledge_documents (source, title, document_type, crop_type, region, language, chunks_count, file_size, metadata)
                VALUES (
                    'FAO',
                    'Wheat Production Guide',
                    'fao_manual',
                    'wheat',
                    'global',
                    'en',
                    50,
                    1024000,
                    '{"author": "FAO", "year": 2023, "topics": ["planting", "harvesting", "disease_management"]}'
                );
            """)
            
            # Insert initial vector sync status (optional for future ChromaDB integration)
            await self.connection.execute("""
                INSERT INTO vector_sync_status (collection_name, last_sync_at, total_objects, sync_status)
                VALUES (
                    'agricultural_knowledge',
                    NOW(),
                    0,
                    'pending'
                );
            """)
            
            logger.info("✅ Initial data inserted")
            
        except Exception as e:
            logger.error(f"❌ Failed to insert initial data: {str(e)}")
            raise
    
    async def _verify_setup(self):
        """Verify database setup is correct"""
        try:
            # Check tables exist
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """
            tables = await self.connection.fetch(tables_query)
            
            expected_tables = [
                'users', 'chat_sessions', 'session_messages', 'analysis_reports',
                'knowledge_documents', 'ingestion_logs', 'rag_query_analytics', 'vector_sync_status'
            ]
            
            actual_tables = [row['table_name'] for row in tables]
            
            for table in expected_tables:
                if table not in actual_tables:
                    raise Exception(f"Table {table} was not created")
            
            # Check extensions
            extensions_query = """
                SELECT extname 
                FROM pg_extension 
                WHERE extname IN ('uuid-ossp', 'pgcrypto', 'pg_trgm');
            """
            extensions = await self.connection.fetch(extensions_query)
            
            if len(extensions) < 3:
                logger.warning("⚠️ Some extensions may not be installed")
            
            logger.info(f"✅ Database verification passed. Tables: {', '.join(actual_tables)}")
            
        except Exception as e:
            logger.error(f"❌ Database verification failed: {str(e)}")
            raise

async def main():
    """Main initialization function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    initializer = DatabaseInitializer()
    await initializer.initialize()

if __name__ == "__main__":
    asyncio.run(main())
