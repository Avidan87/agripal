"""
🌾 AgriPal Backend API - Main FastAPI Application
A comprehensive agricultural assistant system with AI-powered crop analysis.
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
import uvicorn
import asyncio
from datetime import datetime
from typing import Optional
from uuid import UUID
import logging

from .config import settings
from .models import (
    AnalysisRequest, 
    AnalysisResponse, 
    HealthCheckResponse,
    UserSession
)
from .coordinator.agent_coordinator import AgentCoordinator, WorkflowType
from .middleware.auth_middleware import AgentAuthMiddleware
from .api.agent_routes import router as agent_router
from .utils import get_user_session, setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🌾 AgriPal AI API",
    description="Advanced AI-powered agricultural assistant with intelligent agent orchestration for crop analysis, knowledge retrieval, and farming insights",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.ALLOWED_ORIGINS == "*" else settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"] if settings.TRUSTED_HOSTS == "*" else settings.TRUSTED_HOSTS.split(",")
    )

# Add AI agent authentication and rate limiting middleware
app.add_middleware(
    AgentAuthMiddleware,
    enable_auth=getattr(settings, 'ENABLE_AUTH', False),  # Disable auth by default for development
    enable_rate_limiting=getattr(settings, 'ENABLE_RATE_LIMITING', True)
)

# Include new AI agent routes
app.include_router(agent_router)

# Initialize AI-powered coordinator (replaces old orchestrator)
coordinator = AgentCoordinator()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting AgriPal backend services with AI coordinator...")
    
    # Validate Supabase configuration for production
    if settings.ENVIRONMENT == "production":
        from .config import validate_supabase_config
        if not validate_supabase_config():
            logger.error("❌ Supabase configuration missing for production deployment!")
            logger.error("🔧 Please set SUPABASE_DATABASE_URL and SUPABASE_URL environment variables")
        else:
            logger.info("✅ Supabase configuration validated for production")
    
    # Test database connection on startup
    try:
        from .database.connection import get_database_manager
        db_manager = await get_database_manager()
        if db_manager.is_available():
            logger.info("✅ Database connection established successfully")
        else:
            logger.warning("⚠️ Database connection failed, running with fallback storage")
    except Exception as e:
        logger.warning(f"⚠️ Database startup check failed: {str(e)}")
    
    # AI coordinator initializes automatically
    await asyncio.sleep(2)  # Give time for agents to initialize
    logger.info("✅ AgriPal backend services with AI agents ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down AgriPal backend services...")
    
    # Cleanup database connections
    try:
        from .database.connection import close_database
        await close_database()
        logger.info("🔌 Database connections closed")
    except Exception as e:
        logger.warning(f"⚠️ Database cleanup warning: {str(e)}")
    
    # Cleanup AI coordinator
    await coordinator.cleanup()
    logger.info("✅ AgriPal backend services shut down gracefully!")

@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "🌾 Welcome to AgriPal - Advanced AI-Powered Agricultural Assistant",
        "version": "2.0.0",
        "status": "active",
        "features": ["Intelligent Agent Orchestration", "Image Analysis", "Knowledge Retrieval", "Email Reports"],
        "endpoints": {
            "api_docs": "/docs",
            "health_check": "/health",
            "agents": "/api/v1/agents/",
            "analysis": "/analyze"
        },
        "note": "Use http://localhost:8000 to access this API (not 0.0.0.0:8000)"
    }

@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_crop_issue(
    message: str,
    image: Optional[UploadFile] = File(None),
    user_session: UserSession = Depends(get_user_session)
):
    """
    🔍 Main endpoint for crop analysis requests
    
    Uses AI-powered coordinator to orchestrate multi-agent workflow:
    1. Perception Agent: Advanced image analysis using GPT-4o Vision
    2. Knowledge Agent: RAG-powered knowledge retrieval with Hugging Face + Weaviate
    3. Email Agent: Intelligent report generation and SendGrid delivery
    
    Args:
        message: User's agricultural query or concern
        image: Optional field image for visual analysis
        user_session: Current user session context
        
    Returns:
        AnalysisResponse with actionable agricultural advice
    """
    try:
        logger.info(f"🌱 Processing analysis request for session {user_session.id}")
        
        # Process image if provided
        image_data = None
        if image:
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
            
            image_data = await image.read()
            logger.info(f"📸 Image uploaded: {image.filename} ({len(image_data)} bytes)")
        
        # Prepare data for AI coordinator
        user_input = {
            "query": message,
            "images": [{"content": image_data, "filename": image.filename}] if image_data else [],
            "crop_type": getattr(user_session, 'crop_type', None)
        }
        
        user_context = {
            "user_name": getattr(user_session, 'user_name', 'Farmer'),
            "location": getattr(user_session, 'location', None),
            "user_id": user_session.id
        }
        
        # Execute AI-orchestrated workflow
        workflow_result = await coordinator.execute_workflow(
            workflow_type=WorkflowType.FULL_CONSULTATION,
            session_id=user_session.id,
            user_input=user_input,
            user_context=user_context
        )
        
        # 🔥 CRITICAL FIX: Persist conversation messages to maintain conversation history
        try:
            from .database.connection import get_database_manager
            from .database.services import SessionService
            from uuid import UUID
            
            db_manager = await get_database_manager()
            session_service = SessionService(db_manager)
            
            # Save user message
            await session_service.add_message(
                session_id=UUID(user_session.id),
                message_type="user",
                content=message,
                metadata={
                    "has_image": bool(image_data), 
                    "crop_type": getattr(user_session, 'crop_type', None),
                    "location": getattr(user_session, 'location', None)
                }
            )
            
            # Save assistant response (extract best available display text)
            results = workflow_result.get("results", {}) if isinstance(workflow_result, dict) else {}
            display_text = (
                results.get("display_text") or
                (results.get("knowledge") or {}).get("display_text") or
                (results.get("knowledge") or {}).get("contextual_advice") or
                (results.get("perception") or {}).get("analysis_text") or
                workflow_result.get("results", {}).get("knowledge", {}).get("contextual_advice", "Analysis completed")
            )
            
            if display_text:
                await session_service.add_message(
                    session_id=UUID(user_session.id),
                    message_type="assistant", 
                    content=display_text,
                    metadata={
                        "agents_executed": workflow_result.get("agents_executed", []),
                        "workflow_type": workflow_result.get("workflow_type", "FULL_CONSULTATION")
                    }
                )
            logger.info(f"💾 Conversation messages saved for session {user_session.id}")
            
        except Exception as db_error:
            logger.warning(f"⚠️ Database persistence failed, using fallback: {str(db_error)}")
            # Fallback to in-memory storage
            try:
                from .memory_storage import conversation_storage
                
                # Save user message
                conversation_storage.add_message(
                    session_id=user_session.id,
                    message_type="user",
                    content=message,
                    metadata={
                        "has_image": bool(image_data),
                        "crop_type": getattr(user_session, 'crop_type', None)
                    }
                )
                
                # Save assistant response
                results = workflow_result.get("results", {}) if isinstance(workflow_result, dict) else {}
                display_text = (
                    results.get("display_text") or
                    (results.get("knowledge") or {}).get("contextual_advice") or
                    workflow_result.get("results", {}).get("knowledge", {}).get("contextual_advice", "Analysis completed")
                )
                
                if display_text:
                    conversation_storage.add_message(
                        session_id=user_session.id,
                        message_type="assistant",
                        content=display_text,
                        metadata={"agents_executed": workflow_result.get("agents_executed", [])}
                    )
                logger.info(f"💾 Conversation saved to fallback storage for session {user_session.id}")
                
            except Exception as fallback_error:
                logger.error(f"❌ Both database and fallback storage failed: {str(fallback_error)}")
        
        # Convert to original response format for backward compatibility
        response = AnalysisResponse(
            response_text=workflow_result.get("results", {}).get("knowledge", {}).get("contextual_advice", "Analysis completed"),
            confidence_score=0.8,
            recommendations=workflow_result.get("results", {}).get("knowledge", {}).get("relevant_documents", []),
            session_id=user_session.id,
            processing_time_ms=int(workflow_result.get("execution_time_seconds", 0) * 1000)
        )
        
        logger.info(f"✅ Analysis completed for session {user_session.id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    🏥 Comprehensive health check endpoint
    
    Verifies the status of all system components:
    - Database connectivity
    - External API availability
    - Agent system health
    - Storage services
    """
    try:
        # Use AI coordinator health check
        agent_health = await coordinator.health_check()
        
        # Database health check
        database_healthy = False
        try:
            from .database.connection import get_database_manager
            db_manager = await get_database_manager()
            database_healthy = await db_manager.health_check()
        except Exception as e:
            logger.warning(f"⚠️ Database health check failed: {str(e)}")
        
        # Combine with any additional system checks
        checks = {
            **agent_health,
            "api": True,  # API is responding
            "database": database_healthy
        }
        
        overall_status = "healthy" if all(checks.values()) else "unhealthy"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            checks=checks,
            version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            checks={"error": False},
            version="1.0.0",
            error=str(e)
        )

@app.get("/sessions/{session_id}/history", tags=["Sessions"])
async def get_session_history(
    session_id: str,
    user_session: UserSession = Depends(get_user_session)
):
    """
    📜 Retrieve chat history for a specific session
    
    Args:
        session_id: UUID of the chat session
        user_session: Current user session for authorization
        
    Returns:
        List of session messages with metadata
    """
    try:
        # Verify user has access to this session
        if session_id != user_session.id and not user_session.is_admin:
            raise HTTPException(status_code=403, detail="Access denied to this session")
        
        # Try to retrieve from database first
        try:
            from .database.connection import get_database_manager
            from .database.services import SessionService
            db_manager = await get_database_manager()
            
            # Check if database is available
            logger.info(f"🔍 Database available: {db_manager.is_available()}")
            if db_manager.is_available():
                logger.info("📊 Using database for session history")
                session_service = SessionService(db_manager)
                messages = await session_service.get_recent_messages(UUID(session_id), limit=100)
                history = [
                    {
                        "id": str(m.id),
                        "type": m.message_type,
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat() if getattr(m, "timestamp", None) else None,
                        "metadata": m.message_metadata,
                    }
                    for m in reversed(messages)
                ]
                return {"session_id": session_id, "messages": history}
            else:
                logger.info("💾 Database not available, raising exception to trigger fallback")
                raise Exception("Database not available")
            
        except Exception as db_error:
            logger.warning(f"⚠️ Database unavailable, using fallback storage: {str(db_error)}")
            # Fallback to in-memory storage
            from .memory_storage import conversation_storage
            messages = conversation_storage.get_messages(session_id, limit=100)
            logger.info(f"📚 Retrieved {len(messages)} messages from fallback storage")
            return {"session_id": session_id, "messages": messages}
        
    except Exception as e:
        logger.error(f"❌ Failed to retrieve session history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")

@app.post("/sessions/{session_id}/report", tags=["Reports"])
async def generate_session_report(
    session_id: str,
    user_session: UserSession = Depends(get_user_session)
):
    """
    📧 Generate and send email report for completed session
    
    Args:
        session_id: UUID of the completed chat session
        user_session: Current user session for authorization
        
    Returns:
        Report generation status
    """
    try:
        # Verify user has access to this session
        if session_id != user_session.id and not user_session.is_admin:
            raise HTTPException(status_code=403, detail="Access denied to this session")
        
        # Generate report using AI coordinator
        user_input = {
            "recipient_email": getattr(user_session, 'email', None),
            "session_data": {"session_id": session_id}
        }
        
        if not user_input["recipient_email"]:
            raise HTTPException(status_code=400, detail="User email not available for report generation")
        
        workflow_result = await coordinator.execute_workflow(
            workflow_type=WorkflowType.EMAIL_REPORT_ONLY,
            session_id=session_id,
            user_input=user_input,
            user_context={"user_id": user_session.id}
        )
        
        report_result = {
            "report_id": workflow_result.get("workflow_id"),
            "email_sent": workflow_result.get("status") == "completed"
        }
        return {
            "message": "📧 Session report generated and sent successfully!",
            "report_id": report_result["report_id"],
            "email_sent": report_result["email_sent"]
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to generate session report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.middleware("http")
async def add_request_logging(request: Request, call_next):
    """Log all requests for monitoring and debugging"""
    start_time = datetime.utcnow()
    
    # Log request
    logger.info(f"🌐 {request.method} {request.url.path} - Started")
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Log response
    logger.info(f"✅ {request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
