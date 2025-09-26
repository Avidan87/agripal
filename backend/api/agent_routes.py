"""
🌐 Agent API Routes - FastAPI endpoints for AgriPal AI Agents
RESTful interfaces for the AI-powered agent coordination system.
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime
import logging

from ..coordinator.agent_coordinator import AgentCoordinator, WorkflowType
from ..coordinator.workflow_models import WorkflowRequest, WorkflowResponse, CoordinatorMetrics
from ..models import AgentType
from ..middleware.auth_middleware import get_current_user, get_current_user_optional, JWTAuthenticator
from ..config import settings
from ..database.connection import get_database_manager
from ..database.services import SessionService

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Global coordinator instance (will be initialized)
coordinator: Optional[AgentCoordinator] = None

# API Router
router = APIRouter(prefix="/api/v1/agents", tags=["AI Agents"])

# Request/Response Models
class AgentHealthResponse(BaseModel):
    """Health status response"""
    healthy: bool = Field(..., description="Overall health status")
    agents: Dict[str, bool] = Field(..., description="Individual agent health")
    timestamp: datetime = Field(..., description="Health check timestamp")

class WorkflowStatusResponse(BaseModel):
    """Workflow status response"""
    workflow_id: str = Field(..., description="Workflow identifier")
    status: str = Field(..., description="Current status")
    progress: Dict[str, Any] = Field(..., description="Progress information")
    results: Optional[Dict[str, Any]] = Field(None, description="Results if completed")

class ImageAnalysisRequest(BaseModel):
    """Image analysis request"""
    crop_type: Optional[str] = Field(None, description="Type of crop")
    analysis_type: str = Field(default="comprehensive", description="Type of analysis")
    session_id: Optional[str] = Field(None, description="Session identifier")

class KnowledgeSearchRequest(BaseModel):
    """Knowledge search request"""
    query: str = Field(..., description="Search query")
    crop_type: Optional[str] = Field(None, description="Crop type for context")
    location: Optional[str] = Field(None, description="User location")
    session_id: Optional[str] = Field(None, description="Session identifier")

class EmailReportRequest(BaseModel):
    """Email report request"""
    recipient_email: str = Field(..., description="Email recipient")
    session_data: Dict[str, Any] = Field(..., description="Session data for report")
    user_context: Optional[Dict[str, Any]] = Field(None, description="User context")
    session_id: Optional[str] = Field(None, description="Session identifier")

class FullConsultationRequest(BaseModel):
    """Full consultation workflow request"""
    query: str = Field(..., description="User's agricultural question")
    crop_type: Optional[str] = Field(None, description="Type of crop")
    location: Optional[str] = Field(None, description="User location")
    recipient_email: Optional[str] = Field(None, description="Email for report")
    user_name: Optional[str] = Field(None, description="User's name")
    session_id: Optional[str] = Field(None, description="Session identifier")

class LoginRequest(BaseModel):
    """Login request for testing authentication"""
    email: str = Field(..., description="User email")
    user_id: str = Field(..., description="User ID")

class LoginResponse(BaseModel):
    """Login response with JWT token"""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")

# Dependency for coordinator
async def get_coordinator() -> AgentCoordinator:
    """Get coordinator instance"""
    global coordinator
    if coordinator is None:
        coordinator = AgentCoordinator()
        # Wait a bit for initialization
        await asyncio.sleep(2)
    return coordinator

# Authentication dependency helper
async def get_authenticated_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get authenticated user if auth is enabled, otherwise return None"""
    if not settings.ENABLE_AUTH:
        return None
    
    try:
        return await get_current_user_optional(request)
    except Exception as e:
        logger.warning(f"Authentication check failed: {str(e)}")
        return None

# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/api/v1/agents/health",
    "/api/v1/agents/metrics",
    "/api/v1/agents/auth/login"
}

# Health and status endpoints
@router.get("/health", response_model=AgentHealthResponse)
async def get_agent_health(
    request: Request,
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Check health status of all AI agents
    Public endpoint - no authentication required
    """
    try:
        health_status = await coord.health_check()
        overall_healthy = any(health_status.values())
        
        return AgentHealthResponse(
            healthy=overall_healthy,
            agents=health_status,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/metrics", response_model=CoordinatorMetrics)
async def get_coordinator_metrics(
    request: Request,
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Get coordinator performance metrics
    Public endpoint - no authentication required
    """
    try:
        metrics = await coord.get_metrics()
        return CoordinatorMetrics(**metrics)
    except Exception as e:
        logger.error(f"❌ Metrics retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

# Authentication endpoints
@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login endpoint for testing authentication
    Returns a JWT token for authenticated requests
    """
    try:
        authenticator = JWTAuthenticator(settings.JWT_SECRET_KEY)
        
        # Create JWT token
        access_token = authenticator.create_token(
            user_id=request.user_id,
            email=request.email,
            expires_delta=None  # Use default expiration
        )
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except Exception as e:
        logger.error(f"❌ Login failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.get("/workflows/active")
async def get_active_workflows(
    request: Request,
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Get list of currently active workflows
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        active_workflows = await coord.get_active_workflows()
        return {"active_workflows": active_workflows, "count": len(active_workflows)}
    except Exception as e:
        logger.error(f"❌ Failed to get active workflows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active workflows: {str(e)}")

@router.get("/workflows/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(
    workflow_id: str,
    request: Request,
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Get status of specific workflow
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        workflow = await coord.get_workflow_status(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=workflow["status"],
            progress={
                "agents_executed": workflow.get("agents_executed", []),
                "execution_time_seconds": workflow.get("execution_time_seconds"),
                "start_time": workflow.get("start_time")
            },
            results=workflow.get("results")
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get workflow status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")

# Individual agent endpoints
@router.post("/perception/analyze")
async def analyze_images(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request_data: str = Form(...),
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Analyze crop images using the perception agent
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        # Parse request data
        request_params = json.loads(request_data)
        analysis_request = ImageAnalysisRequest(**request_params)
        
        # Process uploaded images
        images = []
        for file in files:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")
            
            content = await file.read()
            images.append({
                "filename": file.filename,
                "content": content,
                "content_type": file.content_type
            })
        
        # Prepare workflow input
        user_input = {
            "images": images,
            "crop_type": analysis_request.crop_type,
            "analysis_type": analysis_request.analysis_type
        }
        
        session_id = analysis_request.session_id or f"img_{datetime.utcnow().timestamp()}"
        
        # Execute workflow
        workflow_result = await coord.execute_workflow(
            workflow_type=WorkflowType.IMAGE_ANALYSIS_ONLY,
            session_id=session_id,
            user_input=user_input
        )
        
        return {
            "workflow_id": workflow_result["workflow_id"],
            "status": workflow_result["status"],
            "results": workflow_result.get("results", {}),
            "execution_time_seconds": workflow_result.get("execution_time_seconds"),
            "agents_executed": workflow_result.get("agents_executed", [])
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request_data")
    except Exception as e:
        logger.error(f"❌ Image analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@router.post("/knowledge/search")
async def search_knowledge(
    request: KnowledgeSearchRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Search agricultural knowledge base
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        # Prepare workflow input
        user_input = {
            "query": request.query,
            "crop_type": request.crop_type
        }
        
        user_context = {
            "location": request.location,
            "user_id": user.get("user_id") if user else None,
            "user_email": user.get("email") if user else None
        }
        
        session_id = request.session_id or f"search_{datetime.utcnow().timestamp()}"
        
        # Execute workflow
        workflow_result = await coord.execute_workflow(
            workflow_type=WorkflowType.KNOWLEDGE_SEARCH_ONLY,
            session_id=session_id,
            user_input=user_input,
            user_context=user_context
        )
        
        # Persist conversation messages
        try:
            db_manager = await get_database_manager()
            session_service = SessionService(db_manager)
            # Save user message
            await session_service.add_message(
                session_id=session_id,
                message_type="user",
                content=request.query,
                metadata={"crop_type": request.crop_type, "location": request.location}
            )
            # Save assistant response (best available display text)
            results = workflow_result.get("results", {}) if isinstance(workflow_result, dict) else {}
            display_text = (
                results.get("display_text")
                or (results.get("knowledge") or {}).get("display_text")
                or (results.get("knowledge") or {}).get("contextual_advice")
                or ""
            )
            if display_text:
                await session_service.add_message(
                    session_id=session_id,
                    message_type="assistant",
                    content=display_text,
                    metadata={"agents_executed": workflow_result.get("agents_executed", [])}
                )
        except Exception as db_error:
            logger.warning(f"Database unavailable, using fallback storage: {str(db_error)}")
            # Fallback to in-memory storage
            from ..memory_storage import conversation_storage
            # Save user message
            conversation_storage.add_message(
                session_id=session_id,
                message_type="user",
                content=request.query,
                metadata={"crop_type": request.crop_type, "location": request.location}
            )
            # Save assistant response
            results = workflow_result.get("results", {}) if isinstance(workflow_result, dict) else {}
            display_text = (
                results.get("display_text")
                or (results.get("knowledge") or {}).get("display_text")
                or (results.get("knowledge") or {}).get("contextual_advice")
                or ""
            )
            if display_text:
                conversation_storage.add_message(
                    session_id=session_id,
                    message_type="assistant",
                    content=display_text,
                    metadata={"agents_executed": workflow_result.get("agents_executed", [])}
                )

        # Extract display_text from results to include at top level for UI compatibility
        results = workflow_result.get("results", {})
        display_text = results.get("display_text")
        
        response = {
            "workflow_id": workflow_result["workflow_id"],
            "status": workflow_result["status"],
            "results": results,
            "execution_time_seconds": workflow_result.get("execution_time_seconds"),
            "agents_executed": workflow_result.get("agents_executed", []),
            "session_id": session_id
        }
        
        # Add display_text at top level if available for frontend compatibility
        if display_text:
            response["display_text"] = display_text
            
        return response
        
    except Exception as e:
        logger.error(f"❌ Knowledge search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Knowledge search failed: {str(e)}")

@router.post("/email/send-report")
async def send_email_report(
    request: EmailReportRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Generate and send email report
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        # Prepare workflow input
        user_input = {
            "recipient_email": request.recipient_email,
            "session_data": request.session_data
        }
        
        session_id = request.session_id or f"email_{datetime.utcnow().timestamp()}"
        
        # Execute workflow
        workflow_result = await coord.execute_workflow(
            workflow_type=WorkflowType.EMAIL_REPORT_ONLY,
            session_id=session_id,
            user_input=user_input,
            user_context=request.user_context or {}
        )
        
        return {
            "workflow_id": workflow_result["workflow_id"],
            "status": workflow_result["status"],
            "results": workflow_result.get("results", {}),
            "execution_time_seconds": workflow_result.get("execution_time_seconds"),
            "agents_executed": workflow_result.get("agents_executed", [])
        }
        
    except Exception as e:
        logger.error(f"❌ Email report failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Email report failed: {str(e)}")

# Comprehensive workflow endpoints
@router.post("/consultation/full")
async def full_consultation(
    http_request: Request,
    background_tasks: BackgroundTasks,
    request: FullConsultationRequest,
    files: Optional[List[UploadFile]] = File(None),
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Execute complete agricultural consultation workflow
    Includes image analysis (if images), knowledge search, and email report (if email provided)
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        # Process uploaded images if any
        images = []
        if files:
            for file in files:
                if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                    raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")
                
                content = await file.read()
                images.append({
                    "filename": file.filename,
                    "content": content,
                    "content_type": file.content_type
                })
        
        # Prepare workflow input
        user_input = {
            "query": request.query,
            "crop_type": request.crop_type,
            "images": images,
            "recipient_email": request.recipient_email
        }
        
        user_context = {
            "location": request.location,
            "user_name": request.user_name,
            "user_id": user.get("user_id") if user else None,
            "user_email": user.get("email") if user else None
        }
        
        session_id = request.session_id or f"consultation_{datetime.utcnow().timestamp()}"
        
        # Execute full consultation workflow
        workflow_result = await coord.execute_workflow(
            workflow_type=WorkflowType.FULL_CONSULTATION,
            session_id=session_id,
            user_input=user_input,
            user_context=user_context
        )
        
        return {
            "workflow_id": workflow_result["workflow_id"],
            "status": workflow_result["status"],
            "results": workflow_result.get("results", {}),
            "execution_time_seconds": workflow_result.get("execution_time_seconds"),
            "agents_executed": workflow_result.get("agents_executed", []),
            "session_id": session_id,
            "recommendation": "Full agricultural consultation completed. Check your email if provided."
        }
        
    except Exception as e:
        logger.error(f"❌ Full consultation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Full consultation failed: {str(e)}")

# Advanced workflow endpoints
@router.post("/workflows/perception-to-knowledge")
async def perception_to_knowledge_workflow(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    query: str = Form(...),
    crop_type: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Execute perception analysis followed by knowledge search with context passing
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        # Process images
        images = []
        for file in files:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(status_code=400, detail=f"Invalid file type: {file.content_type}")
            
            content = await file.read()
            images.append({
                "filename": file.filename,
                "content": content,
                "content_type": file.content_type
            })
        
        # Prepare workflow input
        user_input = {
            "images": images,
            "query": query,
            "crop_type": crop_type
        }
        
        user_context = {
            "location": location
        }
        
        session_id = f"perception_knowledge_{datetime.utcnow().timestamp()}"
        
        # Execute workflow
        workflow_result = await coord.execute_workflow(
            workflow_type=WorkflowType.PERCEPTION_TO_KNOWLEDGE,
            session_id=session_id,
            user_input=user_input,
            user_context=user_context
        )
        
        # Persist conversation (user query and assistant reply)
        try:
            db_manager = await get_database_manager()
            session_service = SessionService(db_manager)
            await session_service.add_message(
                session_id=session_id,
                message_type="user",
                content=query,
                metadata={"crop_type": crop_type, "location": location, "images_count": len(images)}
            )
            results = workflow_result.get("results", {}) if isinstance(workflow_result, dict) else {}
            display_text = (
                results.get("display_text")
                or (results.get("knowledge") or {}).get("display_text")
                or (results.get("knowledge") or {}).get("contextual_advice")
                or ""
            )
            if display_text:
                await session_service.add_message(
                    session_id=session_id,
                    message_type="assistant",
                    content=display_text,
                    metadata={"agents_executed": workflow_result.get("agents_executed", [])}
                )
        except Exception as e:
            logger.warning(f"Conversation persistence skipped: {str(e)}")

        # Extract display_text from results to include at top level for UI compatibility
        results = workflow_result.get("results", {})
        display_text = results.get("display_text")
        
        response = {
            "workflow_id": workflow_result["workflow_id"],
            "status": workflow_result["status"],
            "workflow_type": workflow_result.get("workflow_type", "perception_to_knowledge"),
            "results": results,
            "execution_time_seconds": workflow_result.get("execution_time_seconds"),
            "agents_executed": workflow_result.get("agents_executed", []),
            "session_id": session_id
        }
        
        # Add display_text at top level if available for frontend compatibility
        if display_text:
            response["display_text"] = display_text
            
        return response
        
    except Exception as e:
        logger.error(f"❌ Perception to knowledge workflow failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Perception to knowledge workflow failed: {str(e)}")

# Batch processing endpoint
@router.post("/batch/process")
async def batch_process_requests(
    requests: List[Dict[str, Any]],
    background_tasks: BackgroundTasks,
    request: Request,
    coord: AgentCoordinator = Depends(get_coordinator),
    user: Optional[Dict[str, Any]] = Depends(get_authenticated_user)
):
    """
    Process multiple agent requests in batch
    Requires authentication if ENABLE_AUTH is True
    """
    if settings.ENABLE_AUTH and not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        batch_results = []
        
        for i, req in enumerate(requests):
            try:
                workflow_type = WorkflowType(req.get("workflow_type", "full_consultation"))
                session_id = req.get("session_id", f"batch_{i}_{datetime.utcnow().timestamp()}")
                
                result = await coord.execute_workflow(
                    workflow_type=workflow_type,
                    session_id=session_id,
                    user_input=req.get("user_input", {}),
                    user_context=req.get("user_context", {})
                )
                
                batch_results.append({
                    "request_index": i,
                    "success": True,
                    "result": result
                })
                
            except Exception as e:
                batch_results.append({
                    "request_index": i,
                    "success": False,
                    "error": str(e)
                })
        
        successful_count = sum(1 for r in batch_results if r["success"])
        
        return {
            "batch_id": f"batch_{datetime.utcnow().timestamp()}",
            "total_requests": len(requests),
            "successful_requests": successful_count,
            "failed_requests": len(requests) - successful_count,
            "results": batch_results
        }
        
    except Exception as e:
        logger.error(f"❌ Batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Export router
__all__ = ["router"]

