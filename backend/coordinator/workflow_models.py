"""
📊 Workflow Models and Utilities
Data models and utilities for workflow orchestration and error handling.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum

class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    workflow_type: str = Field(..., description="Type of workflow to execute")
    session_id: str = Field(..., description="Unique session identifier")
    user_input: Dict[str, Any] = Field(..., description="User input data")
    user_context: Optional[Dict[str, Any]] = Field(default=None, description="User context")
    priority: Optional[str] = Field(default="normal", description="Workflow priority")
    timeout_seconds: Optional[int] = Field(default=300, description="Workflow timeout")

class WorkflowResponse(BaseModel):
    """Response model for workflow execution"""
    workflow_id: str = Field(..., description="Unique workflow identifier")
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Workflow status")
    results: Dict[str, Any] = Field(..., description="Workflow results")
    agents_executed: List[str] = Field(..., description="List of agents that were executed")
    execution_time_seconds: Optional[float] = Field(None, description="Total execution time")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentStatus(BaseModel):
    """Status model for individual agents"""
    agent_name: str = Field(..., description="Name of the agent")
    is_healthy: bool = Field(..., description="Whether agent is healthy")
    last_check: datetime = Field(..., description="Last health check timestamp")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed health info")

class CoordinatorMetrics(BaseModel):
    """Metrics model for coordinator performance"""
    total_workflows: int = Field(..., description="Total workflows executed")
    successful_workflows: int = Field(..., description="Number of successful workflows")
    failed_workflows: int = Field(..., description="Number of failed workflows")
    average_execution_time: float = Field(..., description="Average execution time in seconds")
    active_workflows_count: int = Field(..., description="Currently active workflows")
    agent_health: Dict[str, bool] = Field(..., description="Health status of all agents")
    timestamp: datetime = Field(..., description="Metrics timestamp")

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class WorkflowError(BaseModel):
    """Model for workflow errors"""
    error_id: str = Field(..., description="Unique error identifier")
    severity: ErrorSeverity = Field(..., description="Error severity")
    agent: Optional[str] = Field(None, description="Agent that caused the error")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    suggested_action: Optional[str] = Field(None, description="Suggested action to resolve error")

class RetryPolicy(BaseModel):
    """Retry policy configuration"""
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay_seconds: float = Field(default=1.0, description="Delay between retries")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff")
    retry_on_errors: List[str] = Field(
        default_factory=lambda: ["timeout", "connection_error", "rate_limit"],
        description="Error types to retry on"
    )

class FallbackStrategy(BaseModel):
    """Fallback strategy configuration"""
    strategy_type: str = Field(..., description="Type of fallback strategy")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    enabled: bool = Field(default=True, description="Whether fallback is enabled")

class WorkflowConfig(BaseModel):
    """Configuration for workflow execution"""
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy, description="Retry configuration")
    fallback_strategies: List[FallbackStrategy] = Field(
        default_factory=list, description="Fallback strategies"
    )
    timeout_seconds: int = Field(default=300, description="Workflow timeout")
    enable_parallel_execution: bool = Field(default=True, description="Enable parallel agent execution")
    log_level: str = Field(default="INFO", description="Logging level")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache time-to-live")

# Default configurations
DEFAULT_WORKFLOW_CONFIG = WorkflowConfig()

DEFAULT_RETRY_POLICIES = {
    "conservative": RetryPolicy(max_retries=2, retry_delay_seconds=2.0),
    "aggressive": RetryPolicy(max_retries=5, retry_delay_seconds=0.5),
    "none": RetryPolicy(max_retries=0)
}

DEFAULT_FALLBACK_STRATEGIES = {
    "perception_fallback": FallbackStrategy(
        strategy_type="mock_analysis",
        parameters={"confidence": 0.3, "message": "Image analysis unavailable, using fallback"},
        enabled=True
    ),
    "knowledge_fallback": FallbackStrategy(
        strategy_type="dynamic_response",
        parameters={"message": "Knowledge search unavailable, generating contextual guidance"},
        enabled=True
    ),
    "email_fallback": FallbackStrategy(
        strategy_type="simple_notification",
        parameters={"template": "basic", "include_partial_results": True},
        enabled=True
    )
}
