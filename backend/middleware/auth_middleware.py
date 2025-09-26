"""
🔐 Authentication and Rate Limiting Middleware
Security middleware for AgriPal AI agent endpoints with JWT authentication and rate limiting.
"""
import jwt
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from fastapi import HTTPException, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque
import redis.asyncio as redis

from ..config import settings

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Advanced rate limiter with multiple strategies
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        self.local_storage = defaultdict(deque)
        self.use_redis = bool(redis_url)
        
        if self.use_redis:
            asyncio.create_task(self._init_redis(redis_url))
    
    async def _init_redis(self, redis_url: str):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("✅ Redis rate limiter initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis initialization failed, using local storage: {str(e)}")
            self.use_redis = False
    
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int,
        identifier: str = "default"
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is allowed based on rate limit
        
        Args:
            key: Unique identifier for rate limiting (e.g., user_id, ip_address)
            limit: Maximum requests allowed in window
            window_seconds: Time window in seconds
            identifier: Additional identifier for different rate limits
            
        Returns:
            Tuple of (is_allowed, info_dict)
        """
        if self.use_redis and self.redis_client:
            return await self._redis_rate_limit(key, limit, window_seconds, identifier)
        else:
            return await self._local_rate_limit(key, limit, window_seconds, identifier)
    
    async def _redis_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int, 
        identifier: str
    ) -> Tuple[bool, Dict[str, int]]:
        """Redis-based rate limiting using sliding window"""
        try:
            current_time = int(time.time())
            redis_key = f"rate_limit:{identifier}:{key}"
            
            # Clean old entries and count current requests
            pipeline = self.redis_client.pipeline()
            pipeline.zremrangebyscore(redis_key, 0, current_time - window_seconds)
            pipeline.zcard(redis_key)
            pipeline.expire(redis_key, window_seconds)
            
            results = await pipeline.execute()
            current_requests = results[1]
            
            if current_requests >= limit:
                # Get time until reset
                oldest_request = await self.redis_client.zrange(redis_key, 0, 0, withscores=True)
                time_until_reset = (
                    int(oldest_request[0][1]) + window_seconds - current_time
                    if oldest_request else window_seconds
                )
                
                return False, {
                    "requests_made": current_requests,
                    "limit": limit,
                    "window_seconds": window_seconds,
                    "time_until_reset": max(0, time_until_reset)
                }
            
            # Add current request
            await self.redis_client.zadd(redis_key, {str(current_time): current_time})
            
            return True, {
                "requests_made": current_requests + 1,
                "limit": limit,
                "window_seconds": window_seconds,
                "time_until_reset": 0
            }
            
        except Exception as e:
            logger.error(f"❌ Redis rate limiting failed: {str(e)}")
            # Fallback to local rate limiting
            return await self._local_rate_limit(key, limit, window_seconds, identifier)
    
    async def _local_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window_seconds: int, 
        identifier: str
    ) -> Tuple[bool, Dict[str, int]]:
        """Local memory-based rate limiting"""
        current_time = time.time()
        storage_key = f"{identifier}:{key}"
        
        # Clean old entries
        while (self.local_storage[storage_key] and 
               current_time - self.local_storage[storage_key][0] > window_seconds):
            self.local_storage[storage_key].popleft()
        
        current_requests = len(self.local_storage[storage_key])
        
        if current_requests >= limit:
            # Calculate time until reset
            oldest_request_time = (
                self.local_storage[storage_key][0] 
                if self.local_storage[storage_key] else current_time
            )
            time_until_reset = max(0, oldest_request_time + window_seconds - current_time)
            
            return False, {
                "requests_made": current_requests,
                "limit": limit,
                "window_seconds": window_seconds,
                "time_until_reset": int(time_until_reset)
            }
        
        # Add current request
        self.local_storage[storage_key].append(current_time)
        
        return True, {
            "requests_made": current_requests + 1,
            "limit": limit,
            "window_seconds": window_seconds,
            "time_until_reset": 0
        }

class JWTAuthenticator:
    """
    JWT token authentication and validation
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_blacklist = set()  # In production, use Redis
    
    def create_token(
        self, 
        user_id: str, 
        email: str, 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        payload = {
            "user_id": user_id,
            "email": email,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, any]:
        """Verify and decode JWT token"""
        try:
            if token in self.token_blacklist:
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(status_code=401, detail="Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def revoke_token(self, token: str):
        """Add token to blacklist"""
        self.token_blacklist.add(token)

class AgentAuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication and rate limiting middleware for agent endpoints
    """
    
    def __init__(self, app, enable_auth: bool = True, enable_rate_limiting: bool = True):
        super().__init__(app)
        self.enable_auth = enable_auth
        self.enable_rate_limiting = enable_rate_limiting
        
        # Initialize components
        self.jwt_auth = JWTAuthenticator(settings.JWT_SECRET_KEY)
        self.rate_limiter = RateLimiter(getattr(settings, 'REDIS_URL', None))
        
        # Rate limiting rules (requests per time window)
        self.rate_limits = {
            "/api/v1/agents/health": (100, 60),  # 100 requests per minute
            "/api/v1/agents/metrics": (50, 60),   # 50 requests per minute
            "/api/v1/agents/perception": (20, 300),  # 20 requests per 5 minutes
            "/api/v1/agents/knowledge": (50, 300),   # 50 requests per 5 minutes
            "/api/v1/agents/email": (10, 300),       # 10 requests per 5 minutes
            "/api/v1/agents/consultation": (10, 600), # 10 requests per 10 minutes
            "/api/v1/agents/batch": (5, 600),        # 5 requests per 10 minutes
            "default": (30, 300)  # Default: 30 requests per 5 minutes
        }
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            "/api/v1/agents/health",
            "/api/v1/agents/auth/login",
            "/docs",
            "/openapi.json",
            "/redoc"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication and rate limiting"""
        
        # Skip middleware for non-agent endpoints
        if not request.url.path.startswith("/api/v1/agents"):
            return await call_next(request)
        
        try:
            # Step 1: Authentication
            user_info = None
            if self.enable_auth and request.url.path not in self.public_endpoints:
                user_info = await self._authenticate_request(request)
                request.state.user = user_info
            
            # Step 2: Rate Limiting
            if self.enable_rate_limiting:
                rate_limit_result = await self._check_rate_limit(request, user_info)
                if not rate_limit_result["allowed"]:
                    return self._create_rate_limit_response(rate_limit_result)
            
            # Step 3: Process request
            response = await call_next(request)
            
            # Step 4: Add security headers
            response = self._add_security_headers(response, rate_limit_result if self.enable_rate_limiting else None)
            
            return response
            
        except HTTPException as e:
            # Convert HTTPException to Response
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            logger.error(f"❌ Middleware error: {str(e)}")
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
    
    async def _authenticate_request(self, request: Request) -> Dict[str, any]:
        """Authenticate the request using JWT"""
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        token = auth_header.split(" ")[1]
        user_info = self.jwt_auth.verify_token(token)
        
        logger.info(f"🔓 Authenticated user: {user_info.get('email')}")
        return user_info
    
    async def _check_rate_limit(self, request: Request, user_info: Optional[Dict]) -> Dict:
        """Check rate limiting for the request"""
        # Determine rate limit key (user ID or IP address)
        if user_info:
            rate_limit_key = user_info["user_id"]
            identifier = f"user:{user_info['user_id']}"
        else:
            # Use IP address for unauthenticated requests
            client_ip = request.client.host if request.client else "unknown"
            rate_limit_key = client_ip
            identifier = f"ip:{client_ip}"
        
        # Get rate limit for endpoint
        endpoint_path = request.url.path
        limit, window = self._get_rate_limit_for_endpoint(endpoint_path)
        
        # Check rate limit
        allowed, info = await self.rate_limiter.is_allowed(
            key=rate_limit_key,
            limit=limit,
            window_seconds=window,
            identifier=identifier
        )
        
        return {
            "allowed": allowed,
            "limit": limit,
            "window": window,
            "requests_made": info["requests_made"],
            "time_until_reset": info["time_until_reset"]
        }
    
    def _get_rate_limit_for_endpoint(self, path: str) -> Tuple[int, int]:
        """Get rate limit configuration for specific endpoint"""
        for endpoint_pattern, (limit, window) in self.rate_limits.items():
            if endpoint_pattern == "default":
                continue
            if path.startswith(endpoint_pattern):
                return limit, window
        
        # Return default rate limit
        return self.rate_limits["default"]
    
    def _create_rate_limit_response(self, rate_limit_result: Dict):
        """Create rate limit exceeded response"""
        from fastapi.responses import JSONResponse
        
        headers = {
            "X-RateLimit-Limit": str(rate_limit_result["limit"]),
            "X-RateLimit-Remaining": str(max(0, rate_limit_result["limit"] - rate_limit_result["requests_made"])),
            "X-RateLimit-Reset": str(int(time.time()) + rate_limit_result["time_until_reset"]),
            "Retry-After": str(rate_limit_result["time_until_reset"])
        }
        
        return JSONResponse(
            status_code=429,
            headers=headers,
            content={
                "detail": "Rate limit exceeded",
                "limit": rate_limit_result["limit"],
                "window_seconds": rate_limit_result["window"],
                "retry_after_seconds": rate_limit_result["time_until_reset"]
            }
        )
    
    def _add_security_headers(self, response: Response, rate_limit_info: Optional[Dict] = None):
        """Add security headers to response"""
        # Basic security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Rate limit headers
        if rate_limit_info:
            response.headers["X-RateLimit-Limit"] = str(rate_limit_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, rate_limit_info["limit"] - rate_limit_info["requests_made"])
            )
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + rate_limit_info["window"])
        
        return response

# Dependency for extracting authenticated user
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, any]:
    """Dependency to get current authenticated user"""
    authenticator = JWTAuthenticator(settings.JWT_SECRET_KEY)
    return authenticator.verify_token(credentials.credentials)

# Optional dependency for authenticated user (allows None)
async def get_current_user_optional(request: Request) -> Optional[Dict[str, any]]:
    """Optional dependency to get current user (returns None if not authenticated)"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        authenticator = JWTAuthenticator(settings.JWT_SECRET_KEY)
        return authenticator.verify_token(token)
    except:
        return None

# Export components
__all__ = [
    "AgentAuthMiddleware", 
    "JWTAuthenticator", 
    "RateLimiter", 
    "get_current_user", 
    "get_current_user_optional"
]
