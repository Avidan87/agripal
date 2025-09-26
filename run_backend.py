#!/usr/bin/env python3
"""
🚀 AgriPal Backend Startup Script
Proper entry point for running the AgriPal backend with correct Python path setup.
"""
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import and run the backend
if __name__ == "__main__":
    import uvicorn
    from backend.config import settings
    
    print("🌾 Starting AgriPal Backend...")
    print(f"📍 Project root: {project_root}")
    print(f"🔧 Environment: {settings.ENVIRONMENT}")
    print(f"🌐 Server binding to: {settings.HOST}:{settings.PORT}")
    print(f"🔗 Access your app at: http://localhost:{settings.PORT}")
    print(f"📚 API Documentation: http://localhost:{settings.PORT}/docs")
    
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
