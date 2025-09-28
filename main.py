#!/usr/bin/env python3
"""
Railway entry point for AGRIPAL
This file allows Railway to auto-detect and start the FastAPI application
"""
import os
import sys
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    from backend.main import app
    
    # Railway configuration  
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))  # Railway default port
    
    print(f"🚀 Starting AGRIPAL on Railway...")
    print(f"🌐 Host: {host}")
    print(f"🔌 Port: {port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info"
    )
