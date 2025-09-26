#!/usr/bin/env python3
"""
🌾 AgriPal Frontend Launcher
Minimal launcher for the AgriPal Gradio UI application.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import backend modules
sys.path.append(str(Path(__file__).parent.parent))

from agripal_ui import create_agripal_interface

def main():
    """Launch AgriPal Gradio UI"""
    print("🌾 Starting AgriPal UI...")
    
     # Get API URL from environment or use default
     api_url = os.getenv("AGRIPAL_API_URL", "http://localhost:8000/api/v1/agents")
     
     print(f"🔗 API URL: {api_url}")
     print(f"🌐 Frontend will be available at: http://localhost:7860")
    
    # Create and launch interface
    app = create_agripal_interface(api_url)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True
    )
if __name__ == "__main__":
    main()

