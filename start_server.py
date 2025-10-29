#!/usr/bin/env python3
"""
Startup script for the YOLO Loss Prediction API server
"""

import sys
from pathlib import Path
import uvicorn

# Add server directory to path
server_dir = Path(__file__).parent / "server"
sys.path.append(str(server_dir))

if __name__ == "__main__":
    print("🚀 Starting YOLO Loss Prediction API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
