"""
Main entry point for the banner classification API service.

Run with: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import logging
import sys
from src.api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

