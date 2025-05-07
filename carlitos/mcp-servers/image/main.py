import os
from typing import Dict, List, Any, Optional
import logging
import traceback
import sys
import json
from fastmcp import FastMCP
from openai import OpenAI
import time
import base64
import uuid
from pathlib import Path
from flask import Flask, send_from_directory
from threading import Thread, Event
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("image_server")

# Global variables for reload functionality
shutdown_event = Event()
observer = None
flask_server = None

# Create directory for images if it doesn't exist
IMAGE_DIR = Path(__file__).parent / "images"
IMAGE_DIR.mkdir(exist_ok=True)

# Configure a simple Flask server to serve images
flask_app = Flask(__name__)
IMAGE_SERVER_PORT = 8080  # Choose an available port
IMAGE_SERVER_URL = f"http://localhost:{IMAGE_SERVER_PORT}/images/"

@flask_app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(str(IMAGE_DIR), filename)

# Start Flask server in a separate thread
def start_image_server():
    flask_app.run(host='0.0.0.0', port=IMAGE_SERVER_PORT, debug=False, threaded=True)

image_server_thread = Thread(target=start_image_server, daemon=True)
image_server_thread.start()
logger.info(f"Started image server at {IMAGE_SERVER_URL}")

class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            logger.info(f"Python file changed: {event.src_path}. Triggering reload...")
            # Set shutdown event to trigger restart
            shutdown_event.set()
            # Signal the main process
            os.kill(os.getpid(), signal.SIGTERM)

def start_file_watcher(path="."):
    global observer
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    logger.info(f"File watcher started for directory: {path}")

class OpenAIImageGenServer:
    def __init__(self):
        logger.info("Initializing OpenAIImageGenServer...")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        self.client = OpenAI(api_key=api_key)
        logger.info("OpenAIImageGenServer initialized successfully.")
    
    def generate_image_4o(self, prompt: str, size: str = "1024x1024", n: int = 1, 
                         transparent_background: bool = False, 
                         referenced_image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate an image using OpenAI's GPT-4o image generation API (gpt-image-1)"""
        logger.info(f"generate_image_4o called with prompt: '{prompt}', size: {size}, n: {n}, transparent: {transparent_background}")
        try:
            # Create parameters for the request
            params = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "size": size,
                "n": n,
                "quality": "medium",  # Options: low, medium, high, auto
            }
            
            # Add transparent background if requested
            if transparent_background:
                params["background"] = "transparent"
                params["output_format"] = "png"  # Transparency requires PNG or WebP
            
            logger.info(f"Sending request with params: {params}")
            
            # Make the API call using the new client.images.generate method
            response = self.client.images.generate(**params)
            
            # gpt-image-1 returns base64 data, save to file and return local URL
            image_url = None
            if response.data and len(response.data) > 0:
                # Get the base64 data
                b64_data = response.data[0].b64_json
                if b64_data:
                    # Save image to file with unique filename
                    image_format = "png" if transparent_background else "jpg"
                    filename = f"{uuid.uuid4()}.{image_format}"
                    file_path = IMAGE_DIR / filename
                    
                    with open(file_path, "wb") as f:
                        f.write(base64.b64decode(b64_data))
                    
                    # Create local URL for the saved image
                    image_url = f"{IMAGE_SERVER_URL}{filename}"
                    logger.info(f"Saved image to {file_path}, local URL: {image_url}")
            
            result_data = {
                "success": True,
                "data": image_url,
                "model": "gpt-image-1"
            }
            logger.info(f"generate_image_4o successful, returning: {result_data}")
            return result_data
        except Exception as e:
            logger.error(f"Error in generate_image_4o: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

# Initialize server
server = OpenAIImageGenServer()

# Create FastMCP instance
mcp = FastMCP("GPT-Image-1 Generator")

@mcp.tool()
async def generate_image(prompt: str, size: str = "1024x1024", n: int = 1, 
                       transparent_background: bool = False, 
                       referenced_image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate an image using OpenAI's GPT-4o Image Generation API (gpt-image-1)
    
    Args:
        prompt: Text description of the image to generate
        size: Size of the generated image (1024x1024, 1024x1792, or 1792x1024)
        n: Number of images to generate (must be 1 for this model version)
        transparent_background: Whether to generate image with transparent background
        referenced_image_ids: IDs of images to reference for context (not used by gpt-image-1)
    """
    logger.info(f"Tool 'generate_image' called with prompt: '{prompt}', size: {size}, n: {n}, transparent: {transparent_background}")
    try:
        # Note: gpt-image-1 (DALL-E 3 via API) currently only supports n=1.
        result = server.generate_image_4o(
            prompt=prompt,
            size=size,
            n=n, # For gpt-image-1 (DALL-E 3 quality), n must be 1.
            transparent_background=transparent_background,
            referenced_image_ids=referenced_image_ids # This param is for other models/contexts
        )
        logger.info(f"Tool 'generate_image' received from server: {json.dumps(result)}")
        
        # Check if we actually have data
        if not result.get("data"):
            logger.warning("No image URL was returned")
            result["success"] = False
            result["error"] = "No image URL was returned from the API"
            
        return result
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error in generate_image tool: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return {
            "success": False,
            "error": f"Tool error: {str(e)}"
        }

def run_server_with_reload():
    global observer
    
    # Set up signal handling for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        shutdown_event.set()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    while True:
        # Reset shutdown event for new server instance
        shutdown_event.clear()
        
        # Start the file watcher
        start_file_watcher(Path(__file__).parent)
        
        try:
            logger.info("Starting MCP server...")
            # Run in a separate thread so we can check for shutdown events
            server_thread = Thread(target=lambda: mcp.run(transport="sse", host="0.0.0.0", port=8000))
            server_thread.daemon = True
            server_thread.start()
            
            # Wait for shutdown event
            while not shutdown_event.is_set():
                time.sleep(0.5)
                
            # If we get here, we need to restart
            if observer:
                logger.info("Stopping file watcher...")
                observer.stop()
                observer.join()
            
            # If shutdown was triggered by file change, restart
            if shutdown_event.is_set():
                logger.info("Restarting process...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                break
                
        except KeyboardInterrupt:
            logger.info("Server stopped by keyboard interrupt")
            break
        except Exception as e:
            logger.error(f"Error running server: {str(e)}", exc_info=True)
            break
        finally:
            if observer:
                observer.stop()
                observer.join()

if __name__ == "__main__":
    run_server_with_reload()
