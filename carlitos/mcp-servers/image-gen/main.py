import os
from typing import Dict, List, Any, Optional, Tuple
import openai
import json
import uuid
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
from pydantic import BaseModel

# Import MCP-specific modules
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

class OpenAIImageGenServer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        openai.api_key = api_key
    
    def generate_image_4o(self, prompt: str, size: str = "1024x1024", n: int = 1, 
                         transparent_background: bool = False, 
                         referenced_image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate an image using OpenAI's GPT-4o image generation API (gpt-image-1)"""
        try:
            # Create parameters for the request
            params = {
                "model": "gpt-image-1",
                "prompt": prompt,
                "size": size,
                "n": n,
                "quality": "medium"  # Options: low, medium, high, auto
            }
            
            # Add transparent background if requested
            if transparent_background:
                params["background"] = "transparent"
                params["output_format"] = "png"  # Transparency requires PNG or WebP
            
            # Make the API call
            response = openai.Image.create(**params)
            
            return {
                "success": True,
                "data": response["data"][0]["url"] if response["data"] else None,
                "model": "gpt-image-1"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Initialize server
server = OpenAIImageGenServer()

# Create FastMCP instance
mcp = FastMCP("mix_server")

@mcp.tool()
async def generate_image(prompt: str, size: str = "1024x1024", n: int = 1, 
                       transparent_background: bool = False, 
                       referenced_image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Generate an image using OpenAI's GPT-4o Image Generation API (gpt-image-1)
    
    Args:
        prompt: Text description of the image to generate
        size: Size of the generated image (1024x1024, 1024x1792, or 1792x1024)
        n: Number of images to generate (1-10)
        transparent_background: Whether to generate image with transparent background
        referenced_image_ids: IDs of images to reference for context
    """
    result = server.generate_image_4o(
        prompt=prompt,
        size=size,
        n=n,
        transparent_background=transparent_background,
        referenced_image_ids=referenced_image_ids
    )
    return {"type": "text/plain", "text": json.dumps(result)}

# Create an SSE transport for the MCP server
def create_sse_server(mcp: FastMCP):
    """Create a Starlette app that handles SSE connections and message handling"""
    transport = SseServerTransport("/messages/")

    # Define handler functions
    async def handle_sse(request):
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await mcp._mcp_server.run(
                streams[0], streams[1], mcp._mcp_server.create_initialization_options()
            )
        return Response()

    # Create and return a Starlette application with the SSE route
    return Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=transport.handle_post_message),
        ]
    )

# FastAPI application setup
app = FastAPI(title="MCP Image Generation Server")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the SSE MCP server in the FastAPI app
sse_app = create_sse_server(mcp)
app.mount("/", sse_app)

# Add root path handler
@app.get("/info")
async def info():
    """Return information about the server"""
    return {
        "message": "MCP Image Generation Server",
        "description": "Generate images using OpenAI's GPT-4o API",
        "version": "1.0.0",
        "mcp_endpoint": "/sse"
    }

# In-memory request tracking for direct API usage
requests = {}

# Standard API endpoint (non-MCP)
class ImageGenRequest(BaseModel):
    id: Optional[str] = None
    prompt: str
    size: str = "1024x1024"
    n: int = 1
    transparent_background: bool = False

@app.post("/api/generate")
async def generate_image_endpoint(request: ImageGenRequest):
    """Direct API endpoint for image generation (non-MCP)"""
    request_id = request.id or str(uuid.uuid4())
    
    # Store request for SSE updates
    requests[request_id] = {
        "status": "processing",
        "result": None,
        "error": None
    }
    
    # Process request asynchronously
    asyncio.create_task(process_image_request(
        request_id, 
        request.prompt, 
        request.size, 
        request.n, 
        request.transparent_background
    ))
    
    return {"request_id": request_id}

async def process_image_request(request_id: str, prompt: str, size: str, n: int, transparent_background: bool = False):
    try:
        # Call the image generator
        result = server.generate_image_4o(
            prompt=prompt, 
            size=size, 
            n=n,
            transparent_background=transparent_background
        )
        
        # Update request status
        requests[request_id] = {
            "status": "complete",
            "result": result,
            "error": None
        }
    except Exception as e:
        requests[request_id] = {
            "status": "error",
            "result": None,
            "error": str(e)
        }

@app.get("/api/status/{request_id}")
async def request_status(request_id: str):
    """Check status of an image generation request"""
    if request_id not in requests:
        return {"status": "not_found"}
    return requests[request_id]

@app.get("/api/sse/{request_id}")
async def sse_api_endpoint(request_id: str):
    """SSE endpoint for tracking request progress (for direct API use)"""
    async def event_generator():
        while True:
            if request_id in requests:
                status = requests[request_id]["status"]
                
                if status == "complete":
                    result = requests[request_id]["result"]
                    yield {
                        "event": "done",
                        "data": json.dumps({"result": result})
                    }
                    # Clean up after sending final result
                    del requests[request_id]
                    break
                elif status == "error":
                    error = requests[request_id]["error"]
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": error})
                    }
                    # Clean up after sending error
                    del requests[request_id]
                    break
                else:
                    # Still processing
                    yield {
                        "event": "processing",
                        "data": json.dumps({"status": "processing"})
                    }
            
            # Wait before checking again
            await asyncio.sleep(0.5)
    
    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"Starting MCP Image Generation Server on {host}:{port}")
    print(f"SSE MCP endpoint available at http://{host}:{port}/sse")
    print(f"Direct API endpoint available at http://{host}:{port}/api/generate")
    uvicorn.run(app, host=host, port=port)
