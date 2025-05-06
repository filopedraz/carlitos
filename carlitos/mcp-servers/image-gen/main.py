import os
from typing import Dict, List, Any, Optional, Tuple
import openai
import base64
from io import BytesIO
from PIL import Image
import json
import uuid
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import asyncio
from pydantic import BaseModel

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
    
    def generate_image_dalle(self, prompt: str, size: str = "1024x1024", n: int = 1) -> Dict[str, Any]:
        """Generate an image using OpenAI's DALL-E 3 model"""
        try:
            response = openai.Image.create(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                n=n
            )
            return {
                "success": True,
                "data": response["data"][0]["url"] if response["data"] else None,
                "model": "dall-e-3"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def generate_image_4o(self, prompt: str, size: str = "1024x1024", n: int = 1, 
                         transparent_background: bool = False, 
                         referenced_image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate an image using OpenAI's GPT-4o model (when available)"""
        try:
            # Note: This is a placeholder for the GPT-4o API
            # The actual implementation will be updated when the API is released
            raise NotImplementedError("GPT-4o image generation API is not yet available")
            
            # Expected future implementation:
            # response = openai.Image.create(
            #     model="gpt-4o",
            #     prompt=prompt,
            #     size=size,
            #     n=n
            # )
            # return {
            #     "success": True,
            #     "data": response["data"][0]["url"] if response["data"] else None,
            #     "model": "gpt-4o"
            # }
        except Exception as e:
            # Fallback to DALL-E 3
            return self.generate_image_dalle(prompt, size, n)

    def edit_image(self, image: str, mask: Optional[str], prompt: str) -> Dict[str, Any]:
        """Edit an existing image using OpenAI's API"""
        try:
            # Convert base64 to image
            image_data = base64.b64decode(image)
            image_obj = Image.open(BytesIO(image_data))
            
            mask_obj = None
            if mask:
                mask_data = base64.b64decode(mask)
                mask_obj = Image.open(BytesIO(mask_data))

            # Save to temporary files
            image_path = "temp_image.png"
            mask_path = "temp_mask.png" if mask else None
            
            image_obj.save(image_path)
            if mask_obj:
                mask_obj.save(mask_path)

            with open(image_path, "rb") as image_file:
                if mask_path:
                    with open(mask_path, "rb") as mask_file:
                        response = openai.Image.create_edit(
                            image=image_file,
                            mask=mask_file,
                            prompt=prompt
                        )
                else:
                    response = openai.Image.create_edit(
                        image=image_file,
                        prompt=prompt
                    )

            # Cleanup
            os.remove(image_path)
            if mask_path:
                os.remove(mask_path)

            return {
                "success": True,
                "data": response["data"][0]["url"] if response["data"] else None
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
    Generate an image using OpenAI's GPT-4o model (falls back to DALL-E 3 if 4o is not available)
    
    Args:
        prompt: Text description of the image to generate
        size: Size of the generated image (1024x1024, 1024x1792, or 1792x1024)
        n: Number of images to generate (1-10)
        transparent_background: Whether to generate image with transparent background (GPT-4o only)
        referenced_image_ids: IDs of images to reference for context (GPT-4o only)
    """
    try:
        result = server.generate_image_4o(
            prompt=prompt,
            size=size,
            n=n,
            transparent_background=transparent_background,
            referenced_image_ids=referenced_image_ids
        )
        return {"type": "text/plain", "text": json.dumps(result)}
    except NotImplementedError:
        # Fallback to DALL-E 3 if GPT-4o is not available
        result = server.generate_image_dalle(
            prompt=prompt,
            size=size,
            n=n
        )
        return {"type": "text/plain", "text": json.dumps(result)}

@mcp.tool()
async def edit_image(image: str, mask: Optional[str], prompt: str) -> Dict[str, Any]:
    """
    Edit an existing image using OpenAI's API
    
    Args:
        image: Base64 encoded image to edit
        mask: Optional base64 encoded mask image
        prompt: Text description of the desired edit
    """
    result = server.edit_image(
        image=image,
        mask=mask,
        prompt=prompt
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
        "description": "Generate and edit images using OpenAI's APIs",
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
    asyncio.create_task(process_image_request(request_id, request.prompt, request.size, request.n))
    
    return {"request_id": request_id}

async def process_image_request(request_id: str, prompt: str, size: str, n: int):
    try:
        # Call the image generator
        result = server.generate_image_dalle(prompt, size, n)
        
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
