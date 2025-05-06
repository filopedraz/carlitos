import os
from typing import Dict, Any, Optional, List
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

# MCP Tool Definitions
def register_tools():
    return {
        "openai_generate_image": {
            "name": "openai_generate_image",
            "description": "Generate an image using OpenAI's GPT-4o model (falls back to DALL-E 3 if 4o is not available)",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "size": {
                        "type": "string",
                        "description": "Size of the generated image (1024x1024, 1024x1792, or 1792x1024)",
                        "default": "1024x1024"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of images to generate (1-10)",
                        "default": 1
                    },
                    "transparent_background": {
                        "type": "boolean",
                        "description": "Whether to generate image with transparent background (GPT-4o only)",
                        "default": False
                    },
                    "referenced_image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of images to reference for context (GPT-4o only)",
                        "default": []
                    }
                },
                "required": ["prompt"]
            }
        },
        "openai_edit_image": {
            "name": "openai_edit_image",
            "description": "Edit an existing image using OpenAI's API",
            "parameters": {
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Base64 encoded image to edit"
                    },
                    "mask": {
                        "type": "string",
                        "description": "Optional base64 encoded mask image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the desired edit"
                    }
                },
                "required": ["image", "prompt"]
            }
        }
    }

# Initialize server
server = OpenAIImageGenServer()

# MCP handlers
def handle_openai_generate_image(params: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return server.generate_image_4o(
            prompt=params["prompt"],
            size=params.get("size", "1024x1024"),
            n=params.get("n", 1),
            transparent_background=params.get("transparent_background", False),
            referenced_image_ids=params.get("referenced_image_ids", [])
        )
    except NotImplementedError:
        # Fallback to DALL-E 3 if GPT-4o is not available
        return server.generate_image_dalle(
            prompt=params["prompt"],
            size=params.get("size", "1024x1024"),
            n=params.get("n", 1)
        )

def handle_openai_edit_image(params: Dict[str, Any]) -> Dict[str, Any]:
    return server.edit_image(
        image=params["image"],
        mask=params.get("mask"),
        prompt=params["prompt"]
    )

# Map tool names to handlers
HANDLERS = {
    "openai_generate_image": handle_openai_generate_image,
    "openai_edit_image": handle_openai_edit_image
}

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

# Add root path handler
@app.get("/")
async def root():
    """Redirect to tools endpoint"""
    return {"message": "MCP Image Generation Server", "endpoints": {
        "tools": "/tools", 
        "api": "/api/mcp"
    }}

# In-memory request tracking
requests = {}

# Standard MCP endpoints
class MCPRequest(BaseModel):
    id: Optional[str] = None
    tool: str
    params: Dict[str, Any]

# MCP-compatible endpoints
@app.post("/")
async def mcp_root(request: Request):
    """MCP protocol compatible endpoint"""
    data = await request.json()
    method = data.get("method")
    
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": data.get("id"), "result": {}}
    
    elif method == "list_tools":
        tools = register_tools()
        tool_list = []
        for name, tool in tools.items():
            tool_list.append(tool)
        return {"jsonrpc": "2.0", "id": data.get("id"), "result": {"tools": tool_list}}
    
    elif method == "call_tool":
        id = data.get("id")
        params = data.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in HANDLERS:
            return {"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": f"Unknown tool: {tool_name}"}}
        
        try:
            result = HANDLERS[tool_name](arguments)
            return {
                "jsonrpc": "2.0", 
                "id": id, 
                "result": {
                    "content": [
                        {"type": "text/plain", "text": json.dumps(result)}
                    ]
                }
            }
        except Exception as e:
            return {"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": str(e)}}
    
    return {"jsonrpc": "2.0", "id": data.get("id"), "error": {"code": -32601, "message": "Method not found"}}

@app.post("/api/mcp")
async def mcp_endpoint(request: MCPRequest):
    request_id = request.id or str(uuid.uuid4())
    tool_name = request.tool
    params = request.params
    
    # Store request for SSE updates
    requests[request_id] = {
        "status": "processing",
        "result": None,
        "error": None
    }
    
    # Process request asynchronously
    asyncio.create_task(process_request(request_id, tool_name, params))
    
    return {"request_id": request_id}

async def process_request(request_id: str, tool_name: str, params: Dict[str, Any]):
    try:
        if tool_name not in HANDLERS:
            requests[request_id] = {
                "status": "error",
                "result": None,
                "error": f"Unknown tool: {tool_name}"
            }
            return
        
        # Call the appropriate handler
        result = HANDLERS[tool_name](params)
        
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

@app.get("/api/mcp/sse/{request_id}")
async def sse_endpoint(request_id: str):
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

@app.get("/tools")
async def get_tools():
    """Endpoint to get available tools and their definitions"""
    return register_tools()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"Starting MCP Image Generation Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
