import os
from typing import Dict, List, Any, Optional
import json
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import MCP-specific modules
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

# Import mem0 for memory management
from mem0 import MemoryClient

class MemoryManager:
    def __init__(self):
        """Initialize the memory manager with mem0 client"""
        # Get API key from environment or empty string for testing
        api_key = os.getenv("MEM0_API_KEY", "")
        self.client = MemoryClient(api_key=api_key)
        
    def add_memory(self, messages: List[Dict[str, str]], user_id: str = "default") -> Dict[str, Any]:
        """
        Add messages to memory
        
        Args:
            messages: List of message objects with role and content
            user_id: Identifier for the user
        """
        try:
            # Add messages to mem0
            self.client.add(messages, user_id=user_id)
            return {
                "success": True,
                "message": f"Successfully added {len(messages)} messages to memory for user {user_id}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_memory(self, query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
        """
        Search memory for relevant information
        
        Args:
            query: The search query
            user_id: Identifier for the user
            limit: Maximum number of results to return
        """
        try:
            # Search mem0 for relevant memories
            results = self.client.search(query, user_id=user_id, limit=limit)
            
            return {
                "success": True,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_conversation(self, user_message: str, assistant_message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Add a conversation exchange to memory
        
        Args:
            user_message: The user's message
            assistant_message: The assistant's response
            user_id: Identifier for the user
        """
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        
        return self.add_memory(messages, user_id)

# Initialize memory manager
memory_manager = MemoryManager()

# Create FastMCP instance
mcp = FastMCP("Memory Manager")

@mcp.tool()
async def add_memory(messages: List[Dict[str, str]], user_id: str = "default") -> Dict[str, Any]:
    """
    Add messages to memory
    
    Args:
        messages: List of message objects with role and content
        user_id: Identifier for the user
    """
    result = memory_manager.add_memory(messages, user_id)
    return {"type": "text/plain", "text": json.dumps(result)}

@mcp.tool()
async def search_memory(query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
    """
    Search memory for relevant information
    
    Args:
        query: The search query
        user_id: Identifier for the user
        limit: Maximum number of results to return
    """
    result = memory_manager.search_memory(query, user_id, limit)
    return {"type": "text/plain", "text": json.dumps(result)}

@mcp.tool()
async def add_conversation(user_message: str, assistant_message: str, user_id: str = "default") -> Dict[str, Any]:
    """
    Add a conversation exchange to memory
    
    Args:
        user_message: The user's message
        assistant_message: The assistant's response
        user_id: Identifier for the user
    """
    result = memory_manager.add_conversation(user_message, assistant_message, user_id)
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
app = FastAPI(title="MCP Memory Server")

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
app.mount("/mcp", sse_app)

# Add root path handler
@app.get("/")
async def root():
    """Return information about the server"""
    return {
        "message": "MCP Memory Server",
        "description": "Store and retrieve memories using mem0",
        "version": "1.0.0",
        "mcp_endpoint": "/mcp/sse",
    }

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    print(f"Starting MCP Memory Server on {host}:{port}")
    print(f"Root API info: http://{host}:{port}/")
    print(f"MCP Endpoint (Claude/Cursor): http://{host}:{port}/mcp/sse")
    uvicorn.run(app, host=host, port=port) 