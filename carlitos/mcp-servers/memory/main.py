import os
from typing import Dict, List, Any
import logging
import traceback
import json
import sys
from fastmcp import FastMCP
from mem0 import MemoryClient

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("memory_server")

class MemoryManager:
    def __init__(self):
        """Initialize the memory manager with mem0 client"""
        # Get API key from environment or empty string for testing
        api_key = os.getenv("MEM0_API_KEY", "")
        logger.info(f"Initializing MemoryManager with api_key: {'[SET]' if api_key else '[NOT SET]'}")
        self.client = MemoryClient(api_key=api_key)
        logger.info("MemoryClient initialized successfully")
        
    def add_memory(self, messages: List[Dict[str, str]], user_id: str = "default") -> Dict[str, Any]:
        """
        Add messages to memory
        
        Args:
            messages: List of message objects with role and content
            user_id: Identifier for the user
        """
        logger.info(f"MemoryManager.add_memory called with {len(messages)} messages for user_id: '{user_id}'")
        try:
            # Add messages to mem0
            self.client.add(messages, user_id=user_id)
            response_data = {
                "success": True,
                "message": f"Successfully added {len(messages)} messages to memory for user {user_id}"
            }
            logger.info(f"MemoryManager.add_memory returning: {response_data}")
            return response_data
        except Exception as e:
            logger.error(f"Error in MemoryManager.add_memory: {str(e)}", exc_info=True)
            response_data = {
                "success": False,
                "error": str(e)
            }
            logger.info(f"MemoryManager.add_memory returning error response: {response_data}")
            return response_data
    
    def search_memory(self, query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
        """
        Search memory for relevant information
        
        Args:
            query: The search query
            user_id: Identifier for the user
            limit: Maximum number of results to return
        """
        logger.info(f"MemoryManager.search_memory called with query: '{query}', user_id: '{user_id}', limit: {limit}")
        
        # Check if API key is set
        if not hasattr(self.client, 'api_key') or not self.client.api_key:
            logger.error("API key not set for mem0 client")
            return {
                "success": False,
                "error": "API key not set for memory service"
            }
            
        try:
            # Search mem0 for relevant memories
            mem0_results = self.client.search(query, user_id=user_id, limit=limit)
            logger.info(f"mem0 client returned: {type(mem0_results).__name__}, content: {mem0_results}")
            
            # Validate results
            if mem0_results is None:
                logger.warning("mem0 returned None result")
                mem0_results = []
                
            # Ensure we have a list (even if empty)
            if not isinstance(mem0_results, list):
                logger.error(f"mem0 returned non-list result: {mem0_results}")
                return {
                    "success": False,
                    "error": f"Invalid response from memory service: expected list, got {type(mem0_results).__name__}"
                }
            
            response_data = {
                "success": True,
                "results": mem0_results
            }
            logger.info(f"MemoryManager.search_memory returning: {response_data}")
            return response_data
        except Exception as e:
            logger.error(f"Error in MemoryManager.search_memory: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            response_data = {
                "success": False,
                "error": str(e)
            }
            logger.info(f"MemoryManager.search_memory returning error response: {response_data}")
            return response_data
    
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
    logger.info(f"Tool 'add_memory' called with {len(messages)} messages for user_id: '{user_id}'")
    result = memory_manager.add_memory(messages, user_id)
    logger.info(f"Tool 'add_memory' received from manager: {result}")
    return result

@mcp.tool()
async def search_memory(query: str, user_id: str = "default", limit: int = 5) -> Dict[str, Any]:
    """
    Search memory for relevant information
    
    Args:
        query: The search query
        user_id: Identifier for the user
        limit: Maximum number of results to return
    """
    logger.info(f"Tool 'search_memory' called with query: '{query}', user_id: '{user_id}', limit: {limit}")
    try:
        logger.debug("About to call memory_manager.search_memory")
        result = memory_manager.search_memory(query, user_id, limit)
        logger.debug(f"memory_manager.search_memory returned, type: {type(result).__name__}")
        logger.info(f"Tool 'search_memory' received from manager: {json.dumps(result)}")
        
        # Ensure result is properly formatted as a dictionary
        if not isinstance(result, dict):
            logger.error(f"search_memory returned non-dict result: {result}")
            return {"success": False, "error": "Invalid result format"}
        
        # Always return a properly formatted dict that won't cause JSON decode errors
        if not result.get("success", False):
            logger.warning(f"search_memory error: {result.get('error', 'Unknown error')}")
        
        return result
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error in search_memory tool: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return {
            "success": False,
            "error": f"Tool error: {str(e)}"
        }

@mcp.tool()
async def add_conversation(user_message: str, assistant_message: str, user_id: str = "default") -> Dict[str, Any]:
    """
    Add a conversation exchange to memory
    
    Args:
        user_message: The user's message
        assistant_message: The assistant's response
        user_id: Identifier for the user
    """
    logger.info(f"Tool 'add_conversation' called for user_id: '{user_id}'")
    result = memory_manager.add_conversation(user_message, assistant_message, user_id)
    logger.info(f"Tool 'add_conversation' received from manager: {result}")
    return result

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)