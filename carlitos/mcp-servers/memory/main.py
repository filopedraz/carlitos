import os
from typing import Dict, List, Any
import logging
import traceback
import json
import sys
from fastmcp import FastMCP
from mem0 import MemoryClient
import time
import signal
from threading import Thread, Event
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("memory_server")

# Global variables for reload functionality
shutdown_event = Event()
observer = None

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
    
    # Return a clean, standardized response
    clean_result = {
        "success": result.get("success", False)
    }
    
    if "message" in result:
        clean_result["message"] = result["message"]
    
    if not result.get("success", False) and "error" in result:
        clean_result["error"] = result["error"]
        
    logger.debug(f"Returning clean result: {clean_result}")
    return clean_result

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
        
        # Ensure result is properly formatted as a dictionary
        if not isinstance(result, dict):
            logger.error(f"search_memory returned non-dict result: {result}")
            return {"success": False, "error": "Invalid result format"}
        
        # Always return a properly formatted dict that won't cause JSON decode errors
        if not result.get("success", False):
            logger.warning(f"search_memory error: {result.get('error', 'Unknown error')}")
        
        # Ensure we're returning a clean dict with no extra text or prefixes
        # This is critical for proper JSON handling in the mega_agent
        clean_result = {
            "success": result.get("success", False),
            "results": result.get("results", []) if result.get("success", False) else [],
        }
        
        if not result.get("success", False) and "error" in result:
            clean_result["error"] = result["error"]
            
        logger.debug(f"Returning clean result: {clean_result}")
        return clean_result
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
    
    # Return a clean, standardized response
    clean_result = {
        "success": result.get("success", False)
    }
    
    if "message" in result:
        clean_result["message"] = result["message"]
    
    if not result.get("success", False) and "error" in result:
        clean_result["error"] = result["error"]
        
    logger.debug(f"Returning clean result: {clean_result}")
    return clean_result

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