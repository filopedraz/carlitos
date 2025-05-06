import os
from typing import Dict, List, Any, Optional
import logging
import traceback
import sys
import json
from fastmcp import FastMCP
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("image_server")

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
                "quality": "medium"  # Options: low, medium, high, auto
            }
            
            # Add transparent background if requested
            if transparent_background:
                params["background"] = "transparent"
                params["output_format"] = "png"  # Transparency requires PNG or WebP
            
            # Make the API call using the new client.images.generate method
            response = self.client.images.generate(**params)
            
            result_data = {
                "success": True,
                "data": response.data[0].url,
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
        return result
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Unexpected error in generate_image tool: {str(e)}")
        logger.error(f"Traceback: {error_traceback}")
        return {
            "success": False,
            "error": f"Tool error: {str(e)}"
        }

if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
