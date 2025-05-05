import os
from typing import Dict, Any, Optional, List
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image

class OpenAIImageGenServer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
        self.client = OpenAI(api_key=api_key)

    def generate_image_4o(self, prompt: str, size: str = "1024x1024", n: int = 1, 
                         transparent_background: bool = False, 
                         referenced_image_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate an image using OpenAI's GPT-4o model (when available)"""
        try:
            # Note: This is a placeholder for the GPT-4o API
            # The actual implementation will be updated when the API is released
            raise NotImplementedError("GPT-4o image generation API is not yet available")
            
            # Expected future implementation:
            # response = self.client.images.generate(
            #     model="gpt-4o",
            #     prompt=prompt,
            #     size=size,
            #     n=n,
            #     transparent_background=transparent_background,
            #     referenced_image_ids=referenced_image_ids
            # )
            # return {
            #     "success": True,
            #     "data": response.data[0].url if response.data else None,
            #     "model": "gpt-4o"
            # }
        except Exception as e:
            # Fallback to DALL-E 3
            return self.generate_image_dalle(prompt, size, n)

    def generate_image_dalle(self, prompt: str, size: str = "1024x1024", n: int = 1) -> Dict[str, Any]:
        """Generate an image using DALL-E 3 as fallback"""
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                n=n
            )
            return {
                "success": True,
                "data": response.data[0].url if response.data else None,
                "model": "dall-e-3"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": "dall-e-3"
            }

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
                        response = self.client.images.edit(
                            image=image_file,
                            mask=mask_file,
                            prompt=prompt
                        )
                else:
                    response = self.client.images.edit(
                        image=image_file,
                        prompt=prompt
                    )

            # Cleanup
            os.remove(image_path)
            if mask_path:
                os.remove(mask_path)

            return {
                "success": True,
                "data": response.data[0].url if response.data else None
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
