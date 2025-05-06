#!/usr/bin/env python3
import requests
import json
import time
import sys
import os

def generate_image(prompt, size="1024x1024", n=1, transparent=False, host="localhost", port=8000):
    """
    Generate an image using the API endpoint
    
    Args:
        prompt: The text description for the image
        size: Image size (1024x1024, 1024x1792, or 1792x1024)
        n: Number of images to generate
        transparent: Whether to generate image with transparent background
        host: Server hostname
        port: Server port
        
    Returns:
        Generated image URL if successful
    """
    # API endpoint URL
    api_url = f"http://{host}:{port}/api/generate"
    
    # Create the request payload
    payload = {
        "prompt": prompt,
        "size": size,
        "n": n,
        "transparent_background": transparent
    }
    
    print(f"Connecting to server at: {api_url}")
    print(f"Sending request to generate image with prompt: '{prompt}'")
    print(f"Settings: size={size}, transparent={transparent}")
    
    try:
        # Make the POST request to start generation
        response = requests.post(api_url, json=payload)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {api_url}")
        print("Make sure the server is running and accessible.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response text: {response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None
    
    # Get the request ID from the response
    request_id = response.json().get("request_id")
    
    if not request_id:
        print("Error: No request ID returned")
        return None
    
    print(f"Request ID: {request_id}")
    print("Waiting for image generation to complete...")
    
    # Poll the status endpoint until the image is ready
    status_url = f"http://localhost:8000/api/status/{request_id}"
    
    max_attempts = 60  # Maximum number of attempts (30 seconds with 0.5s delay)
    attempts = 0
    
    while attempts < max_attempts:
        status_response = requests.get(status_url)
        
        if status_response.status_code != 200:
            print(f"Error checking status: {status_response.status_code}")
            return None
        
        status_data = status_response.json()
        
        if status_data["status"] == "complete":
            print("Image generation complete!")
            result = status_data["result"]
            if result["success"]:
                image_url = result["data"]
                print(f"Generated image URL: {image_url}")
                print(f"Model used: {result.get('model', 'unknown')}")
                return image_url
            else:
                print(f"Error generating image: {result['error']}")
                return None
        elif status_data["status"] == "error":
            print(f"Error generating image: {status_data['error']}")
            return None
        
        # Wait before polling again
        time.sleep(0.5)
        attempts += 1
        
        # Show progress every 10 attempts
        if attempts % 10 == 0:
            print(f"Still waiting... ({attempts} attempts)")
    
    print("Timed out waiting for image generation")
    return None

def print_help():
    print("Usage: python test_image_gen.py [options] [prompt]")
    print("\nOptions:")
    print("  --help         Show this help message")
    print("  --transparent  Generate image with transparent background")
    print("  --size SIZE    Set image size (1024x1024, 1024x1792, or 1792x1024)")
    print("  --host HOST    Server hostname (default: localhost)")
    print("  --port PORT    Server port (default: 8000)")
    print("\nExamples:")
    print("  python test_image_gen.py \"A futuristic cityscape with flying cars\"")
    print("  python test_image_gen.py --transparent \"A logo for a tech company\"")
    print("  python test_image_gen.py --size 1024x1792 \"A tall skyscraper\"")
    print("  python test_image_gen.py --host 192.168.1.100 --port 8080 \"A mountain landscape\"")

if __name__ == "__main__":
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        sys.exit(1)
    
    # Parse command line arguments
    args = sys.argv[1:]
    transparent = False
    size = "1024x1024"
    host = "localhost"
    port = 8000
    prompt = "A futuristic cityscape with flying cars and neon lights"
    
    if "--help" in args:
        print_help()
        sys.exit(0)
    
    if "--transparent" in args:
        transparent = True
        args.remove("--transparent")
    
    if "--size" in args and len(args) > args.index("--size") + 1:
        size_index = args.index("--size")
        size = args[size_index + 1]
        args.pop(size_index)  # Remove --size
        args.pop(size_index)  # Remove the size value
    
    if "--host" in args and len(args) > args.index("--host") + 1:
        host_index = args.index("--host")
        host = args[host_index + 1]
        args.pop(host_index)  # Remove --host
        args.pop(host_index)  # Remove the host value
    
    if "--port" in args and len(args) > args.index("--port") + 1:
        port_index = args.index("--port")
        port = int(args[port_index + 1])
        args.pop(port_index)  # Remove --port
        args.pop(port_index)  # Remove the port value
    
    # Any remaining arguments are treated as the prompt
    if args:
        prompt = " ".join(args)
    
    # Generate the image
    result = generate_image(prompt, size=size, transparent=transparent, host=host, port=port)
    
    if result:
        print("Image generation successful!")
    else:
        print("Image generation failed.") 