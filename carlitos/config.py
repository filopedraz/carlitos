import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from carlitos.utils import load_server_configs, standardize_server_configs

load_dotenv()

# Configure logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.ERROR,  # Default to ERROR level, will be overridden by DEBUG flag if set
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("carlitos.config")

# Basic configuration
# ------------------------------------------------------------------------------
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")

# Default paths
# ------------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "./.cursor/mcp.json"
BASE_DIR = Path(__file__).resolve().parent.parent

# LLM configuration
# ------------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")  # Only supporting Gemini currently
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-04-17")
LLM_API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "GEMINI_API_KEY")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Langfuse configuration (for Pydantic AI integration)
# ------------------------------------------------------------------------------
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")
LANGFUSE_SERVICE_NAME = os.environ.get("LANGFUSE_SERVICE_NAME", "Carlitos")

# Load and standardize server configurations
# ------------------------------------------------------------------------------
SERVERS = standardize_server_configs(load_server_configs(DEFAULT_CONFIG_PATH))

# Create default LLM config
# ------------------------------------------------------------------------------
DEFAULT_LLM_CONFIG = {
    "provider": LLM_PROVIDER,
    "model": LLM_MODEL,
    "api_key_env": LLM_API_KEY_ENV,
    "temperature": LLM_TEMPERATURE,
    "langfuse_public_key": LANGFUSE_PUBLIC_KEY,
    "langfuse_secret_key": LANGFUSE_SECRET_KEY,
    "langfuse_host": LANGFUSE_HOST,
    "langfuse_service_name": LANGFUSE_SERVICE_NAME
}

# Create default Carlitos config
# ------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "servers": SERVERS,
    "llm": DEFAULT_LLM_CONFIG
} 