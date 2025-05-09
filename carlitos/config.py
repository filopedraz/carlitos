import base64
import logging
import os
from pathlib import Path

import logfire
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
    handlers=[RichHandler(rich_tracebacks=True)],
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
LLM_PROVIDER = os.environ.get(
    "LLM_PROVIDER", "gemini"
)  # Only supporting Gemini currently
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-04-17")
LLM_API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "GEMINI_API_KEY")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Routing configuration
# ------------------------------------------------------------------------------
ROUTING_MODEL = os.environ.get("ROUTING_MODEL", "gemini-2.0-flash")
ROUTING_TEMPERATURE = float(os.environ.get("ROUTING_TEMPERATURE", "0.1"))

# Instrumentation configuration
# ------------------------------------------------------------------------------
LOGFIRE_ENABLED = os.environ.get("LOGFIRE_ENABLED", "False").lower() in (
    "true",
    "1",
    "t",
)

# Langfuse configuration
# ------------------------------------------------------------------------------
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")
LANGFUSE_SERVICE_NAME = os.environ.get("LANGFUSE_SERVICE_NAME", "Carlitos")

# Initialize Langfuse and Logfire once at startup
# ------------------------------------------------------------------------------
# Configure logfire
logfire.configure(
    service_name=LANGFUSE_SERVICE_NAME,
    send_to_logfire=LOGFIRE_ENABLED,  # Only send to Logfire if explicitly enabled
)
logger.info(
    f"Configured Logfire with service name: {LANGFUSE_SERVICE_NAME} (send_to_logfire={LOGFIRE_ENABLED})"
)

# Configure Langfuse if credentials are available
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY and LANGFUSE_HOST:
    logger.info(
        "Configuring Langfuse integration via OpenTelemetry (once for the application)"
    )
    langfuse_auth = base64.b64encode(
        f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
        f"{LANGFUSE_HOST.rstrip('/')}/api/public/otel"
    )
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
    logger.info(f"Langfuse service name set to: {LANGFUSE_SERVICE_NAME}")
else:
    logger.info(
        "Langfuse credentials not found. Telemetry will be collected but not sent to Langfuse."
    )

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
}

# Create routing LLM config
# ------------------------------------------------------------------------------
ROUTING_LLM_CONFIG = {
    "provider": LLM_PROVIDER,
    "model": ROUTING_MODEL,
    "api_key_env": LLM_API_KEY_ENV,
    "temperature": ROUTING_TEMPERATURE,
}

# Create default Carlitos config
# ------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "servers": SERVERS,
    "llm": DEFAULT_LLM_CONFIG,
    "routing": ROUTING_LLM_CONFIG,
}
