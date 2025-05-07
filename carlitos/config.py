import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from dotenv import load_dotenv
from mcp import StdioServerParameters
from rich.logging import RichHandler

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("carlitos.config")

# Basic configuration
DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
VERBOSE = os.environ.get("VERBOSE", "False").lower() in ("true", "1", "t")

# Default paths
DEFAULT_CONFIG_PATH = "./.cursor/mcp.json"
BASE_DIR = Path(__file__).resolve().parent.parent

# LLM configuration
LLM_PROVIDER = "gemini"  # Only supporting Gemini currently
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-04-17")
LLM_API_KEY_ENV = os.environ.get("LLM_API_KEY_ENV", "GEMINI_API_KEY")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Servers will be loaded dynamically from config file
SERVERS = []

# Compatibility classes for backward compatibility
@dataclass
class ServerConfig:
    """MCP server configuration with backward compatibility."""
    name: str
    transport: str = "stdio"  # "stdio" or "http"
    command: Optional[str] = None
    args: List[str] = None
    url: Optional[str] = None
    env: Dict[str, str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}

@dataclass
class LLMConfig:
    """LLM configuration with backward compatibility."""
    provider: str = "gemini"
    model: str = "gemini-2.5-flash-preview-04-17"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2

@dataclass
class CarlitosConfig:
    """Main configuration for Carlitos with backward compatibility."""
    servers: List[ServerConfig]
    llm: LLMConfig

def load_servers_from_config(config_path=DEFAULT_CONFIG_PATH):
    """Load server configurations from a JSON file."""
    import json
    
    path = Path(config_path)
    if not path.exists():
        log.warning(f"Config file not found: {config_path}")
        return []
    
    try:
        with open(path, "r") as f:
            config = json.load(f)
        
        servers = []
        
        # Support MCP style config format
        if "mcpServers" in config:
            for name, server_data in config["mcpServers"].items():
                server_data["name"] = name
                servers.append(server_data)
        # Support original format
        elif "servers" in config:
            servers = config["servers"]
        else:
            log.warning("Invalid config: missing 'mcpServers' or 'servers' key")
            return []
        
        log.debug(f"Loaded {len(servers)} servers from config")
        return servers
        
    except (json.JSONDecodeError, FileNotFoundError) as e:
        log.warning(f"Error loading config: {e}")
        return []

def get_server_params(server_config):
    """Convert server config to MCP server parameters."""
    if isinstance(server_config, dict):
        name = server_config.get("name", "unknown")
        transport = server_config.get("transport")
        
        # Determine transport type if not specified
        if not transport:
            if "command" in server_config:
                transport = "stdio"
            elif "url" in server_config:
                transport = "http"
            else:
                log.warning(f"Server {name} has no transport specified and no command/URL")
                return None
        
        # Return the appropriate parameters based on transport type
        if transport == "stdio":
            command = server_config.get("command")
            if not command:
                log.warning(f"Server {name} has no command specified")
                return None
                
            return StdioServerParameters(
                command=command,
                args=server_config.get("args", []),
                env=server_config.get("env", {})
            )
        elif transport == "http":
            url = server_config.get("url")
            if not url:
                log.warning(f"Server {name} has no URL specified")
                return None
                
            return url
        else:
            log.warning(f"Unsupported transport: {transport}")
            return None
    else:
        # Handle ServerConfig object
        if server_config.transport == "stdio":
            if not server_config.command:
                log.warning(f"Server {server_config.name} has no command specified")
                return None
            return StdioServerParameters(
                command=server_config.command,
                args=server_config.args,
                env=server_config.env
            )
        elif server_config.transport == "http":
            if not server_config.url:
                log.warning(f"Server {server_config.name} has no URL specified")
                return None
            return server_config.url
        else:
            log.warning(f"Unsupported transport: {server_config.transport}")
            return None

# Simple function to get LLM config as a dictionary
def get_llm_config():
    """Get LLM configuration as a dictionary."""
    return {
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "api_key_env": LLM_API_KEY_ENV,
        "temperature": LLM_TEMPERATURE
    }

# Compatibility function to load config
def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> CarlitosConfig:
    """
    Load configuration from JSON file with backward compatibility.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Parsed configuration in CarlitosConfig format
    """
    server_dicts = load_servers_from_config(config_path)
    
    # Convert dictionaries to ServerConfig objects
    server_configs = []
    for server_dict in server_dicts:
        name = server_dict.get("name", "unknown")
        transport = server_dict.get("transport")
        
        # Determine transport if not specified
        if not transport:
            if "command" in server_dict:
                transport = "stdio"
            elif "url" in server_dict:
                transport = "http"
            else:
                log.warning(f"Skipping server {name} - no command or URL specified")
                continue
        
        server_configs.append(ServerConfig(
            name=name,
            transport=transport,
            command=server_dict.get("command"),
            args=server_dict.get("args", []),
            url=server_dict.get("url"),
            env=server_dict.get("env", {}),
            description=server_dict.get("description", "")
        ))
    
    # Create LLMConfig from environment or defaults
    llm_config = LLMConfig(
        provider=LLM_PROVIDER,
        model=LLM_MODEL,
        api_key_env=LLM_API_KEY_ENV,
        temperature=LLM_TEMPERATURE
    )
    
    return CarlitosConfig(
        servers=server_configs,
        llm=llm_config
    )

# Load servers on module import
SERVERS = load_servers_from_config() 