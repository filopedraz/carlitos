import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from mcp import StdioServerParameters
from pydantic import BaseModel, Field, model_validator

load_dotenv()

log = logging.getLogger("carlitos.config")


class ServerConfig(BaseModel):
    """MCP server configuration."""
    name: str
    transport: str = "stdio"  # "stdio" or "http"
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)
    description: str = ""
    
    @model_validator(mode='after')
    def validate_server_config(self):
        """Validate server configuration based on transport type."""
        if self.transport == "stdio" and not self.command:
            raise ValueError("Command must be provided for stdio transport")
        elif self.transport == "http" and not self.url:
            raise ValueError("URL must be provided for http transport")
        return self


class LLMConfig(BaseModel):
    """LLM configuration for tool selection."""
    provider: str = "gemini"  # Only supporting Gemini with PydanticAI
    model: str = "gemini-2.5-flash-preview-04-17"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.2
    
    @model_validator(mode='before')
    def validate_provider(cls, data):
        if isinstance(data, dict) and "provider" in data:
            provider = data["provider"]
            if provider != "gemini":
                raise ValueError(f"Provider {provider} not supported. Only 'gemini' is supported.")
        return data


class CarlitosConfig(BaseModel):
    """Main configuration for Carlitos."""
    servers: List[ServerConfig]
    llm: LLMConfig


def get_llm_config_from_env() -> LLMConfig:
    """
    Get LLM configuration from environment variables with defaults.
    
    Returns:
        LLM configuration
    """
    return LLMConfig(
        provider="gemini",  # Only supporting Gemini
        model=os.environ.get("LLM_MODEL", "gemini-2.5-flash-preview-04-17"),
        api_key_env=os.environ.get("LLM_API_KEY_ENV", "GEMINI_API_KEY"),
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    )


def load_config(config_path: str = "./.cursor/mcp.json") -> CarlitosConfig:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Parsed configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(path, "r") as f:
            raw_config = json.load(f)
        
        # Get LLM config from environment or use defaults
        llm_config = get_llm_config_from_env()
        
        # Convert MCP style JSON to our expected format
        if "mcpServers" in raw_config:
            servers = []
            for name, server_data in raw_config["mcpServers"].items():
                # Add name to server data
                server_data["name"] = name
                
                # Determine transport type based on data
                if "command" in server_data:
                    server_data["transport"] = "stdio"
                elif "url" in server_data:
                    server_data["transport"] = "http"
                else:
                    log.warning(f"Skipping server {name} - no command or URL specified")
                    continue
                
                try:
                    servers.append(ServerConfig(**server_data))
                    log.debug(f"Added server {name} with {server_data['transport']} transport")
                except Exception as e:
                    log.warning(f"Invalid server configuration for {name}: {e}")
            
            config = CarlitosConfig(
                servers=servers,
                llm=llm_config
            )
        # Support our original format too
        elif "servers" in raw_config:
            servers = []
            for server_data in raw_config["servers"]:
                name = server_data.get("name", "unknown")
                
                # Determine transport type based on data
                if "command" in server_data:
                    server_data["transport"] = "stdio"
                elif "url" in server_data:
                    server_data["transport"] = "http"
                else:
                    log.warning(f"Skipping server {name} - no command or URL specified")
                    continue
                
                try:
                    servers.append(ServerConfig(**server_data))
                    log.debug(f"Added server {name} with {server_data['transport']} transport")
                except Exception as e:
                    log.warning(f"Invalid server configuration for {name}: {e}")
            
            # If llm config exists in file, use it, otherwise use env variables
            if "llm" in raw_config:
                file_llm_config = raw_config["llm"]
                # Ensure provider is gemini
                file_llm_config["provider"] = "gemini"
                llm_config = LLMConfig(**file_llm_config)
            
            config = CarlitosConfig(
                servers=servers,
                llm=llm_config
            )
        else:
            raise ValueError("Invalid config: missing 'mcpServers' or 'servers' key")
            
        log.debug(f"Loaded config with {len(config.servers)} servers")
        return config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except Exception as e:
        raise ValueError(f"Invalid config: {e}")


def get_server_params(server_config: ServerConfig) -> Union[StdioServerParameters, str]:
    """
    Convert server config to MCP server parameters.
    
    Args:
        server_config: Server configuration
        
    Returns:
        MCP server parameters or URL for HTTP transport
    """
    if server_config.transport == "stdio":
        if not server_config.command:
            raise ValueError(f"Server {server_config.name} has no command specified")
        return StdioServerParameters(
            command=server_config.command,
            args=server_config.args,
            env=server_config.env
        )
    elif server_config.transport == "http":
        if not server_config.url:
            raise ValueError(f"Server {server_config.name} has no URL specified")
        return server_config.url
    else:
        raise ValueError(f"Unsupported transport: {server_config.transport}") 