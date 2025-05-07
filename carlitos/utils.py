import logging
import json
from pathlib import Path

from mcp import StdioServerParameters

log = logging.getLogger("carlitos.utils")

def load_server_configs(config_path):
    """
    Load server configurations from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        List of server configuration dictionaries
    """
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

def standardize_server_configs(servers_raw):
    """
    Standardize server configurations.
    
    Args:
        servers_raw: List of raw server configuration dictionaries
        
    Returns:
        List of standardized server configuration dictionaries
    """
    standardized = []
    for server_dict in servers_raw:
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
        
        standardized.append({
            "name": name,
            "transport": transport,
            "command": server_dict.get("command"),
            "args": server_dict.get("args", []),
            "url": server_dict.get("url"),
            "env": server_dict.get("env", {}),
            "description": server_dict.get("description", "")
        })
    
    return standardized

def get_server_params(server_config):
    """
    Convert server config to MCP server parameters.
    
    Args:
        server_config: Server configuration dictionary or object
        
    Returns:
        MCP server parameters
    """
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
        # Handle ServerConfig-like object
        if hasattr(server_config, 'transport'):
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
        else:
            log.warning(f"Invalid server config type: {type(server_config)}")
            return None 