import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from mcp import ClientSession, Tool
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from rich.progress import Progress

from carlitos.llm import LLMCoreAgent
from carlitos.utils import get_server_params

log = logging.getLogger("carlitos.agent")


class AgenticMCPAgent:
    """
    An agentic MCP agent that supports a deliberate approach
    to tool selection and execution.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize agentic agent with config.

        Args:
            config: Agent configuration dictionary
        """
        self.config = config
        self.llm = LLMCoreAgent(config["llm"])
        # Create a dictionary of server name -> server config for easier access
        self.servers = {server["name"]: server for server in config["servers"]}
        self._last_tool_results = None  # Store last tool results for debugging
        self._server_tools_map = {}  # Will be populated during tool discovery
        self.all_tools = None  # Cache the tools to avoid rediscovering on each message
        log.debug(f"Initialized agent with {len(self.servers)} servers")

    async def chat(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process a message in a chat session.

        Args:
            message: User message
            chat_history: Optional chat history for context

        Returns:
            Agent response
        """
        log.info(f"Processing chat message: {message}")

        # Discover tools if we haven't done it yet
        if not self.all_tools:
            self.all_tools = await self._discover_tools()
            if not self.all_tools:
                return "No tools available. Please check your MCP server configuration."

        # Get agentic response with thinking and tool selection
        try:
            result = await self._process_chat_message(message, chat_history)
            return result
        except Exception as e:
            log.error(f"Error in chat processing: {e}", exc_info=True)
            return f"I encountered an error while processing your request: {str(e)} - I cannot provide the requested information at this time."

    async def run(
        self, query: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Run the agent with a one-shot query.

        Args:
            query: User query
            chat_history: Optional chat history for context

        Returns:
            Agent response
        """
        # Simply delegate to the chat method
        return await self.chat(query, chat_history)

    async def _discover_tools(self, progress: Optional[Progress] = None) -> List[Tool]:
        """
        Connect to all servers and get available tools.

        Args:
            progress: Optional progress bar

        Returns:
            List of all available tools
        """
        all_tools = []
        server_tools_map = {}

        for server_name, server_config in self.servers.items():
            if progress:
                progress.console.print(f"Connecting to server: {server_name}")

            try:
                tools = await self._get_server_tools(server_config)
                all_tools.extend(tools)
                server_tools_map[server_name] = [tool.name for tool in tools]

                if progress:
                    progress.console.print(
                        f"Found {len(tools)} tools on server {server_name}"
                    )

                log.debug(f"Discovered {len(tools)} tools from server {server_name}")
            except Exception as e:
                log.error(f"Error discovering tools from server {server_name}: {e}")
                if progress:
                    progress.console.print(
                        f"[red]Error connecting to server {server_name}: {e}[/red]"
                    )

        # Store the server-tool mapping for later use
        self._server_tools_map = server_tools_map

        return all_tools

    async def _execute_with_session(
        self,
        server_config: Dict[str, Any],
        session_function: Callable[[ClientSession], Awaitable[Any]],
    ) -> Any:
        """
        Common helper method to execute operations with a session.

        Args:
            server_config: Server configuration
            session_function: Function to execute with the session

        Returns:
            Result of the session function
        """
        server_params = get_server_params(server_config)

        if server_config["transport"] == "stdio":
            client_factory = stdio_client
        elif server_config["transport"] == "http":
            client_factory = sse_client
            server_params = (
                server_params  # For HTTP servers, params is just the URL string
            )
        else:
            raise ValueError(f"Unsupported transport: {server_config['transport']}")

        async with client_factory(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session_function(session)

    async def _get_server_tools(self, server_config: Dict[str, Any]) -> List[Tool]:
        """
        Get tools from a single server.

        Args:
            server_config: Server configuration

        Returns:
            List of tools from the server
        """
        try:
            # Define the session function to list tools
            async def list_tools_function(session):
                tools_response = await session.list_tools()
                log.debug(
                    f"Retrieved tools from {server_config['name']}: {[tool.name for tool in tools_response.tools]}"
                )
                return tools_response.tools

            return await self._execute_with_session(server_config, list_tools_function)
        except Exception as e:
            log.error(f"Error connecting to server {server_config['name']}: {e}")
            raise

    def _find_tool_and_server(
        self, tool_name: str, all_tools: List[Tool]
    ) -> Tuple[Optional[Tool], Optional[str]]:
        """
        Find a tool by name and its server.

        Args:
            tool_name: Name of the tool
            all_tools: List of all available tools

        Returns:
            Tuple of (tool, server_name) or (None, None) if not found
        """
        # Find the tool object
        tool = next((t for t in all_tools if t.name == tool_name), None)

        if not tool:
            return None, None

        # Find which server the tool belongs to
        server_name = None
        for s_name, tool_names in self._server_tools_map.items():
            if tool_name in tool_names:
                server_name = s_name
                break

        return tool, server_name

    async def _execute_tool(
        self, server_name: str, tool_name: str, parameters: Dict[str, Any]
    ) -> str:
        """
        Execute a tool on a server.

        Args:
            server_name: Name of the server
            tool_name: Name of the tool
            parameters: Tool parameters

        Returns:
            Result of tool execution
        """
        server_config = self.servers[server_name]

        try:
            # Define the session function to call the tool
            async def call_tool_function(session):
                log.info(
                    f"Executing tool {tool_name} on server {server_name} with parameters: {parameters}"
                )

                # Call the tool and log the exact moment
                log.debug(f"About to call session.call_tool for {tool_name}")
                result = await session.call_tool(tool_name, arguments=parameters)
                log.debug(f"Received response from session.call_tool for {tool_name}")

                # Log detailed information about the raw result
                if hasattr(result, "content"):
                    log.info(
                        f"Tool {tool_name} returned content of length: {len(result.content)}"
                    )
                    # Log the first few items in content
                    for i, content_item in enumerate(result.content[:3]):
                        log.info(
                            f"Content item {i} type: {type(content_item).__name__}"
                        )
                        if hasattr(content_item, "type"):
                            log.info(
                                f"Content item {i} content-type: {content_item.type}"
                            )
                        if hasattr(content_item, "text"):
                            log.info(
                                f"Content item {i} text (first 100 chars): {content_item.text[:100]}"
                            )
                else:
                    log.info(
                        f"Tool {tool_name} returned result of type {type(result).__name__} without content attribute"
                    )
                    log.info(f"Raw result string representation: {str(result)[:200]}")

                # Store raw result for debugging
                self._store_raw_result(tool_name, parameters, result)

                # Process the result based on its type
                if hasattr(result, "content") and result.content:
                    formatted_result = self._format_result(result)
                    # Log the actual content of the result for debugging
                    log.debug(
                        f"Raw tool execution result type: {type(result).__name__}"
                    )
                    log.debug(
                        f"Formatted tool execution result: {formatted_result[:200]}..."
                    )
                    return formatted_result
                else:
                    log.debug("Tool executed successfully but returned no content.")
                    return "Tool executed successfully but returned no content or empty data. Please note that no data was found for your query."

            return await self._execute_with_session(server_config, call_tool_function)
        except Exception as e:
            log.error(
                f"Error executing tool {tool_name} on server {server_name}: {e}",
                exc_info=True,
            )
            error_msg = f"Error executing tool: {str(e)} - Please note that no data was returned. Do not fabricate or invent any information."
            # Store error for debugging
            self._store_error_result(tool_name, parameters, str(e))
            return error_msg

    def _store_raw_result(
        self, tool_name: str, parameters: Dict[str, Any], result: Any
    ):
        """
        Store raw tool result for debugging purposes.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            result: Raw tool result
        """
        try:
            # Attempt to extract useful content from the result
            formatted_result = self._format_result(result)

            # Create a debug-friendly representation
            debug_info = {
                "tool": tool_name,
                "parameters": parameters,
                "raw_result_type": str(type(result).__name__),
                "formatted_result": formatted_result,
            }

            # Store additional content details if available
            if hasattr(result, "content") and result.content:
                content_details = []
                for i, content in enumerate(result.content):
                    content_info = {
                        "index": i,
                        "type": getattr(content, "type", str(type(content).__name__)),
                        "text": getattr(
                            content, "text", str(content)[:500]
                        ),  # Truncate long texts
                    }
                    content_details.append(content_info)
                debug_info["content_details"] = content_details

            # Store the debug info
            self._last_tool_results = json.dumps(debug_info, indent=2)
            log.debug(
                f"Stored raw result for {tool_name}: {self._last_tool_results[:200]}..."
            )
        except Exception as e:
            # If we fail to store the raw result in a structured way, use string representation
            error_msg = f"Error storing raw result for {tool_name}: {str(e)}\nParameters: {parameters}\nResult: {str(result)[:200]}..."
            log.error(error_msg)
            self._last_tool_results = error_msg

    def _store_error_result(
        self, tool_name: str, parameters: Dict[str, Any], error: str
    ):
        """
        Store error information for debugging purposes.

        Args:
            tool_name: Name of the tool
            parameters: Tool parameters
            error: Error message
        """
        # Create a debug-friendly representation
        debug_info = {"tool": tool_name, "parameters": parameters, "error": error}

        # Store the debug info
        self._last_tool_results = json.dumps(debug_info, indent=2)
        log.debug(
            f"Stored error result for {tool_name}: {self._last_tool_results[:200]}..."
        )

    def _format_result(self, result):
        """
        Format tool call result for display.

        Args:
            result: Tool call result

        Returns:
            Formatted result string
        """
        formatted_parts = []

        if hasattr(result, "content"):
            for content in result.content:
                if hasattr(content, "type") and hasattr(content, "text"):
                    if content.type == "text/plain":
                        formatted_parts.append(content.text)
                        # Log the actual text content
                        log.debug(f"Text content: {content.text[:200]}...")
                    elif content.type == "text/html":
                        formatted_parts.append(f"HTML content: {content.text}")
                        log.debug(f"HTML content: {content.text[:200]}...")
                    elif content.type.startswith("image/"):
                        formatted_parts.append(f"[Image of type {content.type}]")
                    else:
                        # For other content types, attempt to log the content in full
                        formatted_parts.append(
                            f"Content of type {content.type}: {getattr(content, 'text', str(content))}"
                        )
                        log.debug(
                            f"Other content type {content.type}: {getattr(content, 'text', str(content)[:200])}..."
                        )
                else:
                    # Try to convert non-standard content to string for logging
                    content_str = str(content)
                    formatted_parts.append(content_str)
                    log.debug(f"Non-standard content: {content_str[:200]}...")
        else:
            # If result has no content attribute, try to convert it to string
            result_str = str(result)
            formatted_parts.append(result_str)
            log.debug(f"Result without content: {result_str[:200]}...")

        # If the result is empty or just contains content type information,
        # make it explicit that no real data was returned
        if not formatted_parts or all(
            "Content of type" in part for part in formatted_parts
        ):
            formatted_parts.append(
                "No data returned. The tool did not provide any concrete information."
            )

        return "\n".join(formatted_parts)

    async def _process_chat_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Process a chat message using the agentic approach.

        Args:
            message: User message
            chat_history: Optional chat history for context

        Returns:
            Agent response
        """
        # Get server descriptions for context if available
        detailed_server_info = {
            server_name: server["description"]
            for server_name, server in self.servers.items()
            if "description" in server and server["description"]
        }

        # First, get the agent's thinking about what tools might be needed
        thinking, needed_tools = await self.llm.analyze_task(
            message, self.all_tools, chat_history, detailed_server_info
        )
        log.debug(f"Agent thinking: {thinking}")
        log.info(
            f"Agent selected {len(needed_tools)} tools to use: {[t['name'] for t in needed_tools]}"
        )

        # If the agent thinks no tools are needed, just return the thinking
        if not needed_tools:
            return thinking

        # If tools are needed, process them one by one
        results = []
        tool_execution_details = []  # Store detailed info about what was executed

        for tool_info in needed_tools:
            tool_name = tool_info["name"]
            parameters = tool_info.get("parameters", {})
            purpose = tool_info.get("purpose", "No purpose specified")

            log.info(f"Processing tool: {tool_name} for purpose: {purpose}")
            tool_execution_details.append(
                f"- Used {tool_name} with parameters: {json.dumps(parameters)}"
            )

            # Find the tool and its server
            tool, server_name = self._find_tool_and_server(tool_name, self.all_tools)
            if not tool:
                error_msg = f"Error: Tool '{tool_name}' not found in available tools."
                log.error(error_msg)
                results.append(error_msg)
                continue

            # Execute the tool
            log.info(f"Executing tool: {tool_name} on server {server_name}")
            result = await self._execute_tool(server_name, tool_name, parameters)
            tool_result = f"Result from {tool_name}: {result}"
            log.info(f"Tool result summary for {tool_name}: {result[:200]}...")
            results.append(tool_result)

        # Summarize the results with the LLM
        if results:
            tool_results = "\n\n".join(results)
            log.debug(f"All tool results: {tool_results[:500]}...")

            # Add information about which tools were executed and with what parameters
            execution_summary = "\n".join(tool_execution_details)
            tool_results = (
                f"Tool execution summary:\n{execution_summary}\n\n{tool_results}"
            )

            # Store combined results for debugging
            self._last_tool_results = (
                f"EXECUTION SUMMARY:\n{execution_summary}\n\nRESULTS:\n{tool_results}"
            )

            log.info("Synthesizing final response from tool results")
            final_response = await self.llm.synthesize_results(
                message, thinking, tool_results, chat_history
            )
            return final_response
        else:
            return thinking
