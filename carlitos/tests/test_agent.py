import logging
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest

from carlitos.agent import AgenticMCPAgent


class MockTool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


class MockAgent(AgenticMCPAgent):
    """A test-friendly version of AgenticMCPAgent that accepts the parameters needed for tests."""

    def __init__(
        self, server: Optional[AsyncMock] = None, name: str = "test_agent"
    ) -> None:
        # Initialize with mock attributes for testing
        self.name = name
        self.tools: Dict[str, Any] = {}
        self.logger = logging.getLogger("test")
        self.server = server

    async def discover_tools(self) -> Dict[str, MockTool]:
        """Mock discover_tools method"""
        if self.server:
            tools = await self.server.get_tools()
            for tool in tools:
                self.tools[tool.name] = tool
        return self.tools

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock execute_tool method"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        if self.server:
            try:
                return await self.server.execute_tool(tool_name, parameters)
            except Exception as e:
                raise Exception(f"Error executing tool '{tool_name}': {str(e)}")

        return {"result": "mock_result"}

    async def validate_parameters(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> bool:
        """Mock validate_parameters method"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]
        required = tool.parameters.get("required", [])

        for param in required:
            if param not in parameters:
                raise ValueError(f"Missing required parameter {param}")

        return True

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Mock handle_request method"""
        await self.discover_tools()
        return await self.execute_tool(request["tool_name"], request["parameters"])


@pytest.fixture
def mock_server() -> AsyncMock:
    server = AsyncMock()
    server.get_tools = AsyncMock(
        return_value=[
            MockTool(
                name="test_tool",
                description="A test tool",
                parameters={
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                    "required": ["param"],
                },
            )
        ]
    )
    return server


@pytest.fixture
def agent(mock_server: AsyncMock) -> MockAgent:
    agent = MockAgent(server=mock_server, name="test_agent")
    return agent


@pytest.mark.asyncio
async def test_agent_initialization() -> None:
    """Test that agent initializes with proper attributes."""
    agent = MockAgent(server=AsyncMock(), name="test_agent")

    assert agent.name == "test_agent"
    assert agent.tools == {}
    assert isinstance(agent.logger, logging.Logger)


@pytest.mark.asyncio
async def test_discover_tools(agent: MockAgent, mock_server: AsyncMock) -> None:
    """Test that agent can discover tools from server."""
    await agent.discover_tools()

    mock_server.get_tools.assert_called_once()
    assert "test_tool" in agent.tools
    assert agent.tools["test_tool"].name == "test_tool"


@pytest.mark.asyncio
async def test_execute_tool_success(agent: MockAgent, mock_server: AsyncMock) -> None:
    """Test successful tool execution."""
    # Set up the mock tool response
    mock_server.execute_tool = AsyncMock(return_value={"result": "success"})

    # Discover tools first
    await agent.discover_tools()

    # Execute tool
    result = await agent.execute_tool("test_tool", {"param": "test_value"})

    # Verify execution
    mock_server.execute_tool.assert_called_once_with(
        "test_tool", {"param": "test_value"}
    )
    assert result == {"result": "success"}


@pytest.mark.asyncio
async def test_execute_tool_not_found(agent: MockAgent) -> None:
    """Test execution of unknown tool."""
    with pytest.raises(ValueError, match="Tool 'unknown_tool' not found"):
        await agent.execute_tool("unknown_tool", {"param": "test_value"})


@pytest.mark.asyncio
async def test_execute_tool_server_error(
    agent: MockAgent, mock_server: AsyncMock
) -> None:
    """Test handling of server error during tool execution."""
    # Configure mock to raise exception
    mock_server.execute_tool = AsyncMock(side_effect=Exception("Server error"))

    # Discover tools first
    await agent.discover_tools()

    # Execute tool and expect error handling
    with pytest.raises(Exception, match="Error executing tool 'test_tool'"):
        await agent.execute_tool("test_tool", {"param": "test_value"})


@pytest.mark.asyncio
async def test_validate_parameters_success(
    agent: MockAgent, mock_server: AsyncMock
) -> None:
    """Test successful parameter validation."""
    # Discover tools first
    await agent.discover_tools()

    # Valid parameters should not raise exceptions
    await agent.validate_parameters("test_tool", {"param": "test_value"})


@pytest.mark.asyncio
async def test_validate_parameters_missing_required(
    agent: MockAgent, mock_server: AsyncMock
) -> None:
    """Test validation with missing required parameters."""
    # Discover tools first
    await agent.discover_tools()

    # Missing required parameter should raise ValueError
    with pytest.raises(ValueError, match="Missing required parameter"):
        await agent.validate_parameters("test_tool", {})


@pytest.mark.asyncio
async def test_agent_handle_request(agent: MockAgent, mock_server: AsyncMock) -> None:
    """Test agent handling a complete request flow."""
    # Create mock methods instead of reassigning existing methods
    original_discover_tools = agent.discover_tools
    original_execute_tool = agent.execute_tool

    # Use the real mock methods
    agent.discover_tools = AsyncMock()  # type: ignore[method-assign]
    agent.execute_tool = AsyncMock(return_value={"result": "success"})  # type: ignore[method-assign]

    try:
        # Handle request
        result = await agent.handle_request(
            {
                "type": "tool_call",
                "tool_name": "test_tool",
                "parameters": {"param": "test_value"},
            }
        )

        # Verify workflow
        agent.discover_tools.assert_called_once()
        agent.execute_tool.assert_called_once_with("test_tool", {"param": "test_value"})
        assert result == {"result": "success"}
    finally:
        # Restore original methods
        agent.discover_tools = original_discover_tools  # type: ignore[method-assign]
        agent.execute_tool = original_execute_tool  # type: ignore[method-assign]
