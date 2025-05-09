import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from carlitos.agent import Agent
from carlitos.mcp.types import Tool


@pytest.fixture
def mock_server():
    server = AsyncMock()
    server.get_tools = AsyncMock(
        return_value=[
            Tool(
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
def agent(mock_server):
    agent = Agent(server=mock_server, name="test_agent")
    return agent


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test that agent initializes with proper attributes."""
    agent = Agent(server=AsyncMock(), name="test_agent")

    assert agent.name == "test_agent"
    assert agent.tools == {}
    assert isinstance(agent.logger, logging.Logger)


@pytest.mark.asyncio
async def test_discover_tools(agent, mock_server):
    """Test that agent can discover tools from server."""
    await agent.discover_tools()

    mock_server.get_tools.assert_called_once()
    assert "test_tool" in agent.tools
    assert agent.tools["test_tool"].name == "test_tool"


@pytest.mark.asyncio
async def test_execute_tool_success(agent, mock_server):
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
async def test_execute_tool_not_found(agent):
    """Test execution of unknown tool."""
    with pytest.raises(ValueError, match="Tool 'unknown_tool' not found"):
        await agent.execute_tool("unknown_tool", {"param": "test_value"})


@pytest.mark.asyncio
async def test_execute_tool_server_error(agent, mock_server):
    """Test handling of server error during tool execution."""
    # Configure mock to raise exception
    mock_server.execute_tool = AsyncMock(side_effect=Exception("Server error"))

    # Discover tools first
    await agent.discover_tools()

    # Execute tool and expect error handling
    with pytest.raises(Exception, match="Error executing tool 'test_tool'"):
        await agent.execute_tool("test_tool", {"param": "test_value"})


@pytest.mark.asyncio
async def test_validate_parameters_success(agent, mock_server):
    """Test successful parameter validation."""
    # Discover tools first
    await agent.discover_tools()

    # Valid parameters should not raise exceptions
    await agent.validate_parameters("test_tool", {"param": "test_value"})


@pytest.mark.asyncio
async def test_validate_parameters_missing_required(agent, mock_server):
    """Test validation with missing required parameters."""
    # Discover tools first
    await agent.discover_tools()

    # Missing required parameter should raise ValueError
    with pytest.raises(ValueError, match="Missing required parameter"):
        await agent.validate_parameters("test_tool", {})


@pytest.mark.asyncio
async def test_agent_handle_request(agent, mock_server):
    """Test agent handling a complete request flow."""
    # Setup mocks
    agent.discover_tools = AsyncMock()
    agent.execute_tool = AsyncMock(return_value={"result": "success"})

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
