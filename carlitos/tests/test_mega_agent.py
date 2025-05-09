import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from carlitos.agent import Agent
from carlitos.mcp.types import Tool
from carlitos.mega_agent import MegaAgent


@pytest.fixture
def mock_specialized_agents():
    agents = {
        "calendar": AsyncMock(spec=Agent),
        "weather": AsyncMock(spec=Agent),
        "search": AsyncMock(spec=Agent),
    }

    # Configure mock agents
    for name, agent in agents.items():
        agent.name = name
        agent.handle_request = AsyncMock(return_value={"result": f"{name} result"})

    return agents


@pytest.fixture
def mega_agent(mock_specialized_agents):
    # Create MegaAgent with mock specialized agents
    agent = MegaAgent(agents=list(mock_specialized_agents.values()))

    # Mock the routing function
    agent.router = AsyncMock()

    return agent


@pytest.mark.asyncio
async def test_mega_agent_initialization():
    """Test that mega agent initializes with proper attributes."""
    mock_agents = [
        AsyncMock(spec=Agent, name="agent1"),
        AsyncMock(spec=Agent, name="agent2"),
    ]

    mega_agent = MegaAgent(agents=mock_agents)

    assert len(mega_agent.agents) == 2
    assert mega_agent.agents[0].name == "agent1"
    assert mega_agent.agents[1].name == "agent2"
    assert mega_agent.conversation_history == []


@pytest.mark.asyncio
async def test_route_to_agent(mega_agent, mock_specialized_agents):
    """Test routing a request to the appropriate specialized agent."""
    # Configure router to return a specific agent
    mega_agent.router.route_request = AsyncMock(
        return_value=mock_specialized_agents["calendar"]
    )

    request = {"query": "What's on my calendar today?"}
    result = await mega_agent.handle_request(request)

    # Verify router was called
    mega_agent.router.route_request.assert_called_once_with(request, mega_agent.agents)

    # Verify the right agent was called
    mock_specialized_agents["calendar"].handle_request.assert_called_once_with(request)
    assert result == {"result": "calendar result"}


@pytest.mark.asyncio
async def test_handle_request_no_matching_agent(mega_agent):
    """Test handling a request when no matching agent is found."""
    # Configure router to return None (no matching agent)
    mega_agent.router.route_request = AsyncMock(return_value=None)

    request = {"query": "What's the meaning of life?"}

    with pytest.raises(ValueError, match="No suitable agent found"):
        await mega_agent.handle_request(request)


@pytest.mark.asyncio
async def test_handle_multiagent_request(mega_agent, mock_specialized_agents):
    """Test handling a request that requires multiple agents."""
    # Configure router to return multiple agents
    mega_agent.router.route_request = AsyncMock(
        return_value=[
            mock_specialized_agents["weather"],
            mock_specialized_agents["calendar"],
        ]
    )

    request = {"query": "Will it rain during my meeting today?"}
    result = await mega_agent.handle_request(request)

    # Verify both agents were called
    mock_specialized_agents["weather"].handle_request.assert_called_once_with(request)
    mock_specialized_agents["calendar"].handle_request.assert_called_once_with(request)

    # Verify results were combined
    assert "weather result" in str(result)
    assert "calendar result" in str(result)


@pytest.mark.asyncio
async def test_conversation_history_update(mega_agent, mock_specialized_agents):
    """Test that conversation history is updated after handling a request."""
    # Configure router to return a specific agent
    mega_agent.router.route_request = AsyncMock(
        return_value=mock_specialized_agents["search"]
    )

    request = {"query": "Search for Python tutorials"}
    await mega_agent.handle_request(request)

    # Verify conversation history was updated
    assert len(mega_agent.conversation_history) == 1
    assert mega_agent.conversation_history[0]["query"] == "Search for Python tutorials"
    assert "search result" in str(mega_agent.conversation_history[0]["response"])


@pytest.mark.asyncio
async def test_agent_error_handling(mega_agent, mock_specialized_agents):
    """Test handling errors from specialized agents."""
    # Configure agent to raise an exception
    mock_specialized_agents["weather"].handle_request = AsyncMock(
        side_effect=Exception("Agent error")
    )

    # Configure router to return the problematic agent
    mega_agent.router.route_request = AsyncMock(
        return_value=mock_specialized_agents["weather"]
    )

    request = {"query": "What's the weather today?"}

    # Mega agent should handle the error
    with pytest.raises(Exception, match="Error from agent weather"):
        await mega_agent.handle_request(request)


@pytest.mark.asyncio
async def test_resolve_ambiguous_routing(mega_agent, mock_specialized_agents):
    """Test resolving ambiguous routing situations."""
    # Mock an ambiguous response from the router
    mega_agent.router.route_request = AsyncMock(return_value=None)
    mega_agent.router.get_potential_agents = AsyncMock(
        return_value=[
            mock_specialized_agents["calendar"],
            mock_specialized_agents["weather"],
        ]
    )

    # Mock the resolution process
    mega_agent.resolve_ambiguity = AsyncMock(
        return_value=mock_specialized_agents["calendar"]
    )

    request = {"query": "What's happening today?"}
    result = await mega_agent.handle_request(request)

    # Verify ambiguity resolution was called
    mega_agent.resolve_ambiguity.assert_called_once()
    assert result == {"result": "calendar result"}
