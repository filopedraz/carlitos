import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from carlitos.agent import Agent
from carlitos.llm import LLM
from carlitos.routing import Router


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLM)
    llm.generate = AsyncMock(
        return_value={
            "agent": "calendar",
            "confidence": 0.9,
            "reasoning": "This query is about calendar events",
        }
    )
    return llm


@pytest.fixture
def mock_agents():
    agents = [
        AsyncMock(spec=Agent, name="calendar"),
        AsyncMock(spec=Agent, name="weather"),
        AsyncMock(spec=Agent, name="search"),
        AsyncMock(spec=Agent, name="email"),
    ]

    # Configure mock agents with tools
    for agent in agents:
        agent.tools = {
            f"{agent.name}_tool1": MagicMock(),
            f"{agent.name}_tool2": MagicMock(),
        }

    return agents


@pytest.fixture
def router(mock_llm):
    router = Router(llm=mock_llm)
    return router


@pytest.mark.asyncio
async def test_router_initialization(mock_llm):
    """Test that router initializes with proper attributes."""
    router = Router(llm=mock_llm)

    assert router.llm == mock_llm
    assert isinstance(router.logger, object)  # Just verify logger exists


@pytest.mark.asyncio
async def test_route_request_single_agent(router, mock_agents, mock_llm):
    """Test routing a request to a single agent."""
    # Set up the mock LLM response for single agent routing
    mock_llm.generate = AsyncMock(
        return_value={
            "agent": "calendar",
            "confidence": 0.9,
            "reasoning": "This query is about calendar events",
        }
    )

    request = {"query": "What meetings do I have today?"}
    agent = await router.route_request(request, mock_agents)

    # Verify LLM was called
    mock_llm.generate.assert_called_once()

    # Verify the correct agent was selected
    assert agent.name == "calendar"


@pytest.mark.asyncio
async def test_route_request_multiple_agents(router, mock_agents, mock_llm):
    """Test routing a request to multiple agents."""
    # Set up the mock LLM response for multi-agent routing
    mock_llm.generate = AsyncMock(
        return_value={
            "agents": [
                {"name": "weather", "confidence": 0.8},
                {"name": "calendar", "confidence": 0.7},
            ],
            "reasoning": "This query requires both weather and calendar information",
        }
    )

    request = {"query": "Will it rain during my meetings today?"}
    agents = await router.route_request(request, mock_agents)

    # Verify LLM was called
    mock_llm.generate.assert_called_once()

    # Verify multiple agents were selected
    assert len(agents) == 2
    assert agents[0].name == "weather"
    assert agents[1].name == "calendar"


@pytest.mark.asyncio
async def test_route_request_no_matching_agent(router, mock_agents, mock_llm):
    """Test handling when no suitable agent is found."""
    # Set up the mock LLM response for no agent match
    mock_llm.generate = AsyncMock(
        return_value={
            "agent": None,
            "confidence": 0.0,
            "reasoning": "No suitable agent found for this query",
        }
    )

    request = {"query": "What is the meaning of life?"}
    result = await router.route_request(request, mock_agents)

    # Verify LLM was called
    mock_llm.generate.assert_called_once()

    # Verify no agent was selected
    assert result is None


@pytest.mark.asyncio
async def test_route_request_ambiguous_query(router, mock_agents, mock_llm):
    """Test handling of ambiguous queries that could match multiple agents."""
    # Set up the mock LLM response for ambiguous query
    mock_llm.generate = AsyncMock(
        return_value={
            "agent": None,
            "confidence": 0.0,
            "reasoning": "Query is ambiguous",
            "potential_agents": [
                {"name": "calendar", "confidence": 0.6},
                {"name": "email", "confidence": 0.5},
            ],
        }
    )

    request = {"query": "Check my schedule"}
    result = await router.route_request(request, mock_agents)

    # Verify LLM was called
    mock_llm.generate.assert_called_once()

    # In ambiguous cases, router might return None or potential agents
    assert result is None

    # Test getting potential agents
    potential_agents = await router.get_potential_agents(request, mock_agents)
    assert len(potential_agents) == 2
    assert potential_agents[0].name == "calendar"
    assert potential_agents[1].name == "email"


@pytest.mark.asyncio
async def test_route_with_context(router, mock_agents, mock_llm):
    """Test routing with conversation context."""
    # Set up conversation context
    context = [
        {"query": "What's the weather today?", "agent": "weather"},
        {"query": "Will I need an umbrella?", "agent": "weather"},
    ]

    # Set up the mock LLM response considering context
    mock_llm.generate = AsyncMock(
        return_value={
            "agent": "weather",
            "confidence": 0.95,
            "reasoning": "Continuing weather conversation",
        }
    )

    request = {"query": "What about tomorrow?", "context": context}
    agent = await router.route_request(request, mock_agents)

    # Verify LLM was called with context
    mock_llm.generate.assert_called_once()
    assert "context" in mock_llm.generate.call_args[0][0]

    # Verify the correct agent was selected based on context
    assert agent.name == "weather"


@pytest.mark.asyncio
async def test_tool_based_routing(router, mock_agents, mock_llm):
    """Test routing based on required tools."""
    # Set up the mock LLM response with tool requirements
    mock_llm.generate = AsyncMock(
        return_value={
            "required_tools": ["calendar_tool1", "calendar_tool2"],
            "reasoning": "This query requires calendar tools",
        }
    )

    request = {"query": "Schedule a meeting for tomorrow"}
    agent = await router.route_request(
        request, mock_agents, routing_method="tool_based"
    )

    # Verify LLM was called
    mock_llm.generate.assert_called_once()

    # Verify the agent with matching tools was selected
    assert agent.name == "calendar"


@pytest.mark.asyncio
async def test_routing_llm_error(router, mock_agents, mock_llm):
    """Test error handling when LLM fails."""
    # Configure LLM to raise an exception
    mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

    request = {"query": "What meetings do I have today?"}

    # Router should handle the error gracefully
    with pytest.raises(Exception, match="Error during routing"):
        await router.route_request(request, mock_agents)


@pytest.mark.asyncio
async def test_router_confidence_threshold(router, mock_agents, mock_llm):
    """Test that router respects confidence threshold."""
    # Set up the mock LLM response with low confidence
    mock_llm.generate = AsyncMock(
        return_value={
            "agent": "calendar",
            "confidence": 0.3,  # Below default threshold
            "reasoning": "Low confidence match",
        }
    )

    request = {"query": "Can you help me with something?"}
    result = await router.route_request(request, mock_agents, confidence_threshold=0.5)

    # Verify no agent was selected due to low confidence
    assert result is None

    # Try again with lower threshold
    result = await router.route_request(request, mock_agents, confidence_threshold=0.2)

    # Now an agent should be selected
    assert result.name == "calendar"
