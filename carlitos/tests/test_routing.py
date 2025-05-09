from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from carlitos.llm import LLMCoreAgent as LLM
from carlitos.routing import RoutingAgent as Router


class MockRouter(Router):
    """A test-friendly version of RoutingAgent that accepts the parameters needed for tests."""

    def __init__(self, llm: Optional[AsyncMock] = None) -> None:
        # Skip the parent __init__ to avoid actual initialization
        # Router.__init__(self, routing_config, server_descriptions)

        self.llm = llm
        self.logger = MagicMock()

    async def route_request(
        self,
        request: Dict[str, Any],
        agents: List[Any],
        confidence_threshold: float = 0.5,
        routing_method: str = "semantic",
    ) -> Any:
        """Mock routing logic"""
        if self.llm:
            try:
                result = await self.llm.generate(request)

                # Handle single agent routing
                if (
                    "agent" in result
                    and result.get("confidence", 0) >= confidence_threshold
                ):
                    agent_name = result["agent"]
                    agent = next((a for a in agents if a.name == agent_name), None)
                    return agent

                # Handle multi-agent routing
                elif "agents" in result:
                    matched_agents = []
                    for agent_info in result["agents"]:
                        agent_name = agent_info["name"]
                        if agent_info.get("confidence", 0) >= confidence_threshold:
                            agent = next(
                                (a for a in agents if a.name == agent_name), None
                            )
                            if agent:
                                matched_agents.append(agent)

                    if matched_agents:
                        return matched_agents

                # Handle tool-based routing
                elif routing_method == "tool_based" and "required_tools" in result:
                    required_tools = result["required_tools"]
                    for agent in agents:
                        # Check if the agent has all required tools
                        if all(tool in agent.tools for tool in required_tools):
                            return agent

                return None

            except Exception as e:
                raise Exception(f"Error during routing: {str(e)}")

        return None

    async def get_potential_agents(
        self, request: Dict[str, Any], agents: List[Any]
    ) -> List[Any]:
        """Mock getting potential agents for ambiguous queries"""
        if self.llm:
            result = await self.llm.generate(request)
            if "potential_agents" in result:
                potential_agents = []
                for agent_info in result["potential_agents"]:
                    agent_name = agent_info["name"]
                    agent = next((a for a in agents if a.name == agent_name), None)
                    if agent:
                        potential_agents.append(agent)
                return potential_agents

        return []


@pytest.fixture
def mock_llm() -> AsyncMock:
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
def mock_agents() -> List[MagicMock]:
    # Create the mock agents without using spec which restricts attributes
    calendar_agent = MagicMock()
    calendar_agent.name = "calendar"
    calendar_agent.tools = {
        "calendar_tool1": MagicMock(),
        "calendar_tool2": MagicMock(),
    }

    weather_agent = MagicMock()
    weather_agent.name = "weather"
    weather_agent.tools = {"weather_tool1": MagicMock(), "weather_tool2": MagicMock()}

    search_agent = MagicMock()
    search_agent.name = "search"
    search_agent.tools = {"search_tool1": MagicMock(), "search_tool2": MagicMock()}

    email_agent = MagicMock()
    email_agent.name = "email"
    email_agent.tools = {"email_tool1": MagicMock(), "email_tool2": MagicMock()}

    # Add async handle_request method to each agent
    calendar_agent.handle_request = AsyncMock()
    weather_agent.handle_request = AsyncMock()
    search_agent.handle_request = AsyncMock()
    email_agent.handle_request = AsyncMock()

    return [calendar_agent, weather_agent, search_agent, email_agent]


@pytest.fixture
def router(mock_llm: AsyncMock) -> MockRouter:
    router = MockRouter(llm=mock_llm)
    return router


@pytest.mark.asyncio
async def test_router_initialization(mock_llm: AsyncMock) -> None:
    """Test that router initializes with proper attributes."""
    router = MockRouter(llm=mock_llm)

    assert router.llm == mock_llm
    assert isinstance(router.logger, object)  # Just verify logger exists


@pytest.mark.asyncio
async def test_route_request_single_agent(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
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
async def test_route_request_multiple_agents(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
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
async def test_route_request_no_matching_agent(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
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
async def test_route_request_ambiguous_query(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
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
async def test_route_with_context(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
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
async def test_tool_based_routing(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
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
async def test_routing_llm_error(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
    """Test error handling when LLM fails."""
    # Configure LLM to raise an exception
    mock_llm.generate = AsyncMock(side_effect=Exception("LLM error"))

    request = {"query": "What meetings do I have today?"}

    # Router should handle the error gracefully
    with pytest.raises(Exception, match="Error during routing"):
        await router.route_request(request, mock_agents)


@pytest.mark.asyncio
async def test_router_confidence_threshold(
    router: MockRouter, mock_agents: List[MagicMock], mock_llm: AsyncMock
) -> None:
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
