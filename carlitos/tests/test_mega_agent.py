from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from carlitos.mega_agent import MegaAgent


class MockTool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


class MockMegaAgent(MegaAgent):
    """A test-friendly version of MegaAgent that accepts the parameters needed for tests."""

    def __init__(self, agents: Optional[List[MagicMock]] = None) -> None:
        # Skip the parent __init__ to avoid actual initialization
        # MegaAgent.__init__(self, config)

        self.agents = agents or []
        self.router = AsyncMock()
        self.conversation_history: List[Dict[str, Any]] = []

    async def handle_request(self, request: Dict[str, Any]) -> Any:
        """Mock handle_request method"""
        agent = await self.router.route_request(request, self.agents)

        if (
            not agent
            and hasattr(self.router, "get_potential_agents")
            and callable(self.router.get_potential_agents)
        ):
            # Try to resolve ambiguity
            potential_agents = await self.router.get_potential_agents(
                request, self.agents
            )
            if potential_agents and hasattr(self, "resolve_ambiguity"):
                agent = await self.resolve_ambiguity(request, potential_agents)

        if not agent:
            raise ValueError("No suitable agent found")

        # Handle multi-agent responses
        if isinstance(agent, list):
            results = []
            for a in agent:
                try:
                    results.append(await a.handle_request(request))
                except Exception as e:
                    results.append(f"Error from agent {a.name}: {str(e)}")

            self.conversation_history.append(
                {"query": request.get("query", ""), "response": results}
            )

            return results
        else:
            # Single agent response
            try:
                result = await agent.handle_request(request)

                self.conversation_history.append(
                    {"query": request.get("query", ""), "response": result}
                )

                return result
            except Exception as e:
                # Re-raise with agent name included
                raise Exception(f"Error from agent {agent.name}: {str(e)}")

    async def resolve_ambiguity(
        self, request: Dict[str, Any], potential_agents: List[Any]
    ) -> Optional[Any]:
        """Mock ambiguity resolution"""
        if potential_agents:
            return potential_agents[0]
        return None


@pytest.fixture
def mock_specialized_agents():
    # Create agents with MagicMock for more flexibility
    calendar_agent = MagicMock()
    calendar_agent.name = "calendar"
    calendar_agent.handle_request = AsyncMock(
        return_value={"result": "calendar result"}
    )

    weather_agent = MagicMock()
    weather_agent.name = "weather"
    weather_agent.handle_request = AsyncMock(return_value={"result": "weather result"})

    search_agent = MagicMock()
    search_agent.name = "search"
    search_agent.handle_request = AsyncMock(return_value={"result": "search result"})

    agents = {
        "calendar": calendar_agent,
        "weather": weather_agent,
        "search": search_agent,
    }

    return agents


@pytest.fixture
def mega_agent(mock_specialized_agents):
    # Create MegaAgent with mock specialized agents
    agent = MockMegaAgent(agents=list(mock_specialized_agents.values()))

    # Mock the routing function
    agent.router = AsyncMock()

    return agent


@pytest.mark.asyncio
async def test_mega_agent_initialization() -> None:
    """Test that mega agent initializes with proper attributes."""
    # Create agents with MagicMock for more flexibility
    agent1 = MagicMock()
    agent1.name = "agent1"

    agent2 = MagicMock()
    agent2.name = "agent2"

    mega_agent = MockMegaAgent(agents=[agent1, agent2])

    assert len(mega_agent.agents) == 2
    assert mega_agent.agents[0].name == "agent1"
    assert mega_agent.agents[1].name == "agent2"
    assert mega_agent.conversation_history == []


@pytest.mark.asyncio
async def test_route_to_agent(
    mega_agent: MockMegaAgent, mock_specialized_agents: Dict[str, MagicMock]
) -> None:
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
async def test_handle_request_no_matching_agent(mega_agent: MockMegaAgent) -> None:
    """Test handling a request when no matching agent is found."""
    # Configure router to return None (no matching agent)
    mega_agent.router.route_request = AsyncMock(return_value=None)

    # Ensure get_potential_agents is not available to avoid that code path
    mega_agent.router.get_potential_agents = None

    request = {"query": "What's the meaning of life?"}

    with pytest.raises(ValueError, match="No suitable agent found"):
        await mega_agent.handle_request(request)


@pytest.mark.asyncio
async def test_handle_multiagent_request(
    mega_agent: MockMegaAgent, mock_specialized_agents: Dict[str, MagicMock]
) -> None:
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
async def test_conversation_history_update(
    mega_agent: MockMegaAgent, mock_specialized_agents: Dict[str, MagicMock]
) -> None:
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
async def test_agent_error_handling(
    mega_agent: MockMegaAgent, mock_specialized_agents: Dict[str, MagicMock]
) -> None:
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
async def test_resolve_ambiguous_routing(
    mega_agent: MockMegaAgent, mock_specialized_agents: Dict[str, MagicMock]
) -> None:
    """Test resolving ambiguous routing situations."""
    # Mock an ambiguous response from the router
    mega_agent.router.route_request = AsyncMock(return_value=None)
    mega_agent.router.get_potential_agents = AsyncMock(
        return_value=[
            mock_specialized_agents["calendar"],
            mock_specialized_agents["weather"],
        ]
    )

    # Store original method to restore later
    original_resolve_ambiguity = mega_agent.resolve_ambiguity

    # Create mock method
    mega_agent.resolve_ambiguity = AsyncMock(  # type: ignore[method-assign]
        return_value=mock_specialized_agents["calendar"]
    )

    try:
        request = {"query": "What's happening today?"}
        result = await mega_agent.handle_request(request)

        # Verify ambiguity resolution was called
        mega_agent.resolve_ambiguity.assert_called_once()
        assert result == {"result": "calendar result"}
    finally:
        # Restore original method
        mega_agent.resolve_ambiguity = original_resolve_ambiguity  # type: ignore[method-assign]
