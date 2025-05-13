import logging
from typing import Any, Dict, List, NamedTuple

from carlitos.agent import AgenticMCPAgent
from carlitos.config import DEFAULT_LLM_CONFIG, ROUTING_LLM_CONFIG
from carlitos.routing import RoutingAgent

logger = logging.getLogger("carlitos.mega_agent")


class ChatMessage(NamedTuple):
    """Simple structure to hold a chat message."""

    role: str  # "user" or "assistant"
    content: str


class MegaAgent:
    """
    A meta-agent that routes requests to specialized sub-agents based on the query type.
    Each sub-agent handles a specific integration (e.g., Gmail, Calendar, Slack).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MegaAgent with configuration.

        Args:
            config: Agent configuration dictionary
        """
        self.config = config
        self.all_tools = None
        self.server_descriptions = {}
        self.sub_agents = {}
        self._last_tool_results = None
        self.integration_types = set()
        self._current_agent_type = None  # Track the current agent being used
        self.chat_history = []  # Store chat history temporarily

        # Extract server descriptions from config
        for server in config["servers"]:
            if "description" in server and server["description"]:
                self.server_descriptions[server["name"]] = server["description"]

                # Also keep track of integration types
                integration_type = self._get_integration_type(server["name"])
                self.integration_types.add(integration_type)

        # Initialize the routing agent
        routing_config = config.get("routing", ROUTING_LLM_CONFIG)
        self.routing_agent = RoutingAgent(routing_config, self.server_descriptions)

        logger.debug(
            f"Initialized MegaAgent with {len(config['servers'])} potential servers"
        )

    async def initialize_sub_agents(self):
        """
        Initialize all sub-agents by grouping servers by integration type.
        """
        # Group servers by integration type (e.g., gmail, calendar)
        integration_servers = {}

        # Create filters for each integration type
        for server in self.config["servers"]:
            integration_type = self._get_integration_type(server["name"])
            if integration_type not in integration_servers:
                integration_servers[integration_type] = []
            integration_servers[integration_type].append(server)

        # Create a sub-agent for each integration type
        for integration_type, servers in integration_servers.items():
            # Create a filtered config for this integration
            filtered_config = {
                "servers": servers,
                "llm": DEFAULT_LLM_CONFIG,  # Use the default LLM config from constants
            }

            # Create the sub-agent
            self.sub_agents[integration_type] = AgenticMCPAgent(filtered_config)
            logger.debug(
                f"Created sub-agent for {integration_type} with {len(servers)} servers"
            )

        logger.info(
            f"Initialized {len(self.sub_agents)} sub-agents: {', '.join(self.sub_agents.keys())}"
        )

    def _get_integration_type(self, server_name: str) -> str:
        """
        Extract the integration type from a server name.

        Args:
            server_name: The name of the server

        Returns:
            The integration type (e.g., 'gmail', 'calendar')
        """
        # Extract integration type from server name (e.g., gmail_composio -> gmail)
        if "_" in server_name:
            return server_name.split("_")[0]
        return server_name

    async def chat(self, message: str) -> str:
        """
        Process a chat message by routing to appropriate sub-agents.

        Args:
            message: User message

        Returns:
            Agent response
        """
        logger.info(f"Processing chat message through MegaAgent: {message}")

        # Add user message to chat history
        self.chat_history.append(ChatMessage(role="user", content=message))

        # Initialize sub-agents if not done yet
        if not self.sub_agents:
            await self.initialize_sub_agents()

        # Get chat history for context
        formatted_history = self._format_chat_history()

        # Use the routing agent to identify relevant agents
        routing_result = await self.routing_agent.identify_relevant_agents(
            message, list(self.integration_types), formatted_history
        )

        # Get relevant agent types from the routing result
        relevant_agent_types = routing_result["integrations"]

        # Map agent types to actual agent instances
        relevant_agents = {}
        for agent_type in relevant_agent_types:
            if agent_type in self.sub_agents:
                relevant_agents[agent_type] = self.sub_agents[agent_type]
                logger.info(f"✅ Selected integration '{agent_type}' for routing")

        # If we have specific agents identified, use them
        if relevant_agents:
            agent_types = list(relevant_agents.keys())
            logger.info(
                f"🔍 Routing decision: Using {len(relevant_agents)} specialized agent(s): {', '.join(agent_types)}"
            )

            # If we have a single relevant agent, use it directly
            if len(relevant_agents) == 1:
                agent_type = agent_types[0]
                agent = relevant_agents[agent_type]
                self._current_agent_type = agent_type
                logger.info(f"🎯 Activated agent: {agent_type}")

                # Convert chat history to the format expected by AgenticMCPAgent
                formatted_chat_history = self._format_chat_history_for_agent()

                # Pass the chat history to the sub-agent
                response = await agent.chat(
                    message, chat_history=formatted_chat_history
                )

                # Add response to chat history
                self.chat_history.append(
                    ChatMessage(role="assistant", content=response)
                )

                return response

            # If we have multiple relevant agents, coordinate between them
            logger.info(
                f"🔄 Coordination required between {len(relevant_agents)} agents: {', '.join(agent_types)}"
            )
            self._current_agent_type = f"coordinator({','.join(agent_types)})"
            response = await self._coordinate_multi_agent_response(
                message, relevant_agents
            )

            # Add response to chat history
            self.chat_history.append(ChatMessage(role="assistant", content=response))

            return response

        # If we couldn't determine relevant agents, ask a clarifying question
        logger.info(
            "❓ Cannot determine the appropriate agent, asking a clarifying question"
        )
        self._current_agent_type = "clarification"

        # Get available integration descriptions for the clarification question
        available_integrations = self._format_integration_descriptions()

        # Use the routing agent to ask a clarification question
        clarification = await self.routing_agent.ask_clarification_question(
            message, formatted_history, available_integrations
        )

        # Add clarification to chat history
        self.chat_history.append(ChatMessage(role="assistant", content=clarification))

        return clarification

    def _format_chat_history(self) -> str:
        """
        Format the chat history for inclusion in prompts.

        Returns:
            Formatted chat history string
        """
        if not self.chat_history:
            return "No previous conversation."

        # Format the history (limit to last 10 exchanges to save tokens)
        formatted = []
        for msg in self.chat_history[-10:]:  # Last 10 messages
            prefix = "User: " if msg.role == "user" else "Carlitos: "
            formatted.append(f"{prefix}{msg.content}")

        return "\n\n".join(formatted)

    def _format_integration_descriptions(self) -> str:
        """
        Format integration descriptions in a short format for clarification questions.

        Returns:
            Short formatted integration descriptions
        """
        integration_descriptions = {}

        for server_name, description in self.server_descriptions.items():
            integration_type = self._get_integration_type(server_name)
            integration_descriptions[integration_type] = description

        # Format as a bullet list with just the integration names
        formatted = []
        for integration_type in sorted(integration_descriptions.keys()):
            formatted.append(f"- {integration_type}")

        return "\n".join(formatted)

    def _format_chat_history_for_agent(self) -> List[Dict[str, str]]:
        """
        Format chat history for passing to sub-agents.

        Returns:
            Chat history in the format expected by AgenticMCPAgent
        """
        # Transform ChatMessage objects to the dict format expected by AgenticMCPAgent
        formatted_history = []
        for msg in self.chat_history:
            formatted_history.append({"role": msg.role, "content": msg.content})
        return formatted_history

    async def _coordinate_multi_agent_response(
        self, message: str, relevant_agents: Dict[str, AgenticMCPAgent]
    ) -> str:
        """
        Coordinate responses from multiple sub-agents for a complex request.

        Args:
            message: User message
            relevant_agents: Dictionary of relevant sub-agents

        Returns:
            Coordinated response
        """
        agent_types = list(relevant_agents.keys())
        logger.info(
            f"🔄 Coordinating responses from {len(relevant_agents)} agents: {', '.join(agent_types)}"
        )

        # Format chat history for sub-agents
        formatted_chat_history = self._format_chat_history_for_agent()

        # Get responses from each relevant agent
        agent_responses = {}
        for agent_type, agent in relevant_agents.items():
            try:
                logger.info(f"🔹 Getting response from {agent_type} agent")

                # Pass the chat history to each sub-agent
                response = await agent.chat(
                    message, chat_history=formatted_chat_history
                )

                agent_responses[agent_type] = response
                logger.debug(f"Response from {agent_type} agent: {response[:100]}...")
            except Exception as e:
                logger.error(f"❌ Error getting response from {agent_type} agent: {e}")
                agent_responses[agent_type] = f"Error: {str(e)}"

        # Use the first agent's LLM to synthesize a response
        first_agent = next(iter(relevant_agents.values()))

        # Format the sub-agent responses for synthesis
        combined_response = "\n\n".join(
            [
                f"Response from {agent_type.upper()} agent:\n{response}"
                for agent_type, response in agent_responses.items()
            ]
        )

        logger.info(
            f"🔄 Synthesizing final response from {len(agent_responses)} sub-agent responses"
        )

        # Synthesize the final response
        final_response = await first_agent.llm.synthesize_results(
            message,
            f"This request required coordinating between multiple specialized agents: {', '.join(relevant_agents.keys())}",
            combined_response,
            chat_history=self._format_chat_history_for_agent(),
        )

        return final_response

    async def run(self, query: str) -> str:
        """
        Run the agent with a one-shot query.

        Args:
            query: User query

        Returns:
            Agent response
        """
        # Initialize sub-agents if not done yet
        if not self.sub_agents:
            await self.initialize_sub_agents()

        # Delegate to chat method
        return await self.chat(query)
