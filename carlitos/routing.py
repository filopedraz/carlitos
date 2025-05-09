import json
import logging
from typing import Any, Dict, List

from pydantic_ai import Agent

from carlitos.prompt import (
    CLARIFICATION_QUESTION_PROMPT,
    CLARIFICATION_SYSTEM_PROMPT,
    ROUTING_PROMPT,
    ROUTING_SYSTEM_PROMPT,
)

logger = logging.getLogger("carlitos.routing")


class RoutingAgent:
    """
    Agent responsible for routing user requests to the appropriate specialized sub-agents.
    Uses a lightweight model for fast decision-making.
    """

    def __init__(
        self, routing_config: Dict[str, Any], server_descriptions: Dict[str, str]
    ):
        """
        Initialize the RoutingAgent with configuration.

        Args:
            routing_config: LLM configuration for routing
            server_descriptions: Dictionary mapping server names to their descriptions
        """
        self.routing_config = routing_config
        self.server_descriptions = server_descriptions
        self.model_name = f"google-gla:{routing_config['model']}"
        self.temperature = routing_config["temperature"]

        # Log service name for instrumentation
        service_name = routing_config.get("service_name", "carlitos")
        logger.debug(f"Using service name: {service_name} for instrumentation")

        logger.debug(f"Initialized RoutingAgent with model {self.model_name}")

    async def identify_relevant_agents(
        self, message: str, integration_types: List[str], formatted_history: str
    ) -> Dict[str, List[str]]:
        """
        Identify which specialized agents are relevant to the user's request.

        Args:
            message: User message
            integration_types: List of available integration types
            formatted_history: Formatted chat history for context

        Returns:
            Dictionary with "integrations" list and "reasoning" string
        """
        # If we have no server descriptions, we can't route
        if not self.server_descriptions:
            logger.info("Cannot identify relevant agents: missing server descriptions")
            return {"integrations": [], "reasoning": "No server descriptions available"}

        # Create the prompt using the constant from prompt.py with matching parameter names
        prompt = ROUTING_PROMPT.format(
            formatted_history=formatted_history,
            message=message,
            integrations_descriptions=self._format_integration_descriptions(),
        )

        try:
            logger.info("Identifying relevant agents based on integration descriptions")

            # Create a dedicated agent with the proper system prompt for routing
            routing_agent = Agent(
                self.model_name, system_prompt=ROUTING_SYSTEM_PROMPT, instrument=True
            )

            result = await routing_agent.run(prompt, temperature=self.temperature)

            response_text = result.output

            # Clean response from code blocks
            response_text = self._clean_json_response(response_text)
            logger.debug(f"Routing response: {response_text}")

            # Parse JSON response
            response_data = json.loads(response_text)

            integrations = response_data.get("integrations", [])
            reasoning = response_data.get("reasoning", "No reasoning provided")

            logger.info(f"Routing reasoning: {reasoning}")
            logger.info(f"Selected integrations: {integrations}")

            # Validate that integrations exist in our list
            valid_integrations = [i for i in integrations if i in integration_types]

            if len(valid_integrations) < len(integrations):
                logger.warning(
                    f"Some selected integrations are not available: {set(integrations) - set(valid_integrations)}"
                )

            return {"integrations": valid_integrations, "reasoning": reasoning}

        except Exception as e:
            logger.error(f"Error identifying relevant agents: {e}")
            return {"integrations": [], "reasoning": f"Error: {str(e)}"}

    async def ask_clarification_question(
        self, message: str, formatted_history: str, available_integrations: str
    ) -> str:
        """
        Ask a clarifying question when the intent is unclear.

        Args:
            message: The user's message
            formatted_history: Formatted chat history for context
            available_integrations: Short descriptions of available integrations

        Returns:
            A clarifying question
        """
        # Create the prompt using the constant from prompt.py
        prompt = CLARIFICATION_QUESTION_PROMPT.format(
            formatted_history=formatted_history,
            message=message,
            available_integrations=available_integrations,
        )

        try:
            # Create a dedicated agent with the proper system prompt for clarification
            clarification_agent = Agent(
                self.model_name,
                system_prompt=CLARIFICATION_SYSTEM_PROMPT,
                instrument=True,
            )

            result = await clarification_agent.run(prompt, temperature=self.temperature)

            return result.output

        except Exception as e:
            logger.error(f"Error generating clarification question: {e}")
            return "I'm not sure what you're asking for. Could you provide more details about what you'd like to do? For example, are you looking to check your calendar, email, or something else?"

    def _format_integration_descriptions(self) -> str:
        """
        Format integration descriptions for the routing prompt.

        Returns:
            Formatted integration descriptions string
        """
        # Group descriptions by integration type
        integration_descriptions = {}

        for server_name, description in self.server_descriptions.items():
            integration_type = self._get_integration_type(server_name)
            integration_descriptions[integration_type] = description

        # Format the descriptions
        formatted_descriptions = []
        for integration_type, description in integration_descriptions.items():
            formatted_descriptions.append(f"- {integration_type}: {description}")

        return "\n".join(formatted_descriptions)

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

    def _clean_json_response(self, text: str) -> str:
        """
        Clean JSON response from code blocks.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove code block markers if present
        if "```json" in text and "```" in text:
            text = text.split("```json", 1)[1]
            text = text.split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1]
            text = text.split("```", 1)[0]

        return text.strip()
