import logging
import json
from typing import Dict, NamedTuple

from carlitos.agent import AgenticMCPAgent
from carlitos.config import CarlitosConfig, ServerConfig
from carlitos.prompt import CLARIFICATION_QUESTION_PROMPT, ROUTING_PROMPT

log = logging.getLogger("carlitos.mega_agent")


class ChatMessage(NamedTuple):
    """Simple structure to hold a chat message."""
    role: str  # "user" or "assistant"
    content: str


class MegaAgent:
    """
    A meta-agent that routes requests to specialized sub-agents based on the query type.
    Each sub-agent handles a specific integration (e.g., Gmail, Calendar, Slack).
    """
    
    def __init__(self, config: CarlitosConfig):
        """
        Initialize the MegaAgent with configuration.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.all_tools = None
        self.server_descriptions = {}
        self.sub_agents = {}
        self._last_tool_results = None
        self.integration_types = set()
        self._current_agent_type = None  # Track the current agent being used
        self.chat_history = []  # Store chat history temporarily
        self.user_id = "default"  # Default user ID
        
        # Extract server descriptions from config
        for server in config.servers:
            if hasattr(server, 'description') and server.description:
                self.server_descriptions[server.name] = server.description
                
                # Also keep track of integration types
                integration_type = self._get_integration_type(server.name)
                self.integration_types.add(integration_type)
        
        log.debug(f"Initialized MegaAgent with {len(config.servers)} potential servers")
    
    async def initialize_sub_agents(self):
        """
        Initialize all sub-agents by grouping servers by integration type.
        """
        # Group servers by integration type (e.g., gmail, calendar)
        integration_servers = {}
        
        # Create filters for each integration type
        for server in self.config.servers:
            integration_type = self._get_integration_type(server.name)
            if integration_type not in integration_servers:
                integration_servers[integration_type] = []
            integration_servers[integration_type].append(server)
        
        # Create a sub-agent for each integration type
        for integration_type, servers in integration_servers.items():
            # Create a filtered config for this integration
            filtered_config = CarlitosConfig(
                servers=servers,
                llm=self.config.llm
            )
            
            # Create the sub-agent
            self.sub_agents[integration_type] = AgenticMCPAgent(filtered_config)
            log.debug(f"Created sub-agent for {integration_type} with {len(servers)} servers")
        
        log.info(f"Initialized {len(self.sub_agents)} sub-agents: {', '.join(self.sub_agents.keys())}")
    
    def _get_integration_type(self, server_name: str) -> str:
        """
        Extract the integration type from a server name.
        
        Args:
            server_name: The name of the server
            
        Returns:
            The integration type (e.g., 'gmail', 'calendar')
        """
        # Extract integration type from server name (e.g., gmail_composio -> gmail)
        if '_' in server_name:
            return server_name.split('_')[0]
        return server_name
    
    async def chat(self, message: str, user_id: str = None) -> str:
        """
        Process a chat message by routing to appropriate sub-agents.
        
        Args:
            message: User message
            user_id: Optional user ID (defaults to self.user_id)
            
        Returns:
            Agent response
        """
        log.info(f"Processing chat message through MegaAgent: {message}")
        
        # Set user_id if provided
        if user_id:
            self.user_id = user_id
        
        # Add user message to chat history
        self.chat_history.append(ChatMessage(role="user", content=message))
        
        # Initialize sub-agents if not done yet
        if not self.sub_agents:
            await self.initialize_sub_agents()
        
        # Use lightweight routing based on integration descriptions
        relevant_agents = await self._lightweight_route_request(message)
        
        # If we have specific agents identified, use them
        if relevant_agents:
            agent_types = list(relevant_agents.keys())
            log.info(f"🔍 Routing decision: Using {len(relevant_agents)} specialized agent(s): {', '.join(agent_types)}")
            
            # If we have a single relevant agent, use it directly
            if len(relevant_agents) == 1:
                agent_type = agent_types[0]
                agent = relevant_agents[agent_type]
                self._current_agent_type = agent_type
                log.info(f"🎯 Activated agent: {agent_type}")
                response = await agent.chat(message)
                
                # Add response to chat history
                self.chat_history.append(ChatMessage(role="assistant", content=response))
                
                return response
            
            # If we have multiple relevant agents, coordinate between them
            log.info(f"🔄 Coordination required between {len(relevant_agents)} agents: {', '.join(agent_types)}")
            self._current_agent_type = f"coordinator({','.join(agent_types)})"
            response = await self._coordinate_multi_agent_response(message, relevant_agents)
            
            # Add response to chat history
            self.chat_history.append(ChatMessage(role="assistant", content=response))
            
            return response
        
        # If we couldn't determine relevant agents, ask a clarifying question
        log.info("❓ Cannot determine the appropriate agent, asking a clarifying question")
        self._current_agent_type = "clarification"
        
        # Get available integration descriptions for the clarification question
        available_integrations = self._format_integration_descriptions_short()
        clarification = await self._ask_clarification_question(message, available_integrations)
        
        # Add clarification to chat history
        self.chat_history.append(ChatMessage(role="assistant", content=clarification))
        
        return clarification
    
    async def _ask_clarification_question(self, message: str, available_integrations: str) -> str:
        """
        Ask a clarifying question when the intent is unclear.
        
        Args:
            message: The user's message
            available_integrations: Short descriptions of available integrations
            
        Returns:
            A clarifying question
        """
        # Get a client to make the request
        client = self.sub_agents[next(iter(self.sub_agents))].llm.client
        model = self.sub_agents[next(iter(self.sub_agents))].llm.model
        temperature = self.sub_agents[next(iter(self.sub_agents))].llm.temperature
        
        # Format chat history for context
        formatted_history = self._format_chat_history()
        
        # Create the prompt using the constant from prompt.py
        prompt = CLARIFICATION_QUESTION_PROMPT.format(
            formatted_history=formatted_history,
            message=message,
            available_integrations=available_integrations
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that asks clarifying questions to understand user needs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            log.error(f"Error generating clarification question: {e}")
            return "I'm not sure what you're asking for. Could you provide more details about what you'd like to do? For example, are you looking to check your calendar, email, or something else?"
    
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
    
    def _format_integration_descriptions_short(self) -> str:
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
    
    async def _lightweight_route_request(self, message: str) -> Dict[str, AgenticMCPAgent]:
        """
        Use a lightweight approach to route requests based only on integration descriptions.
        This avoids sending all tool details to the LLM for routing.
        
        Args:
            message: User message
            
        Returns:
            Dictionary of integration types to sub-agents
        """
        # If we have no server descriptions or no sub-agents, we can't route
        if not self.server_descriptions or not self.sub_agents:
            log.info("Cannot perform lightweight routing: missing server descriptions or sub-agents")
            return {}
            
        # Get chat history for context
        formatted_history = self._format_chat_history()
            
        # Create the prompt using the constant from prompt.py
        prompt = ROUTING_PROMPT.format(
            formatted_history=formatted_history,
            message=message,
            integrations_descriptions=self._format_integration_descriptions()
        )
        
        # Use the LLM from the main config
        client = self.sub_agents[next(iter(self.sub_agents))].llm.client
        model = self.sub_agents[next(iter(self.sub_agents))].llm.model
        temperature = self.sub_agents[next(iter(self.sub_agents))].llm.temperature
        
        try:
            log.info("Performing lightweight routing based on integration descriptions")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that routes requests to the appropriate integrations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
            
            response_text = response.choices[0].message.content
            
            # Clean response from code blocks
            response_text = self._clean_json_response(response_text)
            log.debug(f"Routing response: {response_text}")
            
            # Parse JSON response
            response_data = json.loads(response_text)
            
            integrations = response_data.get("integrations", [])
            reasoning = response_data.get("reasoning", "No reasoning provided")
            
            log.info(f"Lightweight routing reasoning: {reasoning}")
            log.info(f"Selected integrations: {integrations}")
            
            # Map integrations to agents
            relevant_agents = {}
            for integration in integrations:
                if integration in self.sub_agents:
                    relevant_agents[integration] = self.sub_agents[integration]
                    log.info(f"✅ Selected integration '{integration}' for routing")
                else:
                    log.warning(f"⚠️ Integration '{integration}' selected by router but no matching sub-agent found")
            
            return relevant_agents
            
        except Exception as e:
            log.error(f"Error in lightweight routing: {e}")
            return {}
    
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
    
    def _clean_json_response(self, response: str) -> str:
        """
        Clean JSON response to remove markdown code blocks and other formatting.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks (```json ... ```)
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        import re
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        
        # If no code blocks, just return the original (it might be clean already)
        return response.strip()
    
    async def _coordinate_multi_agent_response(self, message: str, relevant_agents: Dict[str, AgenticMCPAgent]) -> str:
        """
        Coordinate responses from multiple sub-agents for a complex request.
        
        Args:
            message: User message
            relevant_agents: Dictionary of relevant sub-agents
            
        Returns:
            Coordinated response
        """
        agent_types = list(relevant_agents.keys())
        log.info(f"🔄 Coordinating responses from {len(relevant_agents)} agents: {', '.join(agent_types)}")
        
        # Get responses from each relevant agent
        agent_responses = {}
        for agent_type, agent in relevant_agents.items():
            try:
                log.info(f"🔹 Getting response from {agent_type} agent")
                response = await agent.chat(message)
                agent_responses[agent_type] = response
                log.debug(f"Response from {agent_type} agent: {response[:100]}...")
            except Exception as e:
                log.error(f"❌ Error getting response from {agent_type} agent: {e}")
                agent_responses[agent_type] = f"Error: {str(e)}"
        
        # Use the first agent's LLM to synthesize a response
        first_agent = next(iter(relevant_agents.values()))
        
        # Format the sub-agent responses for synthesis
        combined_response = "\n\n".join([
            f"Response from {agent_type.upper()} agent:\n{response}"
            for agent_type, response in agent_responses.items()
        ])
        
        log.info(f"🔄 Synthesizing final response from {len(agent_responses)} sub-agent responses")
        
        # Synthesize the final response
        final_response = await first_agent.llm.synthesize_results(
            message,
            f"This request required coordinating between multiple specialized agents: {', '.join(relevant_agents.keys())}",
            combined_response,
            chat_history=self.chat_history
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