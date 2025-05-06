import re
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, NamedTuple

from mcp import Tool

from carlitos.agent import AgenticMCPAgent
from carlitos.config import CarlitosConfig, ServerConfig

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
        self.chat_history = []  # Store chat history temporarily before saving to memory
        self.user_id = "default"  # Default user ID for memory
        self.memory_server = None  # Will be set during initialization
        
        # Extract server descriptions from config
        for server in config.servers:
            if hasattr(server, 'description') and server.description:
                self.server_descriptions[server.name] = server.description
                
                # Also keep track of integration types
                integration_type = self._get_integration_type(server.name)
                self.integration_types.add(integration_type)
                
                # Find the memory server if present
                if "memory" in server.name.lower():
                    self.memory_server = server.name
        
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
            user_id: Optional user ID for memory (defaults to self.user_id)
            
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
        
        # Retrieve relevant memories for context if memory server is available
        memory_context = ""
        if self.memory_server and "memory" in self.sub_agents:
            memory_agent = self.sub_agents["memory"]
            try:
                # Search memory for relevant context
                memory_result = await memory_agent._execute_tool(
                    self.memory_server,
                    "search_memory",
                    {"query": message, "user_id": self.user_id, "limit": 3}
                )
                log.debug(f"Raw memory_result from memory_agent._execute_tool: {memory_result}")
                
                # Parse memory results
                memory_data = {}
                if isinstance(memory_result, str):
                    try:
                        # Check if the result starts with 'Content of type text:' and extract the JSON
                        if "Content of type text:" in memory_result:
                            # Extract the JSON part that follows the prefix
                            json_text = memory_result.split("Content of type text:", 1)[1].strip()
                            
                            # Try to find valid JSON by looking for matching braces
                            json_pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}')
                            json_match = json_pattern.search(json_text)
                            
                            if json_match:
                                valid_json = json_match.group(0)
                                log.debug(f"Extracted JSON: {valid_json}")
                                memory_data = json.loads(valid_json)
                            else:
                                log.error("Could not extract valid JSON from memory_result")
                        else:
                            # Try parsing as regular JSON
                            memory_data = json.loads(memory_result)
                    except json.JSONDecodeError as e:
                        log.error(f"Failed to decode memory_result string: {e} - Content: '{memory_result[:200]}...'") # Log part of the string
                        pass # Keep memory_data as empty dict
                elif isinstance(memory_result, dict):
                    memory_data = memory_result # Already a dict
                else:
                    log.warning(f"Unexpected type for memory_result: {type(memory_result)}. Expected str or dict.")

                log.debug(f"Parsed memory_data: {memory_data}")

                # Extract relevant memories as context
                if memory_data.get("success") and memory_data.get("results"):
                    memories = memory_data["results"]
                    
                    # Format memory as context
                    memory_context = "Relevant context from memory:\n"
                    for i, memory in enumerate(memories):
                        memory_context += f"{i+1}. {memory}\n"
                    
                    log.info(f"Retrieved {len(memories)} relevant memories for context")
                    log.debug(f"Formatted memory_context: {memory_context}")
            except Exception as e:
                log.error(f"Error retrieving memories: {e}")
            
        # Use lightweight routing based on integration descriptions
        relevant_agents = await self._lightweight_route_request(message, memory_context)
        
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
                
                # Save conversation to memory if memory server is available
                await self._save_conversation_to_memory(message, response)
                
                return response
            
            # If we have multiple relevant agents, coordinate between them
            log.info(f"🔄 Coordination required between {len(relevant_agents)} agents: {', '.join(agent_types)}")
            self._current_agent_type = f"coordinator({','.join(agent_types)})"
            response = await self._coordinate_multi_agent_response(message, relevant_agents)
            
            # Add response to chat history
            self.chat_history.append(ChatMessage(role="assistant", content=response))
            
            # Save conversation to memory if memory server is available
            await self._save_conversation_to_memory(message, response)
            
            return response
        
        # If we couldn't determine relevant agents, ask a clarifying question
        log.info("❓ Cannot determine the appropriate agent, asking a clarifying question")
        self._current_agent_type = "clarification"
        
        # Get available integration descriptions for the clarification question
        available_integrations = self._format_integration_descriptions_short()
        clarification = await self._ask_clarification_question(message, available_integrations)
        
        # Add clarification to chat history
        self.chat_history.append(ChatMessage(role="assistant", content=clarification))
        
        # Save conversation to memory if memory server is available
        await self._save_conversation_to_memory(message, clarification)
        
        return clarification
    
    async def _save_conversation_to_memory(self, user_message: str, assistant_response: str):
        """
        Save the conversation to memory using the memory server.
        
        Args:
            user_message: User message
            assistant_response: Assistant response
        """
        if not self.memory_server or "memory" not in self.sub_agents:
            log.debug("Memory server not available, skipping conversation save")
            return
        
        memory_agent = self.sub_agents["memory"]
        try:
            # Add conversation to memory
            await memory_agent._execute_tool(
                self.memory_server,
                "add_conversation",
                {
                    "user_message": user_message,
                    "assistant_message": assistant_response,
                    "user_id": self.user_id
                }
            )
            log.info("Saved conversation to memory")
        except Exception as e:
            log.error(f"Error saving conversation to memory: {e}")
    
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
        
        prompt = f"""
You are Carlitos, a personal assistant that routes requests to specialized tools based on the user's needs.
You need to ask a clarifying question because you can't determine which specialized system to use.

Current chat history:
{formatted_history}

Most recent user message: {message}

Available specialized systems:
{available_integrations}

Please ask a clarifying question to help determine which specialized system the user needs.
Your question should:
1. Be conversational and friendly
2. Politely explain that you need more context
3. Offer 2-3 specific options related to the available systems that might match what they're asking for
4. Be concise (1-3 sentences maximum)

Do not apologize profusely or be overly formal. Be helpful and direct.
"""
        
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
            
        # Format the history (limit to last 5 exchanges to save tokens)
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
    
    async def _lightweight_route_request(self, message: str, memory_context: str = "") -> Dict[str, AgenticMCPAgent]:
        """
        Use a lightweight approach to route requests based only on integration descriptions.
        This avoids sending all tool details to the LLM for routing.
        
        Args:
            message: User message
            memory_context: Optional context from memory
            
        Returns:
            Dictionary of integration types to sub-agents
        """
        # If we have no server descriptions or no sub-agents, we can't route
        if not self.server_descriptions or not self.sub_agents:
            log.info("Cannot perform lightweight routing: missing server descriptions or sub-agents")
            return {}
            
        # Get chat history for context
        formatted_history = self._format_chat_history()
            
        # Prepare a prompt that only includes integration descriptions
        prompt = f"""
You are Carlitos, a routing assistant. Your task is to determine which integration(s) should handle this request.

### Chat History:
{formatted_history}

### Current User Query:
{message}

{memory_context}

### Available Integrations:
{self._format_integration_descriptions()}

Based on the user's query AND previous context, determine which integration(s) would be most appropriate to handle this request.
Only select integrations that are directly relevant to the request.
If none of the integrations are relevant, return an empty list.

Respond in the following JSON format:
{{
    "reasoning": "Your analysis of why certain integrations are needed",
    "integrations": ["integration1", "integration2"]
}}
"""
        
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
    
    async def _discover_all_tools(self):
        """
        Discover all tools from all servers.
        """
        all_tools = []
        
        # Create a temporary agent to discover tools
        temp_agent = AgenticMCPAgent(self.config)
        self.all_tools = await temp_agent._discover_tools()
        
        log.info(f"Discovered {len(self.all_tools)} tools across all servers")
    
    async def _route_request_by_tools(self, message: str) -> Dict[str, AgenticMCPAgent]:
        """
        Determine which sub-agent(s) should handle a request based on tool selection.
        This is a fallback for when lightweight routing doesn't work.
        
        Args:
            message: User message
            
        Returns:
            Dictionary of integration types to sub-agents
        """
        relevant_agents = {}
        
        # If we don't have tools or subagents, we can't route
        if not self.all_tools or not self.sub_agents:
            log.info("Cannot perform tool-based routing: missing tools or sub-agents")
            return {}
        
        try:
            log.info("Performing tool-based routing analysis")
            # Use the LLM from any sub-agent (they all share the same LLM config)
            first_agent = next(iter(self.sub_agents.values()))
            thinking, needed_tools = await first_agent.llm.analyze_task(
                message, self.all_tools, chat_history=self.chat_history, detailed_server_info=self.server_descriptions
            )
            
            log.debug(f"Tool-based routing thinking: {thinking}")
            
            if not needed_tools:
                log.info("Tool-based routing: No tools needed for this request")
                return {}
                
            # Map tools to their integration types
            integration_tools = {}
            for tool_info in needed_tools:
                tool_name = tool_info.get("name")
                # Find which server/integration this tool belongs to
                for server_name, tools in first_agent._server_tools_map.items():
                    if tool_name in tools:
                        integration_type = self._get_integration_type(server_name)
                        if integration_type not in integration_tools:
                            integration_tools[integration_type] = []
                        integration_tools[integration_type].append(tool_info)
            
            # Add relevant agents based on tools needed
            for integration_type, tools in integration_tools.items():
                if integration_type in self.sub_agents:
                    relevant_agents[integration_type] = self.sub_agents[integration_type]
                    tool_names = [t.get('name') for t in tools]
                    log.info(f"✅ Selected integration '{integration_type}' based on needed tools: {tool_names}")
                else:
                    log.warning(f"⚠️ Integration '{integration_type}' needed but no matching sub-agent found")
            
        except Exception as e:
            log.error(f"Error in tool-based routing: {e}")
            
        return relevant_agents
    
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