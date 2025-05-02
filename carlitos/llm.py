import json
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from langfuse.openai import openai
from mcp import Tool

from carlitos.config import LLMConfig
from carlitos.prompt import TASK_ANALYSIS_PROMPT, SYNTHESIS_PROMPT, TOOL_SELECTION_PROMPT

log = logging.getLogger("carlitos.llm")


class AgenticLLMToolSelector:
    """
    An agentic LLM tool selector that supports deliberate thinking 
    about tasks and tools to use.
    """
    
    def __init__(self, llm_config: LLMConfig):
        """
        Initialize LLM client based on config.
        
        Args:
            llm_config: LLM configuration
        """
        self.config = llm_config
        self.provider = "openai"  # Only support OpenAI
        self.model = llm_config.model
        self.temperature = llm_config.temperature
        
        # Set API key from environment
        api_key = os.environ.get(llm_config.api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable {llm_config.api_key_env}")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
        log.debug(f"Initialized OpenAI client with model {self.model}")
    
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
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        
        # If no code blocks, just return the original (it might be clean already)
        return response.strip()
    
    async def select_tool(self, query: str, available_tools: List[Tool]) -> Tuple[Optional[str], Dict[str, Any], str]:
        """
        Prompt LLM to select appropriate tool and parameters.
        
        Args:
            query: User query
            available_tools: List of available tools
            
        Returns:
            Tuple of (tool_name, parameters, reasoning)
            If tool_name is None, LLM decided no tool is needed
        """
        # Do not filter tools, use all available tools
        log.debug(f"Using all {len(available_tools)} tools without filtering")
        
        # Create prompt for tool selection
        prompt = self._get_tool_selection_prompt(query, available_tools)
        
        log.debug(f"Sending prompt to OpenAI")
        
        try:
            # Get response from OpenAI
            response = self._get_openai_response(prompt)
            
            # Clean response from code blocks  
            cleaned_response = self._clean_json_response(response)
            
            # Parse JSON response
            response_data = json.loads(cleaned_response)
            
            tool_name = response_data.get("tool")
            parameters = response_data.get("parameters", {})
            reasoning = response_data.get("reasoning", "No reasoning provided")
            
            # If tool is NONE, return None for tool name
            if tool_name == "NONE":
                tool_name = None
                
            log.debug(f"Selected tool: {tool_name}, Parameters: {parameters}")
            return tool_name, parameters, reasoning
            
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON from LLM response: {e}")
            log.error(f"Raw response: {response}")
            raise ValueError("LLM returned invalid JSON response")
        except Exception as e:
            log.error(f"Error getting tool selection from LLM: {e}")
            raise
    
    async def analyze_task(self, query: str, available_tools: List[Tool], chat_history: List[Dict[str, str]] = None, detailed_server_info: Dict[str, str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Analyze the task to determine what tools might be needed.
        This is the "thinking" phase of the agentic process.
        
        Args:
            query: User query
            available_tools: List of available tools
            chat_history: Optional chat history
            detailed_server_info: Optional dictionary of server names to descriptions
            
        Returns:
            Tuple of (thinking, needed_tools)
            Where thinking is the agent's thought process
            And needed_tools is a list of tool specifications
        """
        # Use all available tools without filtering
        log.debug(f"Using all {len(available_tools)} tools without filtering")
        
        # Format chat history if available
        chat_history_formatted = ""
        if chat_history:
            try:
                # Format chat history as a string
                history_items = []
                for msg in chat_history[-5:]:  # Last 5 messages to save tokens
                    prefix = "User: " if msg.role == "user" else "Carlitos: "
                    history_items.append(f"{prefix}{msg.content}")
                chat_history_formatted = "\n\n".join(history_items)
            except Exception as e:
                log.warning(f"Error formatting chat history: {e}")
                chat_history_formatted = ""
        
        # Create prompt for task analysis
        prompt = self._get_task_analysis_prompt(query, available_tools, detailed_server_info, chat_history_formatted)
        
        log.debug(f"Sending task analysis prompt to OpenAI")
        
        try:
            # Get response from OpenAI
            response = self._get_openai_response(prompt)
            
            # Clean response from code blocks
            cleaned_response = self._clean_json_response(response)
            log.debug(f"Cleaned response: {cleaned_response[:100]}...")
            
            # Parse JSON response
            response_data = json.loads(cleaned_response)
            
            thinking = response_data.get("thinking", "No thinking provided")
            needed_tools = response_data.get("tools", [])
            
            log.debug(f"Thinking: {thinking[:100]}...")
            log.debug(f"Needed tools: {[t.get('name') for t in needed_tools]}")
            
            return thinking, needed_tools
            
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON from LLM response: {e}")
            log.error(f"Raw response: {response}")
            # Fallback to simple mode
            return "I couldn't properly analyze the task. Let me try to help directly.", []
        except Exception as e:
            log.error(f"Error in task analysis: {e}")
            return f"I encountered an error while analyzing your request: {str(e)}", []
    
    async def synthesize_results(self, query: str, thinking: str, tool_results: str, chat_history: List = None) -> str:
        """
        Synthesize the results of tool executions into a coherent response.
        
        Args:
            query: Original user query
            thinking: Initial agent thinking
            tool_results: Results from tool executions
            chat_history: Optional chat history
            
        Returns:
            Synthesized response
        """
        # Format chat history if available
        chat_history_formatted = ""
        if chat_history:
            try:
                # Format chat history as a string
                history_items = []
                for msg in chat_history[-5:]:  # Last 5 messages to save tokens
                    prefix = "User: " if msg.role == "user" else "Carlitos: "
                    history_items.append(f"{prefix}{msg.content}")
                chat_history_formatted = "\n\n".join(history_items)
            except Exception as e:
                log.warning(f"Error formatting chat history: {e}")
                chat_history_formatted = ""
                
        # Create prompt for synthesis
        prompt = self._get_synthesis_prompt(query, thinking, tool_results, chat_history_formatted)
        
        log.debug(f"Sending synthesis prompt to OpenAI")
        
        try:
            # Get response from OpenAI
            response = self._get_openai_synthesis_response(prompt)
            return response
            
        except Exception as e:
            log.error(f"Error in synthesis: {e}")
            return f"I encountered an error while synthesizing the results: {str(e)}\n\nRaw tool results: {tool_results}"
    
    def _get_tool_selection_prompt(self, query: str, available_tools: List[Tool]) -> str:
        """
        Create prompt for tool selection.
        
        Args:
            query: User query
            available_tools: List of available tools
            
        Returns:
            Formatted prompt string
        """
        tools_description = "\n".join([
            f"- Name: {tool.name}\n  Description: {tool.description}\n  Parameters: {json.dumps(tool.inputSchema)}"
            for tool in available_tools
        ])
        
        # Include current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return TOOL_SELECTION_PROMPT.format(
            tools_description=tools_description,
            query=query,
            current_datetime=current_datetime
        )
    
    def _get_task_analysis_prompt(self, query: str, available_tools: List[Tool], detailed_server_info: Dict[str, str] = None, chat_history_formatted: str = "") -> str:
        """
        Create prompt for agentic task analysis.
        
        Args:
            query: User query
            available_tools: List of available tools
            detailed_server_info: Optional dictionary of server names to descriptions
            chat_history_formatted: Formatted chat history string
            
        Returns:
            Formatted prompt string
        """
        tools_description = "\n".join([
            f"- Name: {tool.name}\n  Description: {tool.description}\n  Parameters: {json.dumps(tool.inputSchema)}"
            for tool in available_tools
        ])
        
        # Include server descriptions if available
        server_info = ""
        if detailed_server_info:
            server_info = "\n\n## INTEGRATION DESCRIPTIONS:\n"
            server_info += "\n".join([
                f"- {server_name}: {description}"
                for server_name, description in detailed_server_info.items()
            ])
        
        # Include current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return TASK_ANALYSIS_PROMPT.format(
            chat_history_formatted=chat_history_formatted,
            query=query,
            tools_description=tools_description + server_info,
            current_datetime=current_datetime
        )
    
    def _get_synthesis_prompt(self, query: str, thinking: str, tool_results: str, chat_history_formatted: str = "") -> str:
        """
        Create prompt for synthesizing results.
        
        Args:
            query: Original user query
            thinking: Initial agent thinking
            tool_results: Results from tool executions
            chat_history_formatted: Formatted chat history string
            
        Returns:
            Formatted prompt string
        """
        # Include current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return SYNTHESIS_PROMPT.format(
            chat_history_formatted=chat_history_formatted,
            query=query,
            thinking=thinking,
            tool_results=tool_results,
            current_datetime=current_datetime
        )
    
    def _get_openai_response(self, prompt: str) -> str:
        """
        Get response from OpenAI.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that selects the appropriate tool based on user queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
    
    def _get_openai_synthesis_response(self, prompt: str) -> str:
        """
        Get synthesis response from OpenAI.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Response text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that synthesizes tool results to answer user queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content 