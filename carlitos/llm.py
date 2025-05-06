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
        Analyze a user query and decide what tools to use.
        
        Args:
            query: User query
            available_tools: List of available tools
            chat_history: Optional chat history
            detailed_server_info: Optional dictionary of server names to descriptions
            
        Returns:
            Tuple of (thinking result, list of needed tools)
        """
        # Format tools for the prompt
        tools_formatted = self._format_tools(available_tools)
        
        # Format chat history if provided
        chat_history_formatted = ""
        if chat_history:
            chat_history_formatted = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in chat_history[-5:]  # Last 5 messages
            ])
        
        # Get the prompt
        prompt = self._get_task_analysis_prompt(
            query, 
            available_tools, 
            detailed_server_info, 
            chat_history_formatted
        )
        
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
    
    async def synthesize_results(self, query: str, thinking: str, tool_results: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Synthesize tool results to provide an answer to the user.
        
        Args:
            query: User query
            thinking: The thinking process that led to tool selection
            tool_results: Results from executing the tools
            chat_history: Optional chat history
            
        Returns:
            Synthesized response
        """
        # Format chat history if provided
        chat_history_formatted = ""
        if chat_history:
            chat_history_formatted = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in chat_history[-5:]  # Last 5 messages
            ])
            
        # Get current date and time for context
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get the synthesis prompt
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            thinking=thinking,
            tool_results=tool_results,
            chat_history_formatted=chat_history_formatted,
            current_datetime=current_datetime
        )
        
        log.debug(f"Sending synthesis prompt to OpenAI with {len(tool_results)} chars of tool results")
        
        try:
            # Get response from OpenAI
            response = self._get_openai_synthesis_response(prompt)
            
            return response
            
        except Exception as e:
            log.error(f"Error in result synthesis: {e}")
            # Fallback response
            return f"I've processed your request, but encountered an error when formatting the results: {str(e)}. Here's the raw data: {tool_results}"
    
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
                {"role": "system", "content": "You are a helpful AI assistant that analyzes tasks and selects appropriate tools to execute them."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
        )
        
        return response.choices[0].message.content
    
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
            Prompt string
        """
        # Format available tools
        tools_description = self._format_tools(available_tools)
        
        # Add detailed server info if available
        if detailed_server_info:
            tools_description += "\n\nAdditional Server Information:\n"
            for server_name, description in detailed_server_info.items():
                tools_description += f"- {server_name}: {description}\n"
                
        # Get current date and time for context
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Return the prompt using the template from prompt.py
        return TASK_ANALYSIS_PROMPT.format(
            query=query,
            tools_description=tools_description,
            chat_history_formatted=chat_history_formatted,
            current_datetime=current_datetime
        )
    
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
    
    def _format_tools(self, available_tools: List[Tool]) -> str:
        """
        Format available tools for inclusion in prompts.
        
        Args:
            available_tools: List of available tools
            
        Returns:
            Formatted tools string
        """
        formatted_tools = []
        
        for tool in available_tools:
            tool_info = f"Tool: {tool.name}\n"
            tool_info += f"  Description: {tool.description}\n"
            
            # Format parameters
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                params = []
                try:
                    if isinstance(tool.inputSchema, dict) and "properties" in tool.inputSchema:
                        properties = tool.inputSchema.get("properties", {})
                        required = tool.inputSchema.get("required", [])
                        
                        for param_name, param_info in properties.items():
                            param_type = param_info.get("type", "unknown")
                            description = param_info.get("description", "")
                            is_required = param_name in required
                            
                            param_str = f"    - {param_name} (Type: {param_type}"
                            if is_required:
                                param_str += ", Required"
                            param_str += f"): {description}"
                            params.append(param_str)
                    else:
                        # Simpler format if schema doesn't match expected structure
                        params.append(f"    Parameters: {json.dumps(tool.inputSchema)}")
                except Exception as e:
                    # Fallback if parsing fails
                    params.append(f"    Parameters: {str(tool.inputSchema)}")
                
                if params:
                    tool_info += "  Parameters:\n" + "\n".join(params)
            
            formatted_tools.append(tool_info)
        
        return "\n".join(formatted_tools) 