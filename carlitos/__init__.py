"""
Carlitos - An agentic MCP client that orchestrates tool selection and execution.
"""

__version__ = "0.1.0"

from carlitos.agent import AgenticMCPAgent
from carlitos.config import CarlitosConfig, load_config
from carlitos.llm import LLMCoreAgent
from carlitos.prompt import TASK_ANALYSIS_PROMPT, SYNTHESIS_PROMPT, TOOL_SELECTION_PROMPT