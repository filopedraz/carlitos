#!/usr/bin/env python3
import asyncio
import sys
import signal
import json
import logging
import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.syntax import Syntax

from carlitos.mega_agent import MegaAgent
from carlitos.config import DEBUG, DEFAULT_CONFIG

logger = logging.getLogger("carlitos")

console = Console()

app = typer.Typer(help="Carlitos - An agentic MCP client")


# Create a null handler that discards all logs
class NullHandler(logging.Handler):
    def emit(self, record):
        pass


async def chat_loop(agent, debug=False):
    """
    Async chat loop with the agent.
    
    Args:
        agent: The MegaAgent instance
        debug: Whether to show raw tool responses
    """
    # First discover tools
    await discover_tools(agent)
    
    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            # Process user query with progress indicator
            with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
                try:
                    # Process the chat message
                    response = await agent.chat(user_input)
                    
                    # Debug output if enabled
                    if debug and hasattr(agent, "_last_tool_results"):
                        display_debug_info(agent)
                    
                    # Display which agent was used
                    agent_info = ""
                    if hasattr(agent, "_current_agent_type") and agent._current_agent_type:
                        if agent._current_agent_type.startswith("coordinator"):
                            agent_info = f" [blue](via multi-agent coordination)[/blue]"
                        else:
                            agent_info = f" [blue](via {agent._current_agent_type} agent)[/blue]"
                    
                    console.print(f"[bold green]Carlitos:{agent_info}[/bold green] {response}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    console.print("[yellow]Please try again with a different query.[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[bold green]Carlitos:[/bold green] Goodbye!")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] An unexpected error occurred: {str(e)}")
            console.print("[yellow]Chat loop will continue. You can still enter messages.[/yellow]")


def display_debug_info(agent):
    """Display debug information about tool execution."""
    console.print("[bold yellow]Debug - Raw Tool Results:[/bold yellow]")
    
    # Try to parse as JSON for nicer formatting
    try:
        if isinstance(agent._last_tool_results, str) and agent._last_tool_results.startswith("{"):
            result_json = json.loads(agent._last_tool_results)
            formatted_json = json.dumps(result_json, indent=2)
            console.print(Syntax(formatted_json, "json", theme="monokai", word_wrap=True))
        elif isinstance(agent._last_tool_results, str) and agent._last_tool_results.startswith("EXECUTION SUMMARY"):
            # Split into sections for better readability
            parts = agent._last_tool_results.split("\n\nRESULTS:\n")
            if len(parts) == 2:
                console.print(Panel(parts[0], title="Execution Summary", border_style="yellow"))
                console.print(Panel(parts[1][:1000] + "..." if len(parts[1]) > 1000 else parts[1], 
                                   title="Results", border_style="green"))
            else:
                console.print(agent._last_tool_results)
        else:
            console.print(agent._last_tool_results)
    except json.JSONDecodeError:
        console.print(agent._last_tool_results)
    
    # If using a sub-agent, try to get its tool results too
    if hasattr(agent, "_current_agent_type") and agent._current_agent_type:
        if agent._current_agent_type in agent.sub_agents:
            sub_agent = agent.sub_agents[agent._current_agent_type]
            if hasattr(sub_agent, "_last_tool_results") and sub_agent._last_tool_results:
                console.print(f"[bold magenta]Debug - {agent._current_agent_type} Sub-Agent Tool Results:[/bold magenta]")
                try:
                    if isinstance(sub_agent._last_tool_results, str) and sub_agent._last_tool_results.startswith("{"):
                        result_json = json.loads(sub_agent._last_tool_results)
                        formatted_json = json.dumps(result_json, indent=2)
                        console.print(Syntax(formatted_json, "json", theme="monokai", word_wrap=True))
                    else:
                        console.print(sub_agent._last_tool_results)
                except json.JSONDecodeError:
                    console.print(sub_agent._last_tool_results)


async def discover_tools(agent):
    """
    Discover tools for the agent.
    
    Args:
        agent: The MegaAgent instance
    """
    with console.status("[cyan]Discovering tools...[/cyan]", spinner="dots"):
        try:
            # Initialize sub-agents based on server descriptions
            if not agent.sub_agents:
                await agent.initialize_sub_agents()
                console.print(f"[bold cyan]Initialized {len(agent.sub_agents)} specialized sub-agents[/bold cyan]")
                
            # Defer tool discovery until necessary
            console.print("[bold cyan]Configured for on-demand tool discovery[/bold cyan]")
                
        except Exception as e:
            logger.error(f"Error initializing agent: {e}", exc_info=True)
            console.print(f"[bold red]Error initializing agent:[/bold red] {str(e)}")


def configure_logging(debug=False):
    """
    Configure logging based on debug flag.
    
    Args:
        debug: Whether to enable debug mode
    """
    root_logger = logging.getLogger()
    
    # Remove all handlers from root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    if debug:
        # Enable DEBUG logging with rich handler when in debug mode
        handler = logging.getLogger().handlers[0] if logging.getLogger().handlers else None
        if not handler or not isinstance(handler, logging.StreamHandler):
            from rich.logging import RichHandler
            handler = RichHandler(rich_tracebacks=True)
            root_logger.addHandler(handler)
        
        root_logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
        
        # Set carlitos.* loggers to DEBUG level
        for logger_name in logging.root.manager.loggerDict:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)
    else:
        # Add null handler to discard all logs
        null_handler = NullHandler()
        root_logger.addHandler(null_handler)
        
        # Set root logger to CRITICAL to minimize any logs
        root_logger.setLevel(logging.CRITICAL)
        
        # Silence all loggers
        for logger_name in logging.root.manager.loggerDict:
            current_logger = logging.getLogger(logger_name)
            # Remove all handlers
            for handler in current_logger.handlers[:]:
                current_logger.removeHandler(handler)
            current_logger.addHandler(null_handler)
            current_logger.setLevel(logging.CRITICAL)


def run_chat(debug=DEBUG):
    """
    Implementation of the chat functionality.
    
    Args:
        debug: Whether to enable debug mode
    """
    # Configure logging
    configure_logging(debug)
    
    try:
        # Create and initialize MegaAgent
        agent = MegaAgent(DEFAULT_CONFIG)
        
        # Display welcome message
        console.print("[bold green]Carlitos Chat[/bold green]")
        console.print("[dim]Press CTRL+C to exit[/dim]")
        console.print("[bold cyan]Using MegaAgent to route requests to specialized sub-agents[/bold cyan]")
        
        # Set up signal handler for graceful exit
        def handle_sigint(sig, frame):
            console.print("\n[bold green]Carlitos:[/bold green] Goodbye!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_sigint)
        
        # Run the chat loop
        asyncio.run(chat_loop(agent, debug))
        
    except KeyboardInterrupt:
        console.print("\n[bold green]Carlitos:[/bold green] Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def chat(
    debug: bool = False
):
    """
    Start an interactive chat session with Carlitos.
    Exit with CTRL+C.
    """
    run_chat(debug)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with verbose logging")
):
    """
    Start chat mode when no command is provided.
    """
    if ctx.invoked_subcommand is None:
        # Pass the debug flag to chat
        chat(debug=debug)


if __name__ == "__main__":
    app()
