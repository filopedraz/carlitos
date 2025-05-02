#!/usr/bin/env python3
import asyncio
import logging
import sys
import signal

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt

from carlitos.mega_agent import MegaAgent
from carlitos.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("carlitos")

console = Console()

app = typer.Typer(help="Carlitos - An agentic MCP client")


def run_chat(
    config_path: str = "./.cursor/mcp.json",
    verbose: bool = False,
    debug: bool = False,
):
    """Implementation of the chat functionality."""
    if verbose or debug:
        log.setLevel(logging.DEBUG)
        # Set all carlitos.* loggers to DEBUG level
        for logger_name in logging.root.manager.loggerDict:
            if logger_name.startswith("carlitos."):
                logging.getLogger(logger_name).setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        log.debug(f"Loading config from {config_path}")
        config = load_config(config_path)
        
        # Create MegaAgent
        agent = MegaAgent(config)
        console.print("[bold green]Carlitos Chat[/bold green]")
        console.print("[dim]Press CTRL+C to exit[/dim]")
        console.print("[bold cyan]Using MegaAgent to route requests to specialized sub-agents[/bold cyan]")
        
        # Set up signal handler for graceful exit
        def handle_sigint(sig, frame):
            console.print("\n[bold green]Carlitos:[/bold green] Goodbye!")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handle_sigint)
        
        # Create and run the asyncio event loop
        asyncio.run(chat_loop(agent, debug))
        
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Config file not found at {config_path}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[bold green]Carlitos:[/bold green] Goodbye!")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if verbose or debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def chat(
    config_path: str = typer.Option(
        "./.cursor/mcp.json", 
        "--config", 
        "-c", 
        help="Path to MCP configuration file"
    ),
    verbose: bool = typer.Option(
        False, 
        "--verbose", 
        "-v", 
        help="Enable verbose output"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug logging level and show raw tool responses"
    ),
):
    """
    Start an interactive chat session with Carlitos.
    Exit with CTRL+C.
    """
    run_chat(config_path, verbose, debug)


async def chat_loop(agent, debug=False):
    """
    Async chat loop with the agent.
    
    Args:
        agent: The MegaAgent instance
        debug: Whether to show raw tool responses
    """
    # First discover tools once to initialize the agent
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
                    
                    # In debug mode, check if there are tool results
                    if debug and hasattr(agent, "_last_tool_results"):
                        console.print("[bold yellow]Debug - Raw Tool Results:[/bold yellow]")
                        console.print(agent._last_tool_results)
                    
                    # Display which agent was used
                    agent_info = ""
                    if hasattr(agent, "_current_agent_type") and agent._current_agent_type:
                        if agent._current_agent_type.startswith("coordinator"):
                            agent_info = f" [blue](via multi-agent coordination)[/blue]"
                        else:
                            agent_info = f" [blue](via {agent._current_agent_type} agent)[/blue]"
                    
                    console.print(f"[bold green]Carlitos:{agent_info}[/bold green] {response}")
                except Exception as e:
                    log.error(f"Error processing message: {e}", exc_info=True)
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    console.print("[yellow]Please try again with a different query.[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[bold green]Carlitos:[/bold green] Goodbye!")
            sys.exit(0)
        except Exception as e:
            log.error(f"Error in chat loop: {e}", exc_info=True)
            console.print(f"[bold red]Error:[/bold red] An unexpected error occurred: {str(e)}")
            console.print("[yellow]Chat loop will continue. You can still enter messages.[/yellow]")


async def discover_tools(agent):
    """
    Discover tools for the agent.
    
    Args:
        agent: The MegaAgent instance
    """
    with console.status("[cyan]Discovering tools...[/cyan]", spinner="dots"):
        try:
            # First initialize sub-agents based on server descriptions
            if not agent.sub_agents:
                await agent.initialize_sub_agents()
                console.print(f"[bold cyan]Initialized {len(agent.sub_agents)} specialized sub-agents[/bold cyan]")
                
            # Only discover all tools when needed (defer until necessary)
            # This avoids token limit issues by not loading all tool details during startup
            console.print("[bold cyan]Configured for on-demand tool discovery[/bold cyan]")
                
        except Exception as e:
            log.error(f"Error initializing agent: {e}", exc_info=True)
            console.print(f"[bold red]Error initializing agent:[/bold red] {str(e)}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    Start chat mode when no command is provided.
    """
    if ctx.invoked_subcommand is None:
        run_chat()


if __name__ == "__main__":
    app()
