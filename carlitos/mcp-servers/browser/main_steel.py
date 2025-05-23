"""
Steel Browser Use Starter Template
Integrates Steel with browser-use framework to create an AI agent for web interactions.
Requires STEEL_API_KEY & OPENAI_API_KEY in .env file.
"""

import asyncio
import json
import os
from pathlib import Path

from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from steel import Steel

# 1. Initialize environment and clients
load_dotenv()

# Get API keys
STEEL_API_KEY = os.getenv("STEEL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not STEEL_API_KEY or not OPENAI_API_KEY:
    raise ValueError("STEEL_API_KEY and OPENAI_API_KEY must be set in .env file")

# Initialize Steel client
client = Steel(steel_api_key=STEEL_API_KEY)

# File to store session context
CONTEXT_FILE = Path("browser_context.json")


async def create_session(context=None):
    """Create a Steel session with optional context."""
    print("Creating Steel session...")
    session_params = {}

    # Only add sessionContext if we have a valid context object
    if context:
        session_params["session_context"] = context

    session = client.sessions.create(**session_params)

    print(
        f"\033[1;93mSteel Session created!\033[0m\n"
        f"View session at \033[1;37m{session.session_viewer_url}\033[0m\n"
    )
    return session


async def get_browser(session):
    """Create a browser instance connected to the Steel session."""
    cdp_url = f"wss://connect.steel.dev?apiKey={STEEL_API_KEY}&sessionId={session.id}"
    browser = Browser(config=BrowserConfig(cdp_url=cdp_url))
    return browser


async def capture_session_context(session_id):
    """Capture and save the session context."""
    print("Capturing session context...")
    session_context = client.sessions.context(session_id)

    # Convert the SessionContext object to a dictionary for JSON serialization
    context_dict = session_context.to_dict()

    # Save context to file
    with open(CONTEXT_FILE, "w") as f:
        json.dump(context_dict, f)

    print(f"Context saved to {CONTEXT_FILE}")
    return context_dict


async def load_session_context():
    """Load session context from file if it exists."""
    if not CONTEXT_FILE.exists():
        print(f"Context file {CONTEXT_FILE} does not exist")
        return None

    # Check if file is empty
    if CONTEXT_FILE.stat().st_size == 0:
        print(f"Context file {CONTEXT_FILE} is empty")
        return None

    print(f"Loading saved context from {CONTEXT_FILE}")
    try:
        with open(CONTEXT_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error loading context: {e}")
        return None


async def main():
    session = None
    browser = None

    try:
        # Check if we have a saved context
        saved_context = await load_session_context()

        # If no context or user wants to create a new one
        if not saved_context or input("Use saved context? (y/n): ").lower() != "y":
            # Create initial session for manual login
            session = await create_session()
            browser = await get_browser(session)

            print(
                "\033[1;92mA browser session has been opened for manual authentication\033[0m"
            )
            print("1. Go to the Steel session URL shown above")
            print("2. Log in to the website(s) you need authenticated access to")
            print("3. Come back here when you're done")

            input("\nPress Enter once you've completed the login process...")

            # Capture context after manual login
            await capture_session_context(session.id)

            # Close initial session
            await browser.close()
            client.sessions.release(session.id)
            print("Initial session released")

            # Load the newly saved context
            saved_context = await load_session_context()

        # Create a new session with the saved context
        session = await create_session(saved_context)
        browser = await get_browser(session)
        browser_context = BrowserContext(browser=browser)

        # The agent's task
        task = """
        Send a message to Nicolas Espinoza on LinkedIn: Hello, I'm a bot from browser-use. I'm testing the browser context persistence.
        """

        # Create model and agent
        model = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=OPENAI_API_KEY)
        agent = Agent(
            task=task,
            llm=model,
            browser=browser,
            browser_context=browser_context,
        )

        # Run the agent with authenticated session
        await agent.run()

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        if browser:
            await browser.close()
        if session:
            client.sessions.release(session.id)
        print("Done")


if __name__ == "__main__":
    asyncio.run(main())
