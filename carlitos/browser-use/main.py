from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig

import asyncio
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

# Get environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Get the default Chrome user data directory for macOS
chrome_user_data_dir = str(Path.home() / "Library/Application Support/Google/Chrome")

# IMPORTANT: Close all Chrome browser windows before running this script!

async def main():
    # Configure the browser to use the persistent user data directory
    browser_config = BrowserConfig(
        user_data_dir=chrome_user_data_dir,
        headless=False,
        launch_args=[
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu',
            '--no-first-run',
            '--no-default-browser-check',
            '--disable-extensions',
        ]
        # Note: Check browser-use docs if 'launch_args' is the correct parameter name
        # inside BrowserConfig. If not, adjust accordingly.
    )
    browser = Browser(config=browser_config)

    # Initialize the agent, passing the pre-configured browser instance
    agent = Agent(
        task="""Go to Digital Ocean (cloud.digitalocean.com), find the most basic droplet configuration and tell me the price per month.""",
        llm=ChatOpenAI(model=OPENAI_MODEL),
        browser=browser # Pass the configured browser instance
    )

    # Run the agent
    try:
        await agent.run()
    finally:
        # Manually close the browser since we provided it to the Agent
        await browser.close()

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
