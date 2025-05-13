import asyncio
import os

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from langchain_openai import ChatOpenAI

# Define path for cookies
cookies_path = os.path.join(os.path.dirname(__file__), "browser_cookies.json")

# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        # Specify the path to your Chrome executable
        browser_binary_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS path
        # For Windows, typically: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
        # For Linux, typically: '/usr/bin/google-chrome'
    )
)

# Create a browser context config with persistent cookies
context_config = BrowserContextConfig(
    cookies_file=cookies_path,  # This will save/load cookies from this file
)


async def main():
    # Async creation of the browser context
    context = await browser.new_context(config=context_config)

    # Create agent with the context
    agent = Agent(
        task="Send a message to Nicolas Espinoza on LinkedIn: Hello, I'm a bot from browser-use. I'm testing the browser context persistence.",
        llm=ChatOpenAI(model="gpt-4o"),
        browser_context=context,  # Use the context instead of browser directly
    )

    await agent.run()

    input("Press Enter to close the browser...")
    await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
