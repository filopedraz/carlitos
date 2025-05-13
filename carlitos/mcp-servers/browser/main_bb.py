import asyncio
import os
from typing import Optional

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContext,
    BrowserContextConfig,
    BrowserSession,
)
from browserbase import Browserbase
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from playwright.async_api import BrowserContext as PlaywrightContext
from playwright.async_api import Page


class ExtendedBrowserSession(BrowserSession):
    """Extended version of BrowserSession that includes current_page"""

    def __init__(
        self,
        context: PlaywrightContext,
        cached_state: Optional[dict] = None,
        current_page: Optional[Page] = None,
    ):
        super().__init__(context=context, cached_state=cached_state)
        self.current_page = current_page


class UseBrowserbaseContext(BrowserContext):
    async def _initialize_session(self) -> ExtendedBrowserSession:
        """Initialize a browser session using existing Browserbase page.

        Returns:
            ExtendedBrowserSession: The initialized browser session with current page.
        """
        playwright_browser = await self.browser.get_playwright_browser()
        context = await self._create_context(playwright_browser)
        self._add_new_page_listener(context)

        self.session = ExtendedBrowserSession(
            context=context,
            cached_state=None,
        )

        # Get existing page or create new one
        self.session.current_page = (
            context.pages[0] if context.pages else await context.new_page()
        )

        # Initialize session state
        self.session.cached_state = await self._update_state()

        return self.session


async def setup_browser(
    context_id: str = None,
) -> tuple[Browser, UseBrowserbaseContext, str]:
    """Set up browser and context configurations.

    Args:
        context_id: Optional Browserbase context ID to reuse

    Returns:
        tuple[Browser, UseBrowserbaseContext, str]: Configured browser, context, and session URL
    """
    bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])

    # Use the provided context ID if available
    browser_settings = {}
    if context_id:
        browser_settings["context"] = {"id": context_id, "persist": True}

    bb_session = bb.sessions.create(
        project_id=os.environ["BROWSERBASE_PROJECT_ID"],
        browser_settings=browser_settings,
    )

    # Generate session URL for Live View
    session_url = f"https://browserbase.com/sessions/{bb_session.id}"

    browser = Browser(config=BrowserConfig(cdp_url=bb_session.connect_url))
    context = UseBrowserbaseContext(
        browser,
        BrowserContextConfig(
            wait_for_network_idle_page_load_time=10.0,
            highlight_elements=True,
        ),
    )

    return browser, context, session_url


async def setup_agent(browser: Browser, context: UseBrowserbaseContext) -> Agent:
    """Set up the browser automation agent.

    Args:
        browser: Configured browser instance
        context: Browser context for the agent

    Returns:
        Agent: Configured automation agent
    """
    llm = ChatAnthropic(
        model_name="claude-3-5-sonnet-20240620",
        temperature=0.0,
        timeout=100,
    )

    task = "go to https://plausible.joandko.io, select joandko.io website card, and tell me how many unique visitors in the last 30 days."
    task = """
    go on linkedin and put like to the last post of Nicolas Espinoza.
    """

    return Agent(
        task=task,
        llm=llm,
        browser=browser,
        browser_context=context,
    )


def wait_for_user_login():
    """Wait for user to manually login and press Enter to continue."""
    print("Please login manually in the browser session.")
    input("Press Enter when you've completed the login process...")
    print("Continuing with automation...")


async def main():
    load_dotenv()

    # Use the existing context ID
    context_id = "000d117d-8cfa-4e97-8473-d1e41082a8dd"

    browser, context, session_url = await setup_browser(context_id)
    await context.get_session()

    # Display the session URL and wait for manual login
    print(
        f"Browser session created. Open this URL to access the session: {session_url}"
    )
    wait_for_user_login()

    try:
        agent = await setup_agent(browser, context)
        await agent.run()
    finally:
        # Simplified cleanup - just close the browser
        # This will automatically close all contexts and pages
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
