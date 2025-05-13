import logging
import os
import signal
import sys
import time
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict

import aiohttp
from fastmcp import FastMCP
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pypi_version_server")

# Global variables for reload functionality
shutdown_event = Event()
observer = None


class FileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".py"):
            logger.info(f"Python file changed: {event.src_path}. Triggering reload...")
            # Set shutdown event to trigger restart
            shutdown_event.set()
            # Signal the main process
            os.kill(os.getpid(), signal.SIGTERM)


def start_file_watcher(path="."):
    global observer
    event_handler = FileChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    logger.info(f"File watcher started for directory: {path}")


class PypiVersionChecker:
    def __init__(self):
        """Initialize the PyPI version checker"""
        logger.info("Initializing PypiVersionChecker")

    async def get_latest_version(self, package_name: str) -> Dict[str, Any]:
        """
        Get the latest version of a Python package from PyPI

        Args:
            package_name: The name of the Python package
        """
        logger.info(
            f"PypiVersionChecker.get_latest_version called for package: '{package_name}'"
        )
        try:
            # Use PyPI JSON API to get package info
            url = f"https://pypi.org/pypi/{package_name}/json"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error fetching package info: HTTP {response.status}"
                        )
                        logger.error(error_msg)
                        return {"success": False, "error": error_msg}

                    data = await response.json()

                    # Extract the latest version
                    latest_version = data.get("info", {}).get("version")

                    if not latest_version:
                        error_msg = f"No version information found for {package_name}"
                        logger.error(error_msg)
                        return {"success": False, "error": error_msg}

                    response_data = {
                        "success": True,
                        "package": package_name,
                        "latest_version": latest_version,
                        "release_url": f"https://pypi.org/project/{package_name}/{latest_version}/",
                    }

                    # Add additional release info if available
                    if "releases" in data and latest_version in data["releases"]:
                        release_info = (
                            data["releases"][latest_version][0]
                            if data["releases"][latest_version]
                            else {}
                        )
                        response_data["upload_time"] = release_info.get(
                            "upload_time", ""
                        )
                        response_data["python_version"] = release_info.get(
                            "python_version", ""
                        )

                    logger.info(
                        f"PypiVersionChecker.get_latest_version returning: {response_data}"
                    )
                    return response_data
        except Exception as e:
            logger.error(
                f"Error in PypiVersionChecker.get_latest_version: {str(e)}",
                exc_info=True,
            )
            response_data = {"success": False, "error": str(e)}
            logger.info(
                f"PypiVersionChecker.get_latest_version returning error response: {response_data}"
            )
            return response_data

    async def get_all_versions(self, package_name: str) -> Dict[str, Any]:
        """
        Get all available versions of a Python package from PyPI

        Args:
            package_name: The name of the Python package
        """
        logger.info(
            f"PypiVersionChecker.get_all_versions called for package: '{package_name}'"
        )
        try:
            # Use PyPI JSON API to get package info
            url = f"https://pypi.org/pypi/{package_name}/json"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Error fetching package info: HTTP {response.status}"
                        )
                        logger.error(error_msg)
                        return {"success": False, "error": error_msg}

                    data = await response.json()

                    # Extract all versions
                    if "releases" not in data:
                        error_msg = f"No release information found for {package_name}"
                        logger.error(error_msg)
                        return {"success": False, "error": error_msg}

                    # Get all version numbers
                    versions = list(data["releases"].keys())

                    # Sort versions in descending order (latest first)
                    # This is a simple sort that might not handle semantic versioning correctly
                    versions.sort(reverse=True)

                    response_data = {
                        "success": True,
                        "package": package_name,
                        "latest_version": versions[0] if versions else None,
                        "all_versions": versions,
                        "total_versions": len(versions),
                    }

                    logger.info(
                        f"PypiVersionChecker.get_all_versions returning: {response_data}"
                    )
                    return response_data
        except Exception as e:
            logger.error(
                f"Error in PypiVersionChecker.get_all_versions: {str(e)}", exc_info=True
            )
            response_data = {"success": False, "error": str(e)}
            logger.info(
                f"PypiVersionChecker.get_all_versions returning error response: {response_data}"
            )
            return response_data

    async def compare_versions(
        self, package_name: str, current_version: str
    ) -> Dict[str, Any]:
        """
        Compare current version with the latest version

        Args:
            package_name: The name of the Python package
            current_version: The current version to compare
        """
        logger.info(
            f"PypiVersionChecker.compare_versions called for package: '{package_name}', current_version: '{current_version}'"
        )
        try:
            # Get the latest version
            latest_version_result = await self.get_latest_version(package_name)

            if not latest_version_result.get("success", False):
                return latest_version_result

            latest_version = latest_version_result["latest_version"]

            # Simple version comparison (doesn't handle complex semantic versioning)
            is_latest = current_version == latest_version

            response_data = {
                "success": True,
                "package": package_name,
                "current_version": current_version,
                "latest_version": latest_version,
                "is_latest": is_latest,
                "needs_update": not is_latest,
            }

            logger.info(
                f"PypiVersionChecker.compare_versions returning: {response_data}"
            )
            return response_data
        except Exception as e:
            logger.error(
                f"Error in PypiVersionChecker.compare_versions: {str(e)}", exc_info=True
            )
            response_data = {"success": False, "error": str(e)}
            logger.info(
                f"PypiVersionChecker.compare_versions returning error response: {response_data}"
            )
            return response_data


# Initialize PyPI version checker
version_checker = PypiVersionChecker()

# Create FastMCP instance
mcp = FastMCP("PyPI Version Checker")


@mcp.tool()
async def get_latest_version(package_name: str) -> Dict[str, Any]:
    """
    Get the latest version of a Python package from PyPI

    Args:
        package_name: The name of the Python package
    """
    logger.info(f"Tool 'get_latest_version' called for package: '{package_name}'")
    result = await version_checker.get_latest_version(package_name)
    logger.info(f"Tool 'get_latest_version' received from checker: {result}")
    return result


@mcp.tool()
async def get_all_versions(package_name: str) -> Dict[str, Any]:
    """
    Get all available versions of a Python package from PyPI

    Args:
        package_name: The name of the Python package
    """
    logger.info(f"Tool 'get_all_versions' called for package: '{package_name}'")
    result = await version_checker.get_all_versions(package_name)
    logger.info(f"Tool 'get_all_versions' received from checker: {result}")
    return result


@mcp.tool()
async def compare_versions(package_name: str, current_version: str) -> Dict[str, Any]:
    """
    Compare current version with the latest version

    Args:
        package_name: The name of the Python package
        current_version: The current version to compare
    """
    logger.info(
        f"Tool 'compare_versions' called for package: '{package_name}', current_version: '{current_version}'"
    )
    result = await version_checker.compare_versions(package_name, current_version)
    logger.info(f"Tool 'compare_versions' received from checker: {result}")
    return result


def run_server_with_reload():
    global observer

    # Set up signal handling for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while True:
        # Reset shutdown event for new server instance
        shutdown_event.clear()

        # Start the file watcher
        start_file_watcher(Path(__file__).parent)

        try:
            logger.info("Starting MCP server...")
            # Run in a separate thread so we can check for shutdown events
            server_thread = Thread(
                target=lambda: mcp.run(transport="sse", host="0.0.0.0", port=8000)
            )
            server_thread.daemon = True
            server_thread.start()

            # Wait for shutdown event
            while not shutdown_event.is_set():
                time.sleep(0.5)

            # If we get here, we need to restart
            if observer:
                logger.info("Stopping file watcher...")
                observer.stop()
                observer.join()

            # If shutdown was triggered by file change, restart
            if shutdown_event.is_set():
                logger.info("Restarting process...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
                break

        except KeyboardInterrupt:
            logger.info("Server stopped by keyboard interrupt")
            break
        except Exception as e:
            logger.error(f"Error running server: {str(e)}", exc_info=True)
            break
        finally:
            if observer:
                observer.stop()
                observer.join()


if __name__ == "__main__":
    run_server_with_reload()
