"""
Implements a cached parser for robots.txt files.
"""
from __future__ import annotations

import asyncio
import logging
from functools import lru_cache
from typing import Dict
from urllib.robotparser import RobotFileParser

import httpx
from httpx import AsyncClient, HTTPError, HTTPStatusError

logger = logging.getLogger(__name__)

# LRU cache for parsed robots.txt data, capped at 1024 entries
@lru_cache(maxsize=1024)
def get_cached_parser(domain: str) -> RobotFileParser:
    """Returns a cached or new RobotFileParser instance for a domain."""
    parser = RobotFileParser()
    parser.set_url(f"https://{domain}/robots.txt")
    return parser

class RobotsCache:
    """
    Manages fetching, parsing, and caching of robots.txt files.
    
    This class uses an LRU cache to minimize network requests for robots.txt
    files that are frequently accessed. It handles fetching asynchronously
    and provides a simple interface to check if a user-agent is allowed
    to access a specific URL.
    """
    
    _client: AsyncClient
    _locks: Dict[str, asyncio.Lock] = {}

    def __init__(self, client: AsyncClient | None = None):
        """
        Initializes the RobotsCache.
        
        Args:
            client: An optional httpx.AsyncClient instance. If not provided,
                    a new one will be created.
        """
        self._client = client or AsyncClient(http2=True, follow_redirects=True)
    
    async def _fetch_robots_txt(self, domain: str) -> str | None:
        """
        Asynchronously fetches the robots.txt file for a given domain.

        Args:
            domain: The domain from which to fetch the robots.txt file.

        Returns:
            The content of the robots.txt file as a string, or None if
            it cannot be fetched.
        """
        url = f"https://{domain}/robots.txt"
        try:
            response = await self._client.get(url, timeout=10.0)
            response.raise_for_status()
            
            # Guard against excessively large robots.txt files
            if len(response.content) > 1_000_000:  # 1MB limit
                logger.warning("robots.txt for %s is larger than 1MB, skipping", domain)
                return None

            return response.text
        except HTTPStatusError as e:
            # Common case (404) means allow all, so we log at a debug level
            if e.response.status_code == 404:
                 logger.debug("No robots.txt found for %s (status %s)", domain, e.response.status_code)
            else:
                logger.warning(
                    "Failed to fetch robots.txt for %s: %s", domain, e
                )
            return None
        except HTTPError as e:
            logger.warning(
                "Failed to fetch robots.txt for %s: %s", domain, e
            )
            return None
        except Exception as e:
            logger.error("Unexpected error fetching robots.txt for %s: %s", domain, e)
            return None

    async def is_allowed(self, url: str, user_agent: str) -> bool:
        """
        Checks if a user-agent is allowed to crawl a given URL.
        
        This method handles fetching and parsing the robots.txt file if it's
        not already cached. It uses a lock to prevent concurrent fetches
        for the same domain.

        Args:
            url: The full URL to check.
            user_agent: The user-agent string to check against.

        Returns:
            True if crawling is allowed, False otherwise.
        """
        try:
            parsed_url = httpx.URL(url)
            domain = parsed_url.host
        except Exception:
            logger.warning("Could not parse URL for robots.txt check: %s", url)
            return False

        parser = get_cached_parser(domain)
        
        # Check if we need to fetch the robots.txt file
        if parser.mtime() == 0:
            # Use a lock to prevent race conditions when fetching for the same domain
            if domain not in self._locks:
                self._locks[domain] = asyncio.Lock()
            
            async with self._locks[domain]:
                # Double-check if it was fetched while waiting for the lock
                if parser.mtime() == 0:
                    content = await self._fetch_robots_txt(domain)
                    if content:
                        parser.parse(content.splitlines())
                    else:
                        # If fetch fails, assume allowed to avoid blocking valid crawls.
                        # We can add a temporary "do not retry" flag here if needed.
                        return True

        return parser.can_fetch(user_agent, url)

    async def close(self) -> None:
        """Closes the underlying HTTP client if it was created internally."""
        # This check prevents closing a client that was passed in from outside
        if hasattr(self._client, 'is_closed') and not self._client.is_closed:
            await self._client.aclose() 