"""
User Agent Rotation with Realistic Browser Fingerprints

Provides a pool of realistic user agents for ethical web scraping
while avoiding detection and respecting server resources.
"""

from __future__ import annotations

import random
from typing import Dict, List


class UserAgentRotator:
    """
    Manages rotation of realistic user agent strings.

    Features:
    - Realistic browser fingerprints
    - Weighted selection based on market share
    - Mobile and desktop variants
    - Regular updates to stay current
    """

    def __init__(self, include_mobile: bool = True, include_bots: bool = True):
        self.include_mobile = include_mobile
        self.include_bots = include_bots

        # Desktop browsers (most common first)
        self.desktop_agents = [
            # Chrome (most popular)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            # Firefox
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            # Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
            # Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        ]

        # Mobile browsers
        self.mobile_agents = [
            # Mobile Chrome
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Android 14; Mobile; rv:121.0) Gecko/121.0 Firefox/121.0",
            "Mozilla/5.0 (Linux; Android 14; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            # iPad
            "Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1",
        ]

        # Bot user agents (for transparency)
        self.bot_agents = [
            "QuarryCore/1.0 (+https://github.com/quarrycore/quarrycore) AI Training Data Miner",
            "Mozilla/5.0 (compatible; QuarryCore/1.0; +https://github.com/quarrycore/quarrycore)",
            "QuarryCore-Bot/1.0 (Ethical AI Training Data Collection)",
        ]

        # Weights for different categories (desktop favored for most content)
        self.category_weights = {
            "desktop": 0.7,
            "mobile": 0.2 if include_mobile else 0.0,
            "bot": 0.1 if include_bots else 0.0,
        }

        # Normalize weights
        total_weight = sum(self.category_weights.values())
        if total_weight > 0:
            self.category_weights = {k: v / total_weight for k, v in self.category_weights.items()}

    def get_random_user_agent(self) -> str:
        """Get a random user agent based on weighted selection."""
        # Choose category based on weights
        categories = list(self.category_weights.keys())
        weights = list(self.category_weights.values())

        if not categories or sum(weights) == 0:
            # Fallback to default bot agent
            return self.bot_agents[0]

        category = random.choices(categories, weights=weights)[0]

        # Select agent from chosen category
        if category == "desktop":
            return random.choice(self.desktop_agents)
        elif category == "mobile" and self.mobile_agents:
            return random.choice(self.mobile_agents)
        elif category == "bot" and self.bot_agents:
            return random.choice(self.bot_agents)
        else:
            # Fallback
            return random.choice(self.desktop_agents)

    def get_desktop_user_agent(self) -> str:
        """Get a random desktop user agent."""
        return random.choice(self.desktop_agents)

    def get_mobile_user_agent(self) -> str:
        """Get a random mobile user agent."""
        if self.mobile_agents:
            return random.choice(self.mobile_agents)
        return self.get_desktop_user_agent()

    def get_bot_user_agent(self) -> str:
        """Get a bot user agent for transparent crawling."""
        if self.bot_agents:
            return random.choice(self.bot_agents)
        return self.bot_agents[0]

    def get_all_agents(self) -> List[str]:
        """Get all available user agents."""
        agents = self.desktop_agents.copy()
        if self.include_mobile:
            agents.extend(self.mobile_agents)
        if self.include_bots:
            agents.extend(self.bot_agents)
        return agents

    def get_agents_by_browser(self, browser: str) -> List[str]:
        """Get user agents for specific browser."""
        browser = browser.lower()
        all_agents = self.get_all_agents()

        browser_keywords = {
            "chrome": ["Chrome/", "Chromium/"],
            "firefox": ["Firefox/", "Gecko/"],
            "safari": ["Safari/", "Version/"],
            "edge": ["Edg/", "Edge/"],
            "bot": ["QuarryCore", "Bot", "compatible;"],
        }

        if browser not in browser_keywords:
            return []

        keywords = browser_keywords[browser]
        return [agent for agent in all_agents if any(keyword in agent for keyword in keywords)]

    def update_weights(
        self,
        desktop_weight: float = 0.7,
        mobile_weight: float = 0.2,
        bot_weight: float = 0.1,
    ) -> None:
        """Update category weights for user agent selection."""
        if not self.include_mobile:
            mobile_weight = 0.0
        if not self.include_bots:
            bot_weight = 0.0

        total = desktop_weight + mobile_weight + bot_weight
        if total > 0:
            self.category_weights = {
                "desktop": desktop_weight / total,
                "mobile": mobile_weight / total,
                "bot": bot_weight / total,
            }

    def add_custom_agent(self, user_agent: str, category: str = "desktop") -> None:
        """Add a custom user agent to the pool."""
        if category == "desktop":
            self.desktop_agents.append(user_agent)
        elif category == "mobile":
            self.mobile_agents.append(user_agent)
        elif category == "bot":
            self.bot_agents.append(user_agent)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about available user agents."""
        return {
            "total_agents": len(self.get_all_agents()),
            "desktop_agents": len(self.desktop_agents),
            "mobile_agents": len(self.mobile_agents),
            "bot_agents": len(self.bot_agents),
            "include_mobile": self.include_mobile,
            "include_bots": self.include_bots,
        }
