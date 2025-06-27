"""
QuarryCore - Production-grade AI training data miner.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config
from .container import DependencyContainer
from .pipeline import Pipeline

__all__ = ["__version__", "Config", "DependencyContainer", "Pipeline"]
