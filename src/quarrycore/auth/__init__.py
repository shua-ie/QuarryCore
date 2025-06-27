"""
Production-grade authentication system for QuarryCore.

Provides JWT-based authentication, API key management, and session handling
for enterprise deployments.
"""

from .api_keys import APIKey, APIKeyManager, create_api_key, verify_api_key
from .jwt_manager import JWTManager, TokenData, create_access_token, verify_token
from .middleware import AuthenticationMiddleware, get_current_user, require_auth
from .models import AuthenticationError, User, UserRole

__all__ = [
    "JWTManager",
    "TokenData",
    "create_access_token",
    "verify_token",
    "APIKeyManager",
    "APIKey",
    "create_api_key",
    "verify_api_key",
    "AuthenticationMiddleware",
    "get_current_user",
    "require_auth",
    "User",
    "UserRole",
    "AuthenticationError",
]
