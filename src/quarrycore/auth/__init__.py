"""
Production-grade authentication system for QuarryCore.

Provides JWT-based authentication, API key management, and session handling
for enterprise deployments.
"""

from .jwt_manager import JWTManager, TokenData, create_access_token, verify_token
from .api_keys import APIKeyManager, APIKey, create_api_key, verify_api_key
from .middleware import AuthenticationMiddleware, get_current_user, require_auth
from .models import User, UserRole, AuthenticationError

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