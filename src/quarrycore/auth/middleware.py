"""
Authentication middleware for QuarryCore FastAPI integration.

Provides JWT and API key authentication with role-based access control.
"""
from __future__ import annotations

import re
from typing import Optional, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.api_key import APIKeyHeader

from .api_keys import verify_api_key as verify_api_key_func
from .jwt_manager import verify_token as verify_token_func
from .models import InvalidAPIKeyError, InvalidTokenError, User


# Security schemes
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthenticationMiddleware:
    """
    Middleware for handling authentication in FastAPI applications.
    
    Supports both JWT bearer tokens and API keys with automatic
    detection and validation.
    """
    
    def __init__(self, exclude_paths: Optional[list[str]] = None):
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        self.exclude_patterns = [re.compile(path) for path in self.exclude_paths]
    
    async def __call__(self, request: Request, call_next):
        """Process authentication for incoming requests."""
        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        # Try to authenticate the request
        try:
            user = await self._authenticate_request(request)
            # Add user to request state
            request.state.user = user
        except (InvalidTokenError, InvalidAPIKeyError) as e:
            return HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return await call_next(request)
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path should be excluded from authentication."""
        return any(pattern.match(path) for pattern in self.exclude_patterns)
    
    async def _authenticate_request(self, request: Request) -> User:
        """Authenticate request using JWT or API key."""
        # Check for API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._authenticate_api_key(api_key, request)
        
        # Check for bearer token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            return await self._authenticate_jwt(token)
        
        raise InvalidTokenError("No authentication credentials provided")
    
    async def _authenticate_jwt(self, token: str) -> User:
        """Authenticate using JWT token."""
        token_data = verify_token_func(token)
        return token_data.to_user()
    
    async def _authenticate_api_key(self, api_key: str, request: Request) -> User:
        """Authenticate using API key."""
        # Get client IP for access control
        client_ip = request.client.host if request.client else None
        
        api_key_obj = verify_api_key_func(api_key, client_ip)
        return api_key_obj.to_user()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
) -> User:
    """
    Dependency for getting the current authenticated user.
    
    Supports both JWT bearer tokens and API keys.
    """
    # Try API key first
    if api_key:
        try:
            # Note: IP checking removed for now - would need Request dependency
            api_key_obj = verify_api_key_func(api_key, None)
            return api_key_obj.to_user()
        except InvalidAPIKeyError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
            )
    
    # Try JWT token
    if credentials and credentials.credentials:
        try:
            token_data = verify_token_func(credentials.credentials)
            return token_data.to_user()
        except InvalidTokenError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # No credentials provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_auth(roles: Optional[list[str]] = None):
    """
    Decorator for requiring authentication and optional role-based access.
    
    Usage:
        @app.get("/admin")
        @require_auth(roles=["admin"])
        async def admin_endpoint(user: User = Depends(get_current_user)):
            return {"message": f"Hello admin {user.username}"}
    """
    def decorator(func):
        async def wrapper(*args, user: User = Depends(get_current_user), **kwargs):
            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is deactivated"
                )
            
            # Check roles if specified
            if roles:
                user_roles = {role.value for role in user.roles}
                if not any(role in user_roles for role in roles):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Required roles: {roles}"
                    )
            
            return await func(*args, user=user, **kwargs)
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator 