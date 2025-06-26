"""
JWT token management for QuarryCore authentication.

Provides secure token generation, validation, and refresh capabilities
with configurable expiry and rotation mechanisms.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Set
from uuid import UUID, uuid4

import jwt
from pydantic import BaseModel, Field

from .models import InvalidTokenError, TokenData, User, UserRole


class JWTSettings(BaseModel):
    """JWT configuration settings."""
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    issuer: str = "quarrycore"
    audience: str = "quarrycore-api"


class JWTManager:
    """
    Manages JWT token creation, validation, and refresh operations.
    
    Features:
    - Secure token generation with configurable expiry
    - Token validation with comprehensive checks
    - Refresh token rotation for enhanced security
    - Token revocation support via JTI tracking
    """
    
    def __init__(self, settings: Optional[JWTSettings] = None):
        self.settings = settings or JWTSettings()
        self._revoked_tokens: Set[str] = set()  # In production, use Redis
    
    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new access token for a user."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                minutes=self.settings.access_token_expire_minutes
            )
        
        token_data = {
            "sub": str(user.user_id),  # Subject
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": self.settings.issuer,
            "aud": self.settings.audience,
            "jti": str(uuid4()),  # JWT ID for revocation
        }
        
        return jwt.encode(
            token_data,
            self.settings.secret_key,
            algorithm=self.settings.algorithm
        )
    
    def create_refresh_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new refresh token for a user."""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(
                days=self.settings.refresh_token_expire_days
            )
        
        token_data = {
            "sub": str(user.user_id),
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": self.settings.issuer,
            "aud": self.settings.audience,
            "jti": str(uuid4()),
        }
        
        return jwt.encode(
            token_data,
            self.settings.secret_key,
            algorithm=self.settings.algorithm
        )
    
    def verify_token(self, token: str, token_type: str = "access") -> TokenData:
        """
        Verify and decode a JWT token.
        
        Args:
            token: The JWT token to verify
            token_type: Type of token ("access" or "refresh")
            
        Returns:
            TokenData object with decoded token information
            
        Raises:
            InvalidTokenError: If token is invalid, expired, or revoked
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.algorithm],
                audience=self.settings.audience,
                issuer=self.settings.issuer,
            )
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                raise InvalidTokenError("Token has been revoked")
            
            # Verify token type
            if token_type == "refresh" and payload.get("type") != "refresh":
                raise InvalidTokenError("Invalid token type")
            
            # Extract token data
            return TokenData(
                user_id=payload["sub"],
                username=payload.get("username", ""),
                roles=set(payload.get("roles", [])),
                exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
                jti=jti,
            )
            
        except jwt.ExpiredSignatureError:
            raise InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise InvalidTokenError(f"Invalid token: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """
        Generate new access and refresh tokens using a valid refresh token.
        
        Returns:
            Tuple of (access_token, new_refresh_token)
        """
        # Verify refresh token
        token_data = self.verify_token(refresh_token, token_type="refresh")
        
        # Revoke old refresh token
        if token_data.jti:
            self.revoke_token(token_data.jti)
        
        # Create user from token data
        user = User(
            user_id=UUID(token_data.user_id),
            username=token_data.username,
            email="",  # Not stored in token
            roles={UserRole(role) for role in token_data.roles}
        )
        
        # Generate new tokens
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)
        
        return access_token, refresh_token
    
    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI."""
        self._revoked_tokens.add(jti)
    
    def is_token_revoked(self, jti: str) -> bool:
        """Check if a token is revoked."""
        return jti in self._revoked_tokens


# Convenience functions for the module interface
_default_manager = JWTManager()

def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create an access token using the default manager."""
    return _default_manager.create_access_token(user, expires_delta)

def verify_token(token: str, token_type: str = "access") -> TokenData:
    """Verify a token using the default manager."""
    return _default_manager.verify_token(token, token_type) 