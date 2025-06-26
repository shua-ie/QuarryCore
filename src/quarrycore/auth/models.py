"""
Authentication models and data types for QuarryCore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Set
from uuid import UUID, uuid4


class UserRole(Enum):
    """User role enumeration for authorization."""
    ADMIN = "admin"
    USER = "user"
    API_CLIENT = "api_client"
    READ_ONLY = "read_only"


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass


class InvalidTokenError(AuthenticationError):
    """Raised when a token is invalid or expired."""
    pass


class InvalidAPIKeyError(AuthenticationError):
    """Raised when an API key is invalid or revoked."""
    pass


class InsufficientPermissionsError(AuthenticationError):
    """Raised when a user lacks required permissions."""
    pass


@dataclass
class User:
    """User model for authentication and authorization."""
    user_id: UUID
    username: str
    email: str
    roles: Set[UserRole]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has a specific role."""
        return role in self.roles
    
    def has_any_role(self, *roles: UserRole) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in self.roles for role in roles)
    
    def is_admin(self) -> bool:
        """Check if user has admin privileges."""
        return UserRole.ADMIN in self.roles


@dataclass
class TokenData:
    """Data stored in JWT tokens."""
    user_id: str
    username: str
    roles: Set[str]
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    jti: Optional[str] = None  # JWT ID for token revocation
    
    def to_user(self) -> User:
        """Convert token data to User object."""
        return User(
            user_id=UUID(self.user_id),
            username=self.username,
            email="",  # Email not stored in token
            roles={UserRole(role) for role in self.roles}
        ) 