"""
API key management for programmatic access to QuarryCore.

Provides secure API key generation, validation, and management
with rate limiting and usage tracking capabilities.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from .models import InvalidAPIKeyError, User, UserRole


@dataclass
class APIKey:
    """API key model with metadata and permissions."""

    key_id: UUID
    key_hash: str  # Hashed version of the actual key
    name: str
    user_id: UUID
    roles: Set[UserRole]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    allowed_ips: List[str] = field(default_factory=list)
    rate_limit_per_hour: int = 1000
    usage_count: int = 0

    def is_expired(self) -> bool:
        """Check if the API key has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_ip_allowed(self, ip: str) -> bool:
        """Check if the given IP is allowed to use this key."""
        if not self.allowed_ips:
            return True  # No IP restrictions
        return ip in self.allowed_ips

    def to_user(self) -> User:
        """Convert API key to User object for authorization."""
        return User(
            user_id=self.user_id,
            username=f"api_key_{self.name}",
            email="",
            roles=self.roles,
            is_active=self.is_active,
        )


class APIKeyManager:
    """
    Manages API key creation, validation, and lifecycle.

    Features:
    - Secure key generation with prefix identification
    - Key hashing for secure storage
    - Usage tracking and rate limiting
    - IP-based access control
    - Key expiration and rotation
    """

    def __init__(self, key_prefix: str = "qc_"):
        self.key_prefix = key_prefix
        self._keys_db: Dict[str, APIKey] = {}  # In production, use database

    def generate_api_key(
        self,
        name: str,
        user_id: UUID,
        roles: Set[UserRole],
        expires_at: Optional[datetime] = None,
        allowed_ips: Optional[List[str]] = None,
        rate_limit_per_hour: int = 1000,
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.

        Returns:
            Tuple of (raw_api_key, api_key_object)
            The raw key is only returned once and should be given to the user.
        """
        # Generate secure random key
        raw_key = f"{self.key_prefix}{secrets.token_urlsafe(32)}"

        # Hash the key for storage
        key_hash = self._hash_key(raw_key)

        # Create API key object
        api_key = APIKey(
            key_id=uuid4(),
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            roles=roles,
            expires_at=expires_at,
            allowed_ips=allowed_ips or [],
            rate_limit_per_hour=rate_limit_per_hour,
        )

        # Store the key
        self._keys_db[key_hash] = api_key

        return raw_key, api_key

    def verify_api_key(self, raw_key: str, request_ip: Optional[str] = None) -> APIKey:
        """
        Verify an API key and return the associated metadata.

        Args:
            raw_key: The raw API key to verify
            request_ip: IP address of the request for IP-based access control

        Returns:
            APIKey object if valid

        Raises:
            InvalidAPIKeyError: If key is invalid, expired, or access denied
        """
        # Check key format
        if not raw_key.startswith(self.key_prefix):
            raise InvalidAPIKeyError("Invalid API key format")

        # Hash the key
        key_hash = self._hash_key(raw_key)

        # Look up the key
        api_key = self._keys_db.get(key_hash)
        if not api_key:
            raise InvalidAPIKeyError("API key not found")

        # Check if key is active
        if not api_key.is_active:
            raise InvalidAPIKeyError("API key has been deactivated")

        # Check expiration
        if api_key.is_expired():
            raise InvalidAPIKeyError("API key has expired")

        # Check IP restrictions
        if request_ip and not api_key.is_ip_allowed(request_ip):
            raise InvalidAPIKeyError(f"Access denied from IP: {request_ip}")

        # Update usage statistics
        api_key.last_used = datetime.utcnow()
        api_key.usage_count += 1

        return api_key

    def revoke_api_key(self, key_id: UUID) -> bool:
        """Revoke an API key by its ID."""
        for api_key in self._keys_db.values():
            if api_key.key_id == key_id:
                api_key.is_active = False
                return True
        return False

    def list_api_keys(self, user_id: UUID) -> List[APIKey]:
        """List all API keys for a user."""
        return [key for key in self._keys_db.values() if key.user_id == user_id]

    def get_usage_stats(self, key_id: UUID) -> Dict[str, Any]:
        """Get usage statistics for an API key."""
        for api_key in self._keys_db.values():
            if api_key.key_id == key_id:
                return {
                    "usage_count": api_key.usage_count,
                    "last_used": api_key.last_used,
                    "rate_limit_per_hour": api_key.rate_limit_per_hour,
                    "is_active": api_key.is_active,
                }
        return {}

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()


# Convenience functions for the module interface
_default_manager = APIKeyManager()


def create_api_key(name: str, user_id: UUID, roles: Set[UserRole], **kwargs) -> tuple[str, APIKey]:
    """Create an API key using the default manager."""
    return _default_manager.generate_api_key(name, user_id, roles, **kwargs)


def verify_api_key(raw_key: str, request_ip: Optional[str] = None) -> APIKey:
    """Verify an API key using the default manager."""
    return _default_manager.verify_api_key(raw_key, request_ip)
