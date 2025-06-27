"""
Input validation utilities for QuarryCore.

Provides secure validation for URLs, file uploads, and other user inputs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field


class URLValidationError(ValueError):
    """Raised when URL validation fails."""

    pass


class FileValidationError(ValueError):
    """Raised when file validation fails."""

    pass


class URLValidationRules(BaseModel):
    """Rules for URL validation."""

    allowed_schemes: List[str] = Field(default=["http", "https"])
    allowed_domains: Optional[List[str]] = None
    blocked_domains: List[str] = Field(
        default_factory=lambda: [
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            "169.254.169.254",  # AWS metadata endpoint
        ]
    )
    max_url_length: int = 2048
    allow_private_ips: bool = False


class FileUploadRules(BaseModel):
    """Rules for file upload validation."""

    max_file_size_mb: int = 100
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [
            ".txt",
            ".json",
            ".jsonl",
            ".csv",
            ".xml",
            ".html",
            ".md",
        ]
    )
    allowed_content_types: List[str] = Field(
        default_factory=lambda: [
            "text/plain",
            "text/html",
            "text/csv",
            "text/xml",
            "application/json",
            "application/xml",
        ]
    )
    scan_for_malware: bool = True


class InputValidator:
    """
    Comprehensive input validation for security.

    Features:
    - URL validation with scheme and domain checks
    - File upload validation with size and type limits
    - SQL injection pattern detection
    - XSS pattern detection
    - Path traversal prevention
    """

    def __init__(
        self,
        url_rules: Optional[URLValidationRules] = None,
        file_rules: Optional[FileUploadRules] = None,
    ):
        self.url_rules = url_rules or URLValidationRules()
        self.file_rules = file_rules or FileUploadRules()

        # Dangerous patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
            r"(--|\||;|\/\*|\*\/)",
            r"(\bOR\b\s*\d+\s*=\s*\d+)",
            r"(\bAND\b\s*\d+\s*=\s*\d+)",
        ]

        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"onerror\s*=",
            r"onclick\s*=",
            r"onload\s*=",
            r"<iframe",
            r"<object",
            r"<embed",
        ]

        self.path_traversal_patterns = [
            r"\.\./",
            r"\.\.\\",
            r"%2e%2e/",
            r"%252e%252e/",
            r"\.\.%2f",
            r"\.\.%5c",
        ]

    def validate_url(self, url: str) -> str:
        """
        Validate and sanitize a URL.

        Returns:
            Sanitized URL

        Raises:
            URLValidationError: If URL is invalid or dangerous
        """
        # Check length
        if len(url) > self.url_rules.max_url_length:
            raise URLValidationError(f"URL exceeds maximum length of {self.url_rules.max_url_length}")

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise URLValidationError(f"Invalid URL format: {e}")

        # Check scheme
        if parsed.scheme not in self.url_rules.allowed_schemes:
            raise URLValidationError(f"Invalid URL scheme: {parsed.scheme}")

        # Check for localhost/private IPs
        if not self.url_rules.allow_private_ips:
            hostname = parsed.hostname
            if hostname and any(blocked in hostname for blocked in self.url_rules.blocked_domains):
                raise URLValidationError(f"Blocked domain: {hostname}")

            # Check for private IP ranges
            if hostname and self._is_private_ip(hostname):
                raise URLValidationError(f"Private IP addresses not allowed: {hostname}")

        # Check allowed domains
        if self.url_rules.allowed_domains:
            if not any(parsed.netloc.endswith(domain) for domain in self.url_rules.allowed_domains):
                raise URLValidationError(f"Domain not in allowlist: {parsed.netloc}")

        # Check for SQL injection patterns
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.sql_injection_patterns):
            raise URLValidationError("Potential SQL injection detected")

        # Check for XSS patterns
        if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.xss_patterns):
            raise URLValidationError("Potential XSS attack detected")

        return url

    def validate_file_upload(
        self,
        filename: str,
        content_type: str,
        file_size: int,
        file_content: Optional[bytes] = None,
    ) -> None:
        """
        Validate a file upload.

        Raises:
            FileValidationError: If file is invalid or dangerous
        """
        # Check file size
        max_size_bytes = self.file_rules.max_file_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise FileValidationError(f"File size exceeds {self.file_rules.max_file_size_mb}MB limit")

        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.file_rules.allowed_extensions:
            raise FileValidationError(f"File type not allowed: {file_ext}")

        # Check content type
        if content_type not in self.file_rules.allowed_content_types:
            raise FileValidationError(f"Content type not allowed: {content_type}")

        # Check for path traversal in filename
        if any(re.search(pattern, filename, re.IGNORECASE) for pattern in self.path_traversal_patterns):
            raise FileValidationError("Potential path traversal detected")

        # Check file content if provided
        if file_content and self.file_rules.scan_for_malware:
            self._scan_for_malware(file_content)

    def sanitize_string(self, input_string: str, max_length: int = 1000) -> str:
        """Sanitize a string input by removing dangerous patterns."""
        # Truncate to max length
        sanitized = input_string[:max_length]

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Remove control characters
        sanitized = "".join(char for char in sanitized if ord(char) >= 32 or char in "\n\r\t")

        # HTML encode special characters
        sanitized = sanitized.replace("&", "&amp;")
        sanitized = sanitized.replace("<", "&lt;")
        sanitized = sanitized.replace(">", "&gt;")
        sanitized = sanitized.replace('"', "&quot;")
        sanitized = sanitized.replace("'", "&#x27;")

        return sanitized

    def _is_private_ip(self, hostname: str) -> bool:
        """Check if hostname is a private IP address."""
        import ipaddress

        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_private or ip.is_loopback or ip.is_link_local
        except ValueError:
            # Not an IP address
            return False

    def _scan_for_malware(self, content: bytes) -> None:
        """Basic malware scanning (placeholder for real implementation)."""
        # Check for common malware signatures
        malware_signatures = [
            b"EICAR-STANDARD-ANTIVIRUS-TEST-FILE",
            b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR",
        ]

        for signature in malware_signatures:
            if signature in content:
                raise FileValidationError("Potential malware detected")


# Convenience functions
_default_validator = InputValidator()


def validate_url(url: str) -> str:
    """Validate a URL using the default validator."""
    return _default_validator.validate_url(url)


def validate_file_upload(
    filename: str,
    content_type: str,
    file_size: int,
    file_content: Optional[bytes] = None,
) -> None:
    """Validate a file upload using the default validator."""
    _default_validator.validate_file_upload(filename, content_type, file_size, file_content)
