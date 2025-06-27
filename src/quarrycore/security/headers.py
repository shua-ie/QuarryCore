"""
Security headers middleware for QuarryCore.

Implements comprehensive security headers following OWASP best practices.
"""

from __future__ import annotations

from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds comprehensive security headers to all responses.

    Implements:
    - Content Security Policy (CSP)
    - X-Frame-Options
    - X-Content-Type-Options
    - Referrer-Policy
    - X-XSS-Protection
    - Strict-Transport-Security (HSTS)
    - Permissions Policy
    """

    def __init__(
        self,
        app,
        csp_directives: Optional[dict[str, str]] = None,
        allowed_origins: Optional[list[str]] = None,
        enable_hsts: bool = True,
        hsts_max_age: int = 31536000,  # 1 year
    ):
        super().__init__(app)
        self.csp_directives = csp_directives or self._default_csp()
        self.allowed_origins = allowed_origins or []
        self.enable_hsts = enable_hsts
        self.hsts_max_age = hsts_max_age

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to the response."""
        response = await call_next(request)

        # Content Security Policy
        csp = self._build_csp()
        response.headers["Content-Security-Policy"] = csp

        # X-Frame-Options - Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options - Prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Referrer Policy - Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # X-XSS-Protection - Enable XSS filtering (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Strict-Transport-Security - Force HTTPS
        if self.enable_hsts:
            response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains; preload"

        # Permissions Policy - Control browser features
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), "
            "gyroscope=(), magnetometer=(), microphone=(), "
            "payment=(), usb=()"
        )

        # Remove potentially dangerous headers
        headers_to_remove = ["X-Powered-By", "Server"]
        for header in headers_to_remove:
            if header in response.headers:
                del response.headers[header]

        # CORS headers if origin is allowed
        origin = request.headers.get("origin")
        if origin and self._is_origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-API-Key, X-Request-ID"
            response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours

        return response

    def _default_csp(self) -> dict[str, str]:
        """Return default CSP directives."""
        return {
            "default-src": "'self'",
            "script-src": "'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net",
            "style-src": "'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src": "'self' https://fonts.gstatic.com",
            "img-src": "'self' data: https:",
            "connect-src": "'self' wss: https:",
            "frame-ancestors": "'none'",
            "base-uri": "'self'",
            "form-action": "'self'",
            "upgrade-insecure-requests": "",
        }

    def _build_csp(self) -> str:
        """Build CSP header from directives."""
        parts = []
        for directive, value in self.csp_directives.items():
            if value:
                parts.append(f"{directive} {value}")
            else:
                parts.append(directive)
        return "; ".join(parts)

    def _is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed for CORS."""
        if not self.allowed_origins:
            return False

        # Allow exact matches
        if origin in self.allowed_origins:
            return True

        # Allow wildcard subdomains
        for allowed in self.allowed_origins:
            if allowed.startswith("*."):
                domain = allowed[2:]
                if origin.endswith(domain):
                    return True

        return False
