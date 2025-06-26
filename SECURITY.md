# Security Policy 🔒

QuarryCore includes security features designed to protect data processing pipelines from common vulnerabilities. This document outlines our security practices and how to report security issues.

## 🛡️ Our Security Approach

- **Security by Design**: Security features built into the core architecture
- **Open Communication**: Transparent about security capabilities and limitations
- **Community-Driven**: We welcome security feedback and contributions

## 📋 Supported Versions

QuarryCore is currently in active development:

| Version | Status | Notes |
| ------- | ------ | ----- |
| main branch | Active Development | Latest features and fixes |
| < 0.1.0 | Pre-release | Not recommended for production |

## 🚨 Reporting Security Issues

### How to Report

If you discover a security vulnerability in QuarryCore:

1. **Email**: Send details to [josh.mcd31@gmail.com](mailto:josh.mcd31@gmail.com)
2. **Subject**: Use "SECURITY: [Brief Description]" as the subject line
3. **Details**: Include reproduction steps and impact assessment

**Please avoid:**
- Creating public GitHub issues for security vulnerabilities
- Discussing vulnerabilities publicly before they're addressed

### What to Include

Please provide:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fixes (if any)

## 🔍 Implemented Security Features

### Authentication & Authorization

QuarryCore implements JWT-based authentication:

```python
# src/quarrycore/auth/jwt_manager.py
- JWT token generation and validation
- Access and refresh token support
- Token revocation capabilities
- Role-based access control (RBAC)
```

**Key Features:**
- Configurable token expiry
- Secure token generation using `secrets` module
- Token revocation tracking
- User role management

### Input Validation

Comprehensive input validation throughout:

```python
# src/quarrycore/security/validation.py
- URL validation with private IP blocking
- File path traversal prevention
- Input size limits
- Content type validation
```

**Protections:**
- Blocks access to private IP ranges (10.x, 192.168.x, etc.)
- Prevents directory traversal attacks
- Limits input sizes to prevent DoS
- Validates file types and content

### Rate Limiting

Distributed rate limiting implementation:

```python
# src/quarrycore/security/rate_limiter.py
- Redis-backed distributed rate limiting
- Memory-based fallback
- Sliding window algorithm
- Per-user and per-endpoint limits
```

**Features:**
- Configurable rate limits
- Burst allowance support
- Graceful degradation without Redis
- IP-based and user-based limiting

### Security Headers

HTTP security headers for web components:

```python
# src/quarrycore/security/headers.py
- Content Security Policy (CSP)
- X-Frame-Options
- X-Content-Type-Options
- Strict-Transport-Security
```

### Enterprise Security Features

Additional enterprise-grade security:

```python
# src/quarrycore/security/enterprise_security.py
- API key management
- Audit logging
- Session management
- Advanced monitoring
```

## 🔧 Security Configuration

### Basic Security Setup

```yaml
# config.yaml
security:
  jwt:
    secret_key: "${JWT_SECRET_KEY}"  # Use environment variable
    access_token_expire_minutes: 30
    refresh_token_expire_days: 7
  
  rate_limiting:
    enabled: true
    default_limit: 1000  # requests per hour
    burst_size: 50
  
  validation:
    max_input_size: 10485760  # 10MB
    allowed_schemes: ["http", "https"]
    block_private_ips: true
```

### Docker Security

When using Docker:

```bash
# Run with security constraints
docker run \
  --user 1000:1000 \           # Non-root user
  --read-only \                # Read-only filesystem
  --security-opt no-new-privileges \
  -e JWT_SECRET_KEY="$SECRET" \
  quarrycore:latest
```

## 🧪 Security Testing

### Running Security Tests

```bash
# Run security-specific tests
pytest tests/test_security_comprehensive.py -v

# Check for common vulnerabilities
bandit -r src/ -ll

# Scan dependencies
pip-audit
```

### What We Test

- Input validation edge cases
- Rate limiting effectiveness
- Authentication flows
- Authorization boundaries
- SQL injection prevention
- Path traversal protection

## 📊 Known Limitations

### Current Limitations

1. **Token Storage**: Token revocation list is in-memory (production should use Redis)
2. **HTTPS**: Not enforced at application level (use reverse proxy)
3. **Encryption**: Data encryption at rest not implemented (use filesystem encryption)
4. **Secrets Management**: Basic environment variable support (consider HashiCorp Vault for production)

### Planned Improvements

- Enhanced secrets management
- Built-in HTTPS support
- Data encryption at rest
- Security event monitoring
- SIEM integration support

## 🔒 Best Practices

### Deployment Security

1. **Use Environment Variables** for sensitive configuration
2. **Enable Rate Limiting** to prevent abuse
3. **Run as Non-Root User** in containers
4. **Use HTTPS** via reverse proxy (nginx, Caddy)
5. **Regular Updates** - Keep dependencies current

### Configuration Security

```bash
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set via environment
export JWT_SECRET_KEY="your-generated-secret"
export REDIS_URL="redis://localhost:6379"
```

## 📞 Contact

### Security Questions

For security-related questions:
- **Email**: [josh.mcd31@gmail.com](mailto:josh.mcd31@gmail.com)
- **GitHub Issues**: For non-sensitive security discussions

### Contributing to Security

We welcome security improvements:
- Add security tests
- Improve validation logic
- Enhance documentation
- Report vulnerabilities responsibly

---

## 🙏 Acknowledgments

Thank you to everyone who helps improve QuarryCore's security. Your efforts help protect all users of the system.

**Remember**: Security is an ongoing process, not a destination. We're committed to continuous improvement. 