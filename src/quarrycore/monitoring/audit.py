"""
Audit logging for QuarryCore compliance and security.

Provides comprehensive logging of API access, configuration changes, and security events.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import structlog
from cryptography.fernet import Fernet


class AuditEventType(Enum):
    """Types of audit events."""
    API_ACCESS = "api_access"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    CONFIGURATION_CHANGE = "configuration_change"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    SECURITY_EVENT = "security_event"
    ADMIN_ACTION = "admin_action"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditEvent:
    """Audit event with all required metadata."""
    event_id: UUID
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    user_roles: list[str]
    ip_address: Optional[str]
    resource: str
    action: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    correlation_id: Optional[UUID]
    risk_score: float = 0.0  # 0-1, higher is riskier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "user_roles": self.user_roles,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "outcome": self.outcome,
            "details": self.details,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "risk_score": self.risk_score,
        }


class AuditLogger:
    """
    Comprehensive audit logging for compliance and security.
    
    Features:
    - All API access logging with user identification
    - Configuration change tracking
    - Security event logging
    - GDPR compliance reporting
    - Structured logging with correlation IDs
    - Encryption at rest for sensitive data
    """
    
    def __init__(
        self,
        log_dir: Path = Path("./logs/audit"),
        encryption_key: Optional[bytes] = None,
        retention_days: int = 365,
        enable_console: bool = True,
    ):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = retention_days
        
        # Set up encryption for sensitive data
        self.fernet = None
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        
        # Set up structured logging
        self.logger = structlog.get_logger("audit")
        
        # Set up file handler for audit logs
        audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        file_handler = logging.FileHandler(audit_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Configure structured logger
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def log_api_access(
        self,
        user_id: Optional[str],
        user_roles: list[str],
        endpoint: str,
        method: str,
        ip_address: Optional[str],
        request_id: Optional[UUID] = None,
        status_code: int = 200,
        response_time_ms: float = 0.0,
        **kwargs
    ) -> None:
        """Log API access event."""
        event = AuditEvent(
            event_id=uuid4(),
            event_type=AuditEventType.API_ACCESS,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            user_roles=user_roles,
            ip_address=ip_address,
            resource=endpoint,
            action=method,
            outcome="success" if 200 <= status_code < 400 else "failure",
            details={
                "status_code": status_code,
                "response_time_ms": response_time_ms,
                **kwargs
            },
            correlation_id=request_id,
        )
        
        self._log_event(event)
    
    def log_authentication(
        self,
        user_id: Optional[str],
        auth_method: str,
        success: bool,
        ip_address: Optional[str],
        failure_reason: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log authentication attempt."""
        event = AuditEvent(
            event_id=uuid4(),
            event_type=AuditEventType.AUTHENTICATION,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            user_roles=[],
            ip_address=ip_address,
            resource="auth",
            action=f"login_{auth_method}",
            outcome="success" if success else "failure",
            details={
                "auth_method": auth_method,
                "failure_reason": failure_reason,
                **kwargs
            },
            correlation_id=None,
            risk_score=0.0 if success else 0.5,
        )
        
        self._log_event(event)
    
    def log_data_access(
        self,
        user_id: str,
        user_roles: list[str],
        resource_type: str,
        resource_id: str,
        action: str,
        ip_address: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log data access for GDPR compliance."""
        event = AuditEvent(
            event_id=uuid4(),
            event_type=AuditEventType.DATA_ACCESS,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            user_roles=user_roles,
            ip_address=ip_address,
            resource=f"{resource_type}/{resource_id}",
            action=action,
            outcome="success",
            details=kwargs,
            correlation_id=None,
        )
        
        self._log_event(event)
    
    def log_configuration_change(
        self,
        user_id: str,
        user_roles: list[str],
        config_key: str,
        old_value: Any,
        new_value: Any,
        ip_address: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log configuration changes."""
        # Mask sensitive values
        masked_old = self._mask_sensitive_value(config_key, old_value)
        masked_new = self._mask_sensitive_value(config_key, new_value)
        
        event = AuditEvent(
            event_id=uuid4(),
            event_type=AuditEventType.CONFIGURATION_CHANGE,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            user_roles=user_roles,
            ip_address=ip_address,
            resource=f"config/{config_key}",
            action="update",
            outcome="success",
            details={
                "old_value": masked_old,
                "new_value": masked_new,
                **kwargs
            },
            correlation_id=None,
            risk_score=0.3,  # Config changes are medium risk
        )
        
        self._log_event(event)
    
    def log_security_event(
        self,
        event_name: str,
        severity: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log security events."""
        risk_scores = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 1.0,
        }
        
        event = AuditEvent(
            event_id=uuid4(),
            event_type=AuditEventType.SECURITY_EVENT,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            user_roles=[],
            ip_address=ip_address,
            resource="security",
            action=event_name,
            outcome="detected",
            details={
                "severity": severity,
                **(details or {}),
                **kwargs
            },
            correlation_id=None,
            risk_score=risk_scores.get(severity, 0.5),
        )
        
        self._log_event(event)
    
    def get_user_activity(
        self,
        user_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> list[AuditEvent]:
        """Get all activity for a user (GDPR support)."""
        events = []
        
        # Read audit logs for date range
        current_date = start_date
        while current_date <= end_date:
            log_file = self.log_dir / f"audit_{current_date.strftime('%Y%m%d')}.jsonl"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            event_data = json.loads(line)
                            if event_data.get("user_id") == user_id:
                                # Convert back to AuditEvent
                                event = self._dict_to_event(event_data)
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
            
            current_date = current_date.replace(day=current_date.day + 1)
        
        return events
    
    def _log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        event_dict = event.to_dict()
        
        # Encrypt sensitive details if encryption is enabled
        if self.fernet and event.details:
            event_dict["details"] = self._encrypt_data(event.details)
        
        # Log with structured logger
        self.logger.info(
            event.action,
            **event_dict
        )
    
    def _mask_sensitive_value(self, key: str, value: Any) -> Any:
        """Mask sensitive configuration values."""
        sensitive_keys = [
            "password", "secret", "key", "token",
            "api_key", "private_key", "credential"
        ]
        
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            if isinstance(value, str) and len(value) > 4:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
            return "***MASKED***"
        
        return value
    
    def _encrypt_data(self, data: Dict[str, Any]) -> str:
        """Encrypt sensitive data."""
        if not self.fernet:
            return json.dumps(data)
        
        json_data = json.dumps(data).encode()
        encrypted = self.fernet.encrypt(json_data)
        return encrypted.decode()
    
    def _decrypt_data(self, encrypted: str) -> Dict[str, Any]:
        """Decrypt sensitive data."""
        if not self.fernet:
            return json.loads(encrypted)
        
        decrypted = self.fernet.decrypt(encrypted.encode())
        return json.loads(decrypted.decode())
    
    def _dict_to_event(self, data: Dict[str, Any]) -> AuditEvent:
        """Convert dictionary back to AuditEvent."""
        return AuditEvent(
            event_id=UUID(data["event_id"]),
            event_type=AuditEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user_id=data.get("user_id"),
            user_roles=data.get("user_roles", []),
            ip_address=data.get("ip_address"),
            resource=data["resource"],
            action=data["action"],
            outcome=data["outcome"],
            details=data.get("details", {}),
            correlation_id=UUID(data["correlation_id"]) if data.get("correlation_id") else None,
            risk_score=data.get("risk_score", 0.0),
        )


# Global audit logger
_audit_logger = AuditLogger()

def audit_log(event_type: AuditEventType, **kwargs) -> None:
    """Log an audit event using the global logger."""
    event = AuditEvent(
        event_id=uuid4(),
        event_type=event_type,
        timestamp=datetime.utcnow(),
        user_id=kwargs.get("user_id"),
        user_roles=kwargs.get("user_roles", []),
        ip_address=kwargs.get("ip_address"),
        resource=kwargs.get("resource", ""),
        action=kwargs.get("action", ""),
        outcome=kwargs.get("outcome", "success"),
        details=kwargs.get("details", {}),
        correlation_id=kwargs.get("correlation_id"),
        risk_score=kwargs.get("risk_score", 0.0),
    )
    
    _audit_logger._log_event(event) 