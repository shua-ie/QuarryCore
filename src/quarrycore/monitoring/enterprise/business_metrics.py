"""
Enterprise Business Metrics System.

This module provides Fortune 500-grade business metrics with:
- Comprehensive KPI tracking with predictive analytics
- Real-time cost tracking and optimization recommendations
- SLA compliance monitoring by service tier
- Revenue impact tracking and customer satisfaction metrics
- Automated alerting with business intelligence
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Callable, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, Summary
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import structlog

logger = structlog.get_logger(__name__)


class AlertType(Enum):
    """Business alert types."""
    QUALITY_DEGRADATION = "quality_degradation"
    COST_SPIKE = "cost_spike"
    SLA_VIOLATION = "sla_violation"
    REVENUE_IMPACT = "revenue_impact"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    CAPACITY_WARNING = "capacity_warning"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProcessingTier(Enum):
    """Customer processing tiers."""
    ENTERPRISE = "enterprise"
    PROFESSIONAL = "professional"
    STANDARD = "standard"
    BASIC = "basic"


@dataclass
class ProcessingContext:
    """Context for document processing operations."""
    correlation_id: UUID
    customer_id: str
    tier: ProcessingTier
    final_stage: str
    start_time: datetime
    customer_tier: str = "standard"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for processing operations."""
    compute_cost: float
    storage_cost: float
    network_cost: float
    ai_model_cost: float
    overhead_cost: float
    
    def compute_total_cost(self) -> float:
        """Calculate total processing cost."""
        return (
            self.compute_cost + 
            self.storage_cost + 
            self.network_cost + 
            self.ai_model_cost + 
            self.overhead_cost
        )


@dataclass
class QualityTrend:
    """Quality trend analysis result."""
    is_declining: bool
    severity: float
    details: str
    suggested_fixes: List[str]
    time_window: timedelta
    trend_score: float = 0.0


@dataclass
class CostTrend:
    """Cost trend analysis result."""
    is_spiking: bool
    increase_percent: float
    projected_monthly_impact: float
    time_window: timedelta
    spike_reasons: List[str] = field(default_factory=list)


@dataclass
class SLAMetrics:
    """SLA compliance metrics."""
    target_uptime: float
    actual_uptime: float
    target_response_time: float
    actual_response_time: float
    target_throughput: int
    actual_throughput: int
    compliance_score: float = 0.0


class EnterpriseBusinessMetrics:
    """
    Advanced business metrics with predictive analytics and alerting.
    Provides actionable insights for business operations.
    """
    
    def __init__(self):
        self.business_analyzer = BusinessMetricsAnalyzer()
        self.alert_manager = AlertManager()
        self.cost_tracker = CostTracker()
        
        # Metrics storage
        self.document_counts: Dict[str, int] = {}
        self.processing_costs: Dict[str, List[float]] = {}
        self.quality_scores: Dict[str, List[float]] = {}
        self.customer_satisfaction: Dict[str, float] = {}
        self.sla_metrics: Dict[str, SLAMetrics] = {}
        
    async def track_document_processing(
        self,
        document: Any,
        processing_context: ProcessingContext,
        cost_breakdown: CostBreakdown
    ):
        """Track comprehensive document processing metrics."""
        
        domain = getattr(document, 'domain', 'unknown')
        quality_score = getattr(document, 'quality_score', 0.5)
        
        # Basic processing metrics
        key = f"{domain}_{processing_context.tier.value}"
        self.document_counts[key] = self.document_counts.get(key, 0) + 1
        
        # Cost tracking with detailed breakdown
        total_cost = cost_breakdown.compute_total_cost()
        if key not in self.processing_costs:
            self.processing_costs[key] = []
        self.processing_costs[key].append(total_cost)
        
        # Quality score tracking
        if key not in self.quality_scores:
            self.quality_scores[key] = []
        self.quality_scores[key].append(quality_score)
        
        # Business value assessment
        business_value = await self.business_analyzer.calculate_value_score(
            document=document,
            processing_context=processing_context,
            market_demand=await self._get_market_demand(domain)
        )
        
        logger.info(
            "Document processed",
            domain=domain,
            tier=processing_context.tier.value,
            cost=total_cost,
            quality=quality_score,
            business_value=business_value,
            correlation_id=str(processing_context.correlation_id)
        )
        
        # Predictive analytics and alerting
        await self._analyze_trends_and_alert(document, processing_context, business_value)
        
        # Update cost tracker
        await self.cost_tracker.track_processing_cost(
            customer_id=processing_context.customer_id,
            domain=domain,
            cost=total_cost,
            timestamp=datetime.utcnow()
        )
    
    async def _analyze_trends_and_alert(
        self,
        document: Any,
        context: ProcessingContext,
        business_value: float
    ):
        """Real-time trend analysis with predictive alerting."""
        
        domain = getattr(document, 'domain', 'unknown')
        
        # Quality degradation detection
        quality_trend = await self.business_analyzer.analyze_quality_trend(
            domain=domain,
            time_window=timedelta(hours=24)
        )
        
        if quality_trend.is_declining and quality_trend.severity > 0.3:
            await self.alert_manager.send_alert(
                alert_type=AlertType.QUALITY_DEGRADATION,
                severity=AlertSeverity.HIGH,
                message=f"Quality declining in {domain}: {quality_trend.details}",
                recommended_actions=quality_trend.suggested_fixes,
                correlation_id=context.correlation_id
            )
        
        # Cost spike detection
        cost_trend = await self.business_analyzer.analyze_cost_trend(
            domain=domain,
            processing_tier=context.tier,
            time_window=timedelta(hours=6)
        )
        
        if cost_trend.is_spiking and cost_trend.increase_percent > 50:
            await self.alert_manager.send_alert(
                alert_type=AlertType.COST_SPIKE,
                severity=AlertSeverity.CRITICAL,
                message=f"Processing costs spiking: {cost_trend.increase_percent}% increase",
                estimated_impact=cost_trend.projected_monthly_impact,
                correlation_id=context.correlation_id
            )
        
        # SLA compliance monitoring
        await self._check_sla_compliance(context.customer_id, context.tier)
    
    async def _check_sla_compliance(self, customer_id: str, tier: ProcessingTier):
        """Monitor SLA compliance for customer tier."""
        
        sla_key = f"{customer_id}_{tier.value}"
        current_sla = self.sla_metrics.get(sla_key)
        
        if current_sla:
            # Check if SLA is being violated
            if (current_sla.actual_uptime < current_sla.target_uptime or
                current_sla.actual_response_time > current_sla.target_response_time or
                current_sla.actual_throughput < current_sla.target_throughput):
                
                await self.alert_manager.send_alert(
                    alert_type=AlertType.SLA_VIOLATION,
                    severity=AlertSeverity.HIGH,
                    message=f"SLA violation detected for customer {customer_id}",
                    sla_details=current_sla
                )
    
    async def _get_market_demand(self, domain: str) -> float:
        """Get market demand score for domain (simplified)."""
        demand_scores = {
            'finance': 0.9,
            'healthcare': 0.8,
            'technology': 0.9,
            'legal': 0.7,
            'education': 0.6,
            'retail': 0.7
        }
        return demand_scores.get(domain.lower(), 0.5)
    
    async def get_business_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive business dashboard data."""
        
        total_documents = sum(self.document_counts.values())
        total_cost = sum(sum(costs) for costs in self.processing_costs.values())
        
        # Calculate average quality score
        all_quality_scores = []
        for scores in self.quality_scores.values():
            all_quality_scores.extend(scores)
        avg_quality = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0
        
        # Cost per document
        cost_per_document = total_cost / total_documents if total_documents > 0 else 0
        
        return {
            "total_documents_processed": total_documents,
            "total_processing_cost": total_cost,
            "average_quality_score": avg_quality,
            "cost_per_document": cost_per_document,
            "active_customers": len(set(self.customer_satisfaction.keys())),
            "average_customer_satisfaction": sum(self.customer_satisfaction.values()) / len(self.customer_satisfaction) if self.customer_satisfaction else 0,
            "sla_compliance_rate": await self._calculate_overall_sla_compliance(),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def _calculate_overall_sla_compliance(self) -> float:
        """Calculate overall SLA compliance rate."""
        if not self.sla_metrics:
            return 1.0
        
        compliance_scores = [sla.compliance_score for sla in self.sla_metrics.values()]
        return sum(compliance_scores) / len(compliance_scores)


class BusinessMetricsAnalyzer:
    """Analyzes business metrics for trends and insights."""
    
    def __init__(self):
        self.historical_data: Dict[str, List[Tuple[datetime, float]]] = {}
    
    async def calculate_value_score(
        self,
        document: Any,
        processing_context: ProcessingContext,
        market_demand: float
    ) -> float:
        """Calculate business value score for processed document."""
        
        # Base value from quality
        quality_score = getattr(document, 'quality_score', 0.5)
        base_value = quality_score * 0.4
        
        # Market demand factor
        market_value = market_demand * 0.3
        
        # Customer tier multiplier
        tier_multipliers = {
            ProcessingTier.ENTERPRISE: 1.5,
            ProcessingTier.PROFESSIONAL: 1.2,
            ProcessingTier.STANDARD: 1.0,
            ProcessingTier.BASIC: 0.8
        }
        tier_value = tier_multipliers.get(processing_context.tier, 1.0) * 0.2
        
        # Content uniqueness (simplified)
        uniqueness_score = getattr(document, 'uniqueness_score', 0.7) * 0.1
        
        return min(1.0, base_value + market_value + tier_value + uniqueness_score)
    
    async def analyze_quality_trend(self, domain: str, time_window: timedelta) -> QualityTrend:
        """Analyze quality trends for a domain."""
        
        # Simplified trend analysis
        key = f"quality_{domain}"
        if key not in self.historical_data:
            return QualityTrend(
                is_declining=False,
                severity=0.0,
                details="Insufficient data for trend analysis",
                suggested_fixes=[],
                time_window=time_window
            )
        
        # Get recent data points
        recent_data = self.historical_data[key][-10:]  # Last 10 data points
        
        if len(recent_data) < 3:
            return QualityTrend(
                is_declining=False,
                severity=0.0,
                details="Insufficient recent data",
                suggested_fixes=[],
                time_window=time_window
            )
        
        # Simple trend calculation
        values = [point[1] for point in recent_data]
        recent_avg = sum(values[-3:]) / 3
        older_avg = sum(values[:3]) / 3
        
        is_declining = recent_avg < older_avg * 0.9  # 10% decline
        severity = (older_avg - recent_avg) / older_avg if older_avg > 0 else 0
        
        suggested_fixes = []
        if is_declining:
            suggested_fixes = [
                "Review content extraction parameters",
                "Check for data source quality issues",
                "Analyze processing pipeline bottlenecks",
                "Consider model retraining"
            ]
        
        return QualityTrend(
            is_declining=is_declining,
            severity=max(0.0, severity),
            details=f"Quality trend: {recent_avg:.3f} vs {older_avg:.3f}",
            suggested_fixes=suggested_fixes,
            time_window=time_window
        )
    
    async def analyze_cost_trend(
        self,
        domain: str,
        processing_tier: ProcessingTier,
        time_window: timedelta
    ) -> CostTrend:
        """Analyze cost trends for domain and tier."""
        
        # Simplified cost trend analysis
        key = f"cost_{domain}_{processing_tier.value}"
        if key not in self.historical_data:
            return CostTrend(
                is_spiking=False,
                increase_percent=0.0,
                projected_monthly_impact=0.0,
                time_window=time_window
            )
        
        recent_data = self.historical_data[key][-10:]
        if len(recent_data) < 5:
            return CostTrend(
                is_spiking=False,
                increase_percent=0.0,
                projected_monthly_impact=0.0,
                time_window=time_window
            )
        
        # Calculate cost trend
        values = [point[1] for point in recent_data]
        recent_avg = sum(values[-3:]) / 3
        baseline_avg = sum(values[:5]) / 5
        
        increase_percent = ((recent_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
        is_spiking = increase_percent > 30  # 30% increase threshold
        
        # Project monthly impact
        daily_increase = recent_avg - baseline_avg
        projected_monthly_impact = daily_increase * 30
        
        return CostTrend(
            is_spiking=is_spiking,
            increase_percent=increase_percent,
            projected_monthly_impact=projected_monthly_impact,
            time_window=time_window,
            spike_reasons=["Increased processing complexity", "Higher volume"] if is_spiking else []
        )


class AlertManager:
    """Manages business alerts and notifications."""
    
    def __init__(self):
        self.alert_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            AlertType.QUALITY_DEGRADATION: 0.3,
            AlertType.COST_SPIKE: 50.0,
            AlertType.SLA_VIOLATION: 0.95,
            AlertType.REVENUE_IMPACT: 1000.0
        }
    
    async def send_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        correlation_id: Optional[UUID] = None,
        **kwargs
    ):
        """Send business alert with context."""
        
        alert = {
            "alert_id": str(uuid4()),
            "alert_type": alert_type.value,
            "severity": severity.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": str(correlation_id) if correlation_id else None,
            "metadata": kwargs
        }
        
        self.alert_history.append(alert)
        
        # In production, send to alerting systems (PagerDuty, Slack, etc.)
        logger.warning(
            f"BUSINESS ALERT: {alert_type.value}",
            severity=severity.value,
            message=message,
            alert_id=alert["alert_id"]
        )
        
        # Auto-escalation for critical alerts
        if severity == AlertSeverity.CRITICAL:
            await self._escalate_critical_alert(alert)
    
    async def _escalate_critical_alert(self, alert: Dict[str, Any]):
        """Escalate critical alerts to leadership."""
        logger.critical(
            "CRITICAL BUSINESS ALERT ESCALATED",
            alert_id=alert["alert_id"],
            alert_type=alert["alert_type"],
            message=alert["message"]
        )


class CostTracker:
    """Tracks and optimizes processing costs."""
    
    def __init__(self):
        self.cost_history: Dict[str, List[Dict[str, Any]]] = {}
        self.cost_optimization_rules = {
            "high_cost_threshold": 1.0,  # $1 per document
            "optimization_suggestions": [
                "Consider batch processing for similar documents",
                "Optimize AI model usage for cost efficiency",
                "Implement caching for repeated processing patterns"
            ]
        }
    
    async def track_processing_cost(
        self,
        customer_id: str,
        domain: str,
        cost: float,
        timestamp: datetime
    ):
        """Track processing cost for optimization."""
        
        key = f"{customer_id}_{domain}"
        if key not in self.cost_history:
            self.cost_history[key] = []
        
        cost_entry = {
            "cost": cost,
            "timestamp": timestamp.isoformat(),
            "domain": domain,
            "customer_id": customer_id
        }
        
        self.cost_history[key].append(cost_entry)
        
        # Cost optimization analysis
        if cost > self.cost_optimization_rules["high_cost_threshold"]:
            await self._suggest_cost_optimization(customer_id, domain, cost)
    
    async def _suggest_cost_optimization(self, customer_id: str, domain: str, cost: float):
        """Suggest cost optimization strategies."""
        
        logger.info(
            "Cost optimization opportunity",
            customer_id=customer_id,
            domain=domain,
            cost=cost,
            suggestions=self.cost_optimization_rules["optimization_suggestions"]
        )
    
    async def get_cost_report(self, customer_id: str, time_period: timedelta) -> Dict[str, Any]:
        """Generate cost report for customer."""
        
        cutoff_time = datetime.utcnow() - time_period
        total_cost = 0.0
        document_count = 0
        
        for key, history in self.cost_history.items():
            if customer_id in key:
                for entry in history:
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    if entry_time >= cutoff_time:
                        total_cost += entry["cost"]
                        document_count += 1
        
        avg_cost_per_document = total_cost / document_count if document_count > 0 else 0
        
        return {
            "customer_id": customer_id,
            "time_period_days": time_period.days,
            "total_cost": total_cost,
            "documents_processed": document_count,
            "average_cost_per_document": avg_cost_per_document,
            "cost_trend": "stable",  # Simplified
            "optimization_opportunities": document_count if avg_cost_per_document > 0.5 else 0
        }


# Mock classes for when dependencies are not available

class MockMetricsRegistry:
    """Mock metrics registry."""
    def collect(self):
        return []


class MockCounter:
    """Mock Prometheus counter."""
    def labels(self, **kwargs):
        return self
    
    def inc(self, amount=1):
        pass


class MockHistogram:
    """Mock Prometheus histogram."""
    def labels(self, **kwargs):
        return self
    
    def observe(self, amount):
        pass


class MockGauge:
    """Mock Prometheus gauge."""
    def labels(self, **kwargs):
        return self
    
    def set(self, value):
        pass
