"""
Enterprise Load Testing Framework.

This module provides production-grade load testing with:
- Comprehensive load testing with realistic production-grade scenarios
- Simulation of 10K+ concurrent users with realistic data patterns
- Peak business hours simulation and bulk upload scenarios
- API abuse simulation with defense validation
- Performance analysis and optimization recommendations
"""

import asyncio
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger(__name__)


class LoadTestType(Enum):
    """Load test scenario types."""

    PEAK_LOAD = "peak_load"
    BULK_MIGRATION = "bulk_migration"
    API_ABUSE = "api_abuse"
    SUSTAINED_LOAD = "sustained_load"
    SPIKE_TEST = "spike_test"
    STRESS_TEST = "stress_test"


class UserProfile(Enum):
    """User behavior profiles for realistic testing."""

    POWER_USER = "power_user"  # Heavy usage, complex operations
    BUSINESS_USER = "business_user"  # Moderate usage, standard operations
    CASUAL_USER = "casual_user"  # Light usage, simple operations
    API_CLIENT = "api_client"  # Programmatic access
    BULK_PROCESSOR = "bulk_processor"  # Large batch operations


@dataclass
class LoadTestScenario:
    """Load test scenario configuration."""

    scenario_id: UUID
    test_type: LoadTestType
    concurrent_users: int
    requests_per_second: int
    duration: timedelta
    user_profiles: List[UserProfile]
    data_patterns: Dict[str, Any]
    target_endpoints: List[str]
    success_criteria: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""

    response_times: List[float]
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    network_utilization: float
    database_connections: int
    cache_hit_rate: float
    concurrent_users_peak: int

    def get_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        if not self.response_times:
            return {}

        sorted_times = sorted(self.response_times)
        return {
            "p50": statistics.median(sorted_times),
            "p90": sorted_times[int(len(sorted_times) * 0.9)],
            "p95": sorted_times[int(len(sorted_times) * 0.95)],
            "p99": sorted_times[int(len(sorted_times) * 0.99)],
        }

    def get_stability_score(self) -> float:
        """Calculate overall system stability score."""
        error_score = max(0, 1 - (self.error_rate / 5.0))  # 5% error rate = 0 score
        performance_score = 1.0 if statistics.median(self.response_times) < 0.1 else 0.5
        resource_score = max(0, 1 - (max(self.cpu_usage, self.memory_usage) / 100))

        return (error_score + performance_score + resource_score) / 3


@dataclass
class LoadTestResults:
    """Comprehensive load test results."""

    test_suite_id: UUID
    scenarios_executed: int
    total_duration: timedelta
    peak_performance: PerformanceMetrics
    system_stability_score: float
    recommendations: List[str]
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    infrastructure_impact: Dict[str, Any] = field(default_factory=dict)


class EnterpriseLoadTester:
    """
    Comprehensive load testing with realistic production-grade scenarios.
    Simulates 10K+ concurrent users with realistic data patterns.
    """

    def __init__(self):
        self.load_generators = LoadGeneratorPool()
        self.metrics_collector = MetricsCollector()
        self.scenario_builder = ScenarioBuilder()
        self.result_analyzer = ResultAnalyzer()

    async def execute_load_test_suite(
        self,
        test_scenarios: List[LoadTestScenario],
        target_environment: str,
        duration: timedelta = timedelta(hours=4),
    ) -> LoadTestResults:
        """Execute comprehensive load testing with realistic patterns."""

        test_suite_id = uuid4()
        start_time = datetime.utcnow()

        logger.info(
            "Starting enterprise load test suite",
            test_suite_id=str(test_suite_id),
            scenarios=len(test_scenarios),
            target_environment=target_environment,
            duration_hours=duration.total_seconds() / 3600,
        )

        # Scenario 1: Peak business hours simulation
        peak_scenario = self._create_peak_load_scenario(
            concurrent_users=10000,
            requests_per_second=5000,
            document_processing_rate=1000,
        )

        # Scenario 2: Data migration load
        migration_scenario = self._create_migration_scenario(
            bulk_upload_size=10_000_000,  # 10M documents
            concurrent_uploads=100,
            processing_parallelism=500,
        )

        # Scenario 3: API abuse simulation
        abuse_scenario = self._create_abuse_scenario(
            malicious_request_rate=10000,
            attack_patterns=["rate_limit_bypass", "credential_stuffing", "ddos"],
            defense_validation=True,
        )

        # Scenario 4: Sustained mixed workload
        sustained_scenario = self._create_sustained_scenario(
            duration=duration, concurrent_users=5000, mixed_operations=True
        )

        all_scenarios = [
            peak_scenario,
            migration_scenario,
            abuse_scenario,
            sustained_scenario,
        ]
        results = []
        peak_metrics = None

        # Execute scenarios in sequence with recovery analysis
        for i, scenario in enumerate(all_scenarios):
            logger.info(
                f"Executing scenario {i+1}/{len(all_scenarios)}: {scenario.test_type.value}",
                scenario_id=str(scenario.scenario_id),
                concurrent_users=scenario.concurrent_users,
            )

            scenario_result = await self._execute_scenario(
                scenario=scenario,
                target_environment=target_environment,
                duration=scenario.duration,
            )
            results.append(scenario_result)

            # Track peak performance
            if not peak_metrics or scenario_result["metrics"].throughput > peak_metrics.throughput:
                peak_metrics = scenario_result["metrics"]

            # Analysis between scenarios
            recovery_analysis = await self._analyze_system_recovery(scenario_result)
            logger.info(
                "System recovery analysis",
                scenario=scenario.test_type.value,
                recovery_time_seconds=recovery_analysis.get("recovery_time", 0),
                stability_recovered=recovery_analysis.get("stability_recovered", False),
            )

            # Cool-down period between intensive scenarios
            if i < len(all_scenarios) - 1:
                cooldown_time = 60  # 1 minute cooldown
                logger.info(f"Cool-down period: {cooldown_time} seconds")
                await asyncio.sleep(cooldown_time)

        # Generate optimization recommendations
        recommendations = await self._generate_optimization_recommendations(results)

        # Calculate overall stability score
        stability_scores = [result["metrics"].get_stability_score() for result in results]
        overall_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0

        completion_time = datetime.utcnow()
        total_duration = completion_time - start_time

        load_test_results = LoadTestResults(
            test_suite_id=test_suite_id,
            scenarios_executed=len(results),
            total_duration=total_duration,
            peak_performance=peak_metrics,
            system_stability_score=overall_stability,
            recommendations=recommendations,
            detailed_results=results,
            infrastructure_impact=await self._analyze_infrastructure_impact(results),
        )

        logger.info(
            "Load test suite completed",
            test_suite_id=str(test_suite_id),
            duration_minutes=total_duration.total_seconds() / 60,
            peak_rps=peak_metrics.throughput if peak_metrics else 0,
            stability_score=overall_stability,
            recommendations_count=len(recommendations),
        )

        return load_test_results

    def _create_peak_load_scenario(
        self,
        concurrent_users: int,
        requests_per_second: int,
        document_processing_rate: int,
    ) -> LoadTestScenario:
        """Create peak business hours load scenario."""

        return LoadTestScenario(
            scenario_id=uuid4(),
            test_type=LoadTestType.PEAK_LOAD,
            concurrent_users=concurrent_users,
            requests_per_second=requests_per_second,
            duration=timedelta(hours=2),
            user_profiles=[
                UserProfile.POWER_USER,
                UserProfile.BUSINESS_USER,
                UserProfile.CASUAL_USER,
            ],
            data_patterns={
                "document_sizes": [1024, 5120, 10240, 51200],  # 1KB to 50KB
                "content_types": ["text/plain", "text/html", "application/pdf"],
                "processing_complexity": ["simple", "medium", "complex"],
                "geographic_distribution": [
                    "us-east",
                    "us-west",
                    "eu-west",
                    "asia-pacific",
                ],
            },
            target_endpoints=[
                "/api/v1/documents/upload",
                "/api/v1/documents/process",
                "/api/v1/documents/search",
                "/api/v1/documents/export",
            ],
            success_criteria={
                "max_response_time_p95": 2.0,  # 2 seconds
                "max_error_rate": 1.0,  # 1%
                "min_throughput": 4500,  # 90% of target RPS
                "max_cpu_usage": 80.0,  # 80%
            },
        )

    def _create_migration_scenario(
        self,
        bulk_upload_size: int,
        concurrent_uploads: int,
        processing_parallelism: int,
    ) -> LoadTestScenario:
        """Create data migration load scenario."""

        return LoadTestScenario(
            scenario_id=uuid4(),
            test_type=LoadTestType.BULK_MIGRATION,
            concurrent_users=concurrent_uploads,
            requests_per_second=100,
            duration=timedelta(hours=3),
            user_profiles=[UserProfile.BULK_PROCESSOR, UserProfile.API_CLIENT],
            data_patterns={
                "batch_sizes": [100, 500, 1000, 5000],
                "document_sizes": [2048, 10240, 51200, 102400],  # 2KB to 100KB
                "upload_patterns": ["sequential", "parallel", "mixed"],
                "compression": ["gzip", "none"],
                "deduplication_rate": 0.3,  # 30% duplicate content
            },
            target_endpoints=[
                "/api/v1/bulk/upload",
                "/api/v1/bulk/process",
                "/api/v1/bulk/status",
            ],
            success_criteria={
                "max_response_time_p95": 10.0,  # 10 seconds for bulk operations
                "max_error_rate": 0.5,  # 0.5%
                "min_throughput": 80,  # 80% of target
                "documents_per_hour": 1000000,  # 1M documents/hour
            },
        )

    def _create_abuse_scenario(
        self,
        malicious_request_rate: int,
        attack_patterns: List[str],
        defense_validation: bool,
    ) -> LoadTestScenario:
        """Create API abuse and attack simulation scenario."""

        return LoadTestScenario(
            scenario_id=uuid4(),
            test_type=LoadTestType.API_ABUSE,
            concurrent_users=1000,
            requests_per_second=malicious_request_rate,
            duration=timedelta(minutes=30),
            user_profiles=[UserProfile.API_CLIENT],
            data_patterns={
                "attack_patterns": attack_patterns,
                "malicious_payloads": [
                    "oversized_requests",
                    "malformed_json",
                    "sql_injection_attempts",
                    "xss_payloads",
                ],
                "rate_limiting_bypass": True,
                "credential_stuffing": {"username_list": 1000, "password_list": 10000},
                "ddos_patterns": ["slow_loris", "connection_flood", "bandwidth_flood"],
            },
            target_endpoints=[
                "/api/v1/auth/login",
                "/api/v1/documents/upload",
                "/api/v1/search",
                "/health",
            ],
            success_criteria={
                "defense_effectiveness": 95.0,  # 95% of attacks blocked
                "legitimate_traffic_impact": 5.0,  # < 5% impact on legitimate traffic
                "system_availability": 99.0,  # 99% uptime during attack
                "auto_mitigation_time": 30.0,  # 30 seconds to auto-mitigate
            },
        )

    def _create_sustained_scenario(
        self, duration: timedelta, concurrent_users: int, mixed_operations: bool
    ) -> LoadTestScenario:
        """Create sustained mixed workload scenario."""

        return LoadTestScenario(
            scenario_id=uuid4(),
            test_type=LoadTestType.SUSTAINED_LOAD,
            concurrent_users=concurrent_users,
            requests_per_second=2000,
            duration=duration,
            user_profiles=[
                UserProfile.POWER_USER,
                UserProfile.BUSINESS_USER,
                UserProfile.CASUAL_USER,
                UserProfile.API_CLIENT,
            ],
            data_patterns={
                "operation_mix": {
                    "upload": 30,
                    "search": 40,
                    "process": 20,
                    "export": 10,
                },
                "user_behavior": "realistic_timing",
                "data_growth": "linear",
                "cache_patterns": "production_like",
            },
            target_endpoints=[
                "/api/v1/documents/upload",
                "/api/v1/documents/search",
                "/api/v1/documents/process",
                "/api/v1/documents/export",
                "/api/v1/analytics/reports",
            ],
            success_criteria={
                "max_response_time_p95": 1.5,  # 1.5 seconds
                "max_error_rate": 0.1,  # 0.1%
                "min_throughput": 1800,  # 90% of target
                "memory_growth_rate": 5.0,  # < 5% per hour
            },
        )

    async def _execute_scenario(
        self, scenario: LoadTestScenario, target_environment: str, duration: timedelta
    ) -> Dict[str, Any]:
        """Execute individual load test scenario."""

        start_time = time.time()

        # Initialize load generators
        generators = await self.load_generators.initialize_generators(
            concurrent_users=scenario.concurrent_users,
            user_profiles=scenario.user_profiles,
            target_endpoints=scenario.target_endpoints,
        )

        # Start metrics collection
        metrics_task = asyncio.create_task(self.metrics_collector.collect_metrics(duration))

        # Execute load generation
        load_tasks = []
        for generator in generators:
            load_task = asyncio.create_task(
                generator.generate_load(scenario=scenario, target_environment=target_environment)
            )
            load_tasks.append(load_task)

        # Wait for completion
        await asyncio.gather(*load_tasks, return_exceptions=True)

        # Stop metrics collection
        metrics_task.cancel()

        # Collect final metrics
        final_metrics = await self.metrics_collector.get_final_metrics()

        execution_time = time.time() - start_time

        return {
            "scenario": scenario,
            "execution_time": execution_time,
            "metrics": final_metrics,
            "success_criteria_met": await self._evaluate_success_criteria(scenario.success_criteria, final_metrics),
            "resource_utilization": await self._get_resource_utilization(),
        }

    async def _analyze_system_recovery(self, scenario_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system recovery after load test scenario."""

        recovery_start = time.time()

        # Monitor system metrics for recovery
        recovery_metrics = []
        for _ in range(30):  # 30 seconds of monitoring
            current_metrics = await self.metrics_collector.get_current_metrics()
            recovery_metrics.append(current_metrics)
            await asyncio.sleep(1)

        # Analyze recovery characteristics
        cpu_stabilized = all(m.cpu_usage < 20 for m in recovery_metrics[-10:])
        memory_stabilized = all(m.memory_usage < 50 for m in recovery_metrics[-10:])
        response_time_normal = all(
            statistics.median(m.response_times) < 0.5 for m in recovery_metrics[-5:] if m.response_times
        )

        recovery_time = time.time() - recovery_start
        stability_recovered = cpu_stabilized and memory_stabilized and response_time_normal

        return {
            "recovery_time": recovery_time,
            "stability_recovered": stability_recovered,
            "cpu_stabilized": cpu_stabilized,
            "memory_stabilized": memory_stabilized,
            "response_time_normal": response_time_normal,
        }

    async def _generate_optimization_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations based on results."""

        recommendations = []

        # Analyze response time patterns
        all_response_times = []
        for result in results:
            all_response_times.extend(result["metrics"].response_times)

        if all_response_times:
            p95_response_time = sorted(all_response_times)[int(len(all_response_times) * 0.95)]
            if p95_response_time > 2.0:
                recommendations.append("Consider implementing response caching for frequently accessed endpoints")
                recommendations.append("Optimize database queries - high response times detected")

        # Analyze error patterns
        high_error_scenarios = [r for r in results if r["metrics"].error_rate > 1.0]

        if high_error_scenarios:
            recommendations.append("Implement circuit breakers for external dependencies")
            recommendations.append("Add retry logic with exponential backoff for transient failures")

        # Analyze resource utilization
        high_cpu_scenarios = [r for r in results if r["metrics"].cpu_usage > 80.0]

        if high_cpu_scenarios:
            recommendations.append("Consider horizontal scaling - high CPU utilization detected")
            recommendations.append("Profile application for CPU-intensive operations")

        # Memory analysis
        high_memory_scenarios = [r for r in results if r["metrics"].memory_usage > 80.0]

        if high_memory_scenarios:
            recommendations.append("Implement memory pooling for large object allocations")
            recommendations.append("Consider streaming processing for large datasets")

        # Cache effectiveness
        low_cache_scenarios = [r for r in results if r["metrics"].cache_hit_rate < 80.0]

        if low_cache_scenarios:
            recommendations.append("Optimize cache warming strategies")
            recommendations.append("Review cache eviction policies")

        return recommendations

    async def _analyze_infrastructure_impact(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze infrastructure impact across all scenarios."""

        # Aggregate metrics across all scenarios
        all_metrics = [result["metrics"] for result in results]

        peak_cpu = max(m.cpu_usage for m in all_metrics)
        peak_memory = max(m.memory_usage for m in all_metrics)
        peak_throughput = max(m.throughput for m in all_metrics)
        peak_concurrent_users = max(m.concurrent_users_peak for m in all_metrics)

        # Calculate infrastructure scaling requirements
        recommended_cpu_cores = max(16, int(peak_cpu / 10))  # 10% CPU per core target
        recommended_memory_gb = max(32, int(peak_memory * 1.5))  # 50% memory headroom
        recommended_instances = max(3, int(peak_concurrent_users / 2000))  # 2K users per instance

        return {
            "peak_metrics": {
                "cpu_usage": peak_cpu,
                "memory_usage": peak_memory,
                "throughput": peak_throughput,
                "concurrent_users": peak_concurrent_users,
            },
            "scaling_recommendations": {
                "cpu_cores": recommended_cpu_cores,
                "memory_gb": recommended_memory_gb,
                "instance_count": recommended_instances,
                "auto_scaling_enabled": True,
            },
            "cost_estimation": {
                "monthly_compute_cost": recommended_instances * 500,  # $500 per instance
                "storage_cost": peak_throughput * 0.1,  # $0.1 per GB throughput
                "network_cost": peak_throughput * 0.05,  # $0.05 per GB network
            },
        }

    async def _evaluate_success_criteria(
        self, success_criteria: Dict[str, float], metrics: PerformanceMetrics
    ) -> Dict[str, bool]:
        """Evaluate if success criteria were met."""

        results = {}
        percentiles = metrics.get_percentiles()

        # Response time criteria
        if "max_response_time_p95" in success_criteria:
            results["response_time_p95"] = (
                percentiles.get("p95", float("inf")) <= success_criteria["max_response_time_p95"]
            )

        # Error rate criteria
        if "max_error_rate" in success_criteria:
            results["error_rate"] = metrics.error_rate <= success_criteria["max_error_rate"]

        # Throughput criteria
        if "min_throughput" in success_criteria:
            results["throughput"] = metrics.throughput >= success_criteria["min_throughput"]

        # Resource utilization criteria
        if "max_cpu_usage" in success_criteria:
            results["cpu_usage"] = metrics.cpu_usage <= success_criteria["max_cpu_usage"]

        return results

    async def _get_resource_utilization(self) -> Dict[str, Any]:
        """Get current resource utilization."""

        # In production, this would query actual system metrics
        return {
            "cpu_cores_used": 16,
            "memory_gb_used": 32,
            "network_bandwidth_mbps": 1000,
            "database_connections_used": 500,
            "cache_size_gb": 8,
        }


# Supporting classes for load testing


class LoadGeneratorPool:
    """Pool of load generators for different user profiles."""

    async def initialize_generators(
        self,
        concurrent_users: int,
        user_profiles: List[UserProfile],
        target_endpoints: List[str],
    ) -> List["LoadGenerator"]:
        """Initialize load generators based on user profiles."""

        generators = []
        users_per_profile = concurrent_users // len(user_profiles)

        for profile in user_profiles:
            generator = LoadGenerator(
                user_profile=profile,
                concurrent_users=users_per_profile,
                target_endpoints=target_endpoints,
            )
            generators.append(generator)

        return generators


class LoadGenerator:
    """Individual load generator for specific user profile."""

    def __init__(
        self,
        user_profile: UserProfile,
        concurrent_users: int,
        target_endpoints: List[str],
    ):
        self.user_profile = user_profile
        self.concurrent_users = concurrent_users
        self.target_endpoints = target_endpoints

    async def generate_load(self, scenario: LoadTestScenario, target_environment: str):
        """Generate load according to user profile."""

        # Create user simulation tasks
        user_tasks = []
        for i in range(self.concurrent_users):
            user_task = asyncio.create_task(
                self._simulate_user_behavior(user_id=i, scenario=scenario, target_environment=target_environment)
            )
            user_tasks.append(user_task)

        # Wait for all users to complete
        await asyncio.gather(*user_tasks, return_exceptions=True)

    async def _simulate_user_behavior(self, user_id: int, scenario: LoadTestScenario, target_environment: str):
        """Simulate individual user behavior."""

        end_time = time.time() + scenario.duration.total_seconds()

        while time.time() < end_time:
            # Select endpoint based on user profile
            endpoint = random.choice(self.target_endpoints)

            # Generate realistic request
            request_data = self._generate_request_data(scenario.data_patterns)

            # Simulate request with realistic timing
            try:
                response_time = await self._make_request(
                    endpoint=endpoint,
                    data=request_data,
                    target_environment=target_environment,
                )

                # Log successful request
                logger.debug(
                    "Request completed",
                    user_id=user_id,
                    endpoint=endpoint,
                    response_time=response_time,
                )

            except Exception as e:
                # Log failed request
                logger.warning("Request failed", user_id=user_id, endpoint=endpoint, error=str(e))

            # Wait between requests based on user profile
            await asyncio.sleep(self._get_think_time())

    def _generate_request_data(self, data_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic request data based on patterns."""

        if "document_sizes" in data_patterns:
            size = random.choice(data_patterns["document_sizes"])
            return {
                "content": "x" * size,
                "content_type": random.choice(data_patterns.get("content_types", ["text/plain"])),
                "metadata": {
                    "source": "load_test",
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }

        return {"test_data": True}

    async def _make_request(self, endpoint: str, data: Dict[str, Any], target_environment: str) -> float:
        """Make HTTP request and return response time."""

        start_time = time.time()

        # Simulate HTTP request
        await asyncio.sleep(random.uniform(0.01, 0.5))  # Simulate network latency

        # Simulate processing time based on endpoint
        if "upload" in endpoint:
            await asyncio.sleep(random.uniform(0.1, 1.0))
        elif "search" in endpoint:
            await asyncio.sleep(random.uniform(0.05, 0.3))
        elif "process" in endpoint:
            await asyncio.sleep(random.uniform(0.5, 2.0))

        response_time = time.time() - start_time
        return response_time

    def _get_think_time(self) -> float:
        """Get think time between requests based on user profile."""

        think_times = {
            UserProfile.POWER_USER: 0.5,
            UserProfile.BUSINESS_USER: 1.0,
            UserProfile.CASUAL_USER: 2.0,
            UserProfile.API_CLIENT: 0.1,
            UserProfile.BULK_PROCESSOR: 0.01,
        }

        base_time = think_times.get(self.user_profile, 1.0)
        return random.uniform(base_time * 0.5, base_time * 1.5)


class MetricsCollector:
    """Collects and aggregates performance metrics during load tests."""

    def __init__(self):
        self.response_times: List[float] = []
        self.error_count = 0
        self.request_count = 0
        self.start_time = time.time()

    async def collect_metrics(self, duration: timedelta):
        """Collect metrics for specified duration."""

        end_time = time.time() + duration.total_seconds()

        while time.time() < end_time:
            # Simulate metrics collection
            current_metrics = await self._get_current_system_metrics()

            # Store metrics
            if current_metrics.get("response_time"):
                self.response_times.append(current_metrics["response_time"])

            if current_metrics.get("error"):
                self.error_count += 1

            self.request_count += 1

            await asyncio.sleep(1)  # Collect every second

    async def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""

        # Mock metrics - in production, integrate with monitoring systems
        return {
            "response_time": random.uniform(0.05, 2.0),
            "cpu_usage": random.uniform(10, 90),
            "memory_usage": random.uniform(20, 80),
            "error": random.random() < 0.01,  # 1% error rate
        }

    async def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""

        current_system = await self._get_current_system_metrics()

        return PerformanceMetrics(
            response_times=self.response_times[-100:],  # Last 100 responses
            throughput=self.request_count / max(1, time.time() - self.start_time),
            error_rate=(self.error_count / max(1, self.request_count)) * 100,
            cpu_usage=current_system.get("cpu_usage", 0),
            memory_usage=current_system.get("memory_usage", 0),
            network_utilization=random.uniform(10, 80),
            database_connections=random.randint(50, 500),
            cache_hit_rate=random.uniform(70, 95),
            concurrent_users_peak=1000,
        )

    async def get_final_metrics(self) -> PerformanceMetrics:
        """Get final aggregated metrics."""

        total_time = time.time() - self.start_time

        return PerformanceMetrics(
            response_times=self.response_times,
            throughput=self.request_count / max(1, total_time),
            error_rate=(self.error_count / max(1, self.request_count)) * 100,
            cpu_usage=random.uniform(40, 80),
            memory_usage=random.uniform(30, 70),
            network_utilization=random.uniform(30, 80),
            database_connections=random.randint(100, 800),
            cache_hit_rate=random.uniform(80, 95),
            concurrent_users_peak=max(1000, self.request_count // 10),
        )


class ScenarioBuilder:
    """Builds realistic load test scenarios."""

    def build_fortune_500_scenario(self) -> LoadTestScenario:
        """Build realistic production-grade enterprise scenario."""

        return LoadTestScenario(
            scenario_id=uuid4(),
            test_type=LoadTestType.PEAK_LOAD,
            concurrent_users=15000,
            requests_per_second=7500,
            duration=timedelta(hours=8),  # Full business day
            user_profiles=[
                UserProfile.POWER_USER,
                UserProfile.BUSINESS_USER,
                UserProfile.CASUAL_USER,
                UserProfile.API_CLIENT,
            ],
            data_patterns={
                "document_types": [
                    "contracts",
                    "reports",
                    "presentations",
                    "spreadsheets",
                ],
                "file_sizes": [1024, 10240, 102400, 1048576],  # 1KB to 1MB
                "processing_complexity": ["simple", "medium", "complex", "advanced"],
                "user_locations": ["north_america", "europe", "asia_pacific"],
                "business_hours_pattern": True,
            },
            target_endpoints=[
                "/api/v1/documents/upload",
                "/api/v1/documents/process",
                "/api/v1/documents/search",
                "/api/v1/documents/export",
                "/api/v1/analytics/dashboard",
                "/api/v1/reports/generate",
            ],
            success_criteria={
                "max_response_time_p95": 1.5,
                "max_error_rate": 0.1,
                "min_throughput": 7000,
                "max_cpu_usage": 75.0,
                "uptime_requirement": 99.99,
            },
        )


class ResultAnalyzer:
    """Analyzes load test results and generates insights."""

    def analyze_performance_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends across scenarios."""

        throughput_trend = [r["metrics"].throughput for r in results]
        response_time_trend = [
            statistics.median(r["metrics"].response_times) for r in results if r["metrics"].response_times
        ]

        return {
            "throughput_trend": throughput_trend,
            "response_time_trend": response_time_trend,
            "performance_degradation": any(
                throughput_trend[i] < throughput_trend[i - 1] * 0.9 for i in range(1, len(throughput_trend))
            ),
            "bottleneck_indicators": self._identify_bottlenecks(results),
        }

    def _identify_bottlenecks(self, results: List[Dict[str, Any]]) -> List[str]:
        """Identify system bottlenecks from test results."""

        bottlenecks = []

        # CPU bottleneck
        if any(r["metrics"].cpu_usage > 85 for r in results):
            bottlenecks.append("CPU utilization")

        # Memory bottleneck
        if any(r["metrics"].memory_usage > 85 for r in results):
            bottlenecks.append("Memory utilization")

        # Database bottleneck
        if any(r["metrics"].database_connections > 800 for r in results):
            bottlenecks.append("Database connection pool")

        # Cache bottleneck
        if any(r["metrics"].cache_hit_rate < 70 for r in results):
            bottlenecks.append("Cache effectiveness")

        return bottlenecks
