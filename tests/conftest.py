"""
Comprehensive test configuration for QuarryCore.

This module provides fixtures and utilities for testing all components
with proper isolation, mocking, and performance validation.
"""

# Standard library imports
import asyncio
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Generator, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

# Third-party imports
import httpx
import numpy as np
import psutil
import pytest
import pytest_asyncio

# Local imports
from quarrycore.config import Config, SQLiteConfig
from quarrycore.protocols import (
    ContentMetadata,
    CrawlResult,
    DomainType,
    DuplicationResult,
    ExtractedContent,
    HardwareCapabilities,
    HardwareType,
    QualityScore,
)
from quarrycore.storage import SQLiteManager

# Set test mode to prevent metric conflicts
os.environ["QUARRY_TEST_MODE"] = "1"

# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    os.environ["QUARRY_TEST_MODE"] = "1"
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests across modules")
    config.addinivalue_line("markers", "performance: Performance and load tests")
    config.addinivalue_line("markers", "security: Security and vulnerability tests")
    config.addinivalue_line("markers", "slow: Tests that take >10 seconds")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU hardware")
    config.addinivalue_line("markers", "network: Tests requiring network access")
    config.addinivalue_line("markers", "chaos: Chaos engineering tests")


# ============================================================================
# Core Test Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def event_loop_policy():
    """Set the asyncio event loop policy for the test session."""
    # pytest-asyncio handles this by default, but explicit is fine
    return asyncio.get_event_loop_policy()


@pytest_asyncio.fixture(autouse=True)
async def cleanup_tasks() -> AsyncGenerator[None, None]:
    """
    An aggressive auto-use fixture to ensure all created asyncio tasks
    are properly cancelled after a test, preventing resource leaks and
    the kind of hangs seen in test failures.
    """
    tasks_before = asyncio.all_tasks()
    yield
    tasks_after = asyncio.all_tasks()
    new_tasks = tasks_after - tasks_before

    for task in new_tasks:
        if not task.done():
            task.cancel()
            try:
                # Await the task to allow it to process the cancellation.
                # Suppress the expected CancelledError.
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                # Log any other exceptions that occur during cleanup.
                print(f"Unexpected error during task cleanup: {e}")


# ============================================================================
# Temporary Directory and File Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_data_dir():
    """Provide test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_html():
    """Provide sample HTML content for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article</title>
        <meta name="description" content="Sample article for testing">
        <meta property="og:title" content="Test Article">
        <meta property="og:description" content="Sample article description">
    </head>
    <body>
        <article>
            <h1>Test Article Title</h1>
            <p>This is a sample paragraph with <strong>bold text</strong> and
               <a href="https://example.com">a link</a>.</p>
            <table>
                <tr><th>Column 1</th><th>Column 2</th></tr>
                <tr><td>Data 1</td><td>Data 2</td></tr>
            </table>
            <pre><code class="python">
def hello_world():
    print("Hello, World!")
            </code></pre>
        </article>
    </body>
    </html>
    """


# ============================================================================
# Configuration Fixtures
# ============================================================================


@pytest.fixture
def test_config(temp_dir):
    """Provide test configuration."""
    config = Config()

    # Override paths to use temp directory
    config.storage.hot.db_path = temp_dir / "test.db"
    config.storage.warm.base_path = temp_dir / "parquet"
    config.storage.retention.cold_storage_path = temp_dir / "cold"
    config.storage.backup.path = temp_dir / "backups"
    config.monitoring.log_file = temp_dir / "test.log"

    # Test-friendly settings
    config.crawler.concurrent_requests = 2
    config.monitoring.enabled = False  # Disable for most tests
    config.debug.test_mode = True
    config.debug.max_urls_to_process = 100

    return config


@pytest.fixture
def pi_config(test_config):
    """Configuration optimized for Raspberry Pi."""
    config = test_config
    config.crawler.concurrent_requests = 1
    config.quality.min_content_length = 50
    return config


@pytest.fixture
def workstation_config(test_config):
    """Configuration optimized for workstation."""
    config = test_config
    config.crawler.concurrent_requests = 20
    config.quality.min_content_length = 100
    return config


# ============================================================================
# Hardware Capability Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def hardware_caps_pi():
    """Hardware capabilities for a Raspberry Pi."""
    return HardwareCapabilities(
        hardware_type=HardwareType.RASPBERRY_PI,
        cpu_cores=4,
        total_memory_gb=4.0,
        available_memory_gb=3.0,
        has_gpu=False,
    )


@pytest.fixture(scope="module")
def hardware_caps_workstation():
    """Hardware capabilities for a workstation."""
    return HardwareCapabilities(
        hardware_type=HardwareType.WORKSTATION,
        cpu_cores=16,
        total_memory_gb=64.0,
        available_memory_gb=48.0,
        has_gpu=True,
        gpu_memory_gb=12.0,
    )


@pytest.fixture
def server_hardware():
    """Server hardware capabilities."""
    return HardwareCapabilities(
        hardware_type=HardwareType.SERVER,
        cpu_cores=64,
        total_memory_gb=128.0,
        available_memory_gb=96.0,
        has_gpu=True,
        gpu_memory_gb=24.0,
        storage_available_gb=10000.0,
        network_bandwidth_mbps=10000.0,
    )


# ============================================================================
# Protocol Data Fixtures
# ============================================================================


@pytest.fixture
def sample_crawl_result():
    """Provide sample crawl result."""
    return CrawlResult(
        url="https://example.com/test-article",
        final_url="https://example.com/test-article",
        content=b"<html><body><h1>Test</h1><p>Content</p></body></html>",
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content_encoding="utf-8",
        is_valid=True,
        robots_allowed=True,
        warnings=[],
        errors=[],
    )


@pytest.fixture
def sample_extracted_content():
    """Provide sample extracted content."""
    return ExtractedContent(
        title="Test Article",
        text="This is test content for validation.",
        language="en",
        word_count=7,
        tables=[],
        images=[],
        links=[],
        code_blocks=[],
        confidence_score=0.95,
        extraction_method="trafilatura",
    )


@pytest.fixture
def sample_metadata():
    """Provide sample content metadata."""
    return ContentMetadata(
        url="https://example.com/test-article",
        title="Test Article",
        description="Test article description",
        author="Test Author",
        published_date=None,
        domain="example.com",
        domain_type=DomainType.GENERAL,
        schema_data={},
        social_shares={},
    )


@pytest.fixture
def sample_quality_score():
    """Provide sample quality score."""
    return QualityScore(
        overall_score=0.85,
        confidence=0.9,
        grammar_score=0.90,
        readability_score=0.80,
        coherence_score=0.85,
        information_density=0.75,
        domain_relevance=0.95,
        bias_score=0.1,
        toxicity_score=0.05,
    )


@pytest.fixture
def sample_dedup_result():
    """Provide sample deduplication result."""
    return DuplicationResult(
        is_duplicate=False,
        duplicate_type="",
        jaccard_similarity=0.0,
        semantic_similarity=0.0,
        processing_time_ms=50.0,
    )


@pytest.fixture
def sample_document():
    """Provide a sample document combining various components for testing."""
    return {
        "crawl_result": CrawlResult(
            url="https://example.com/test-article",
            final_url="https://example.com/test-article",
            content=b"<html><body><h1>Test</h1><p>Content for testing.</p></body></html>",
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            content_encoding="utf-8",
            is_valid=True,
            robots_allowed=True,
            warnings=[],
            errors=[],
        ),
        "extracted_content": ExtractedContent(
            title="Test Article",
            text="This is test content for validation.",
            language="en",
            word_count=7,
            tables=[],
            images=[],
            links=[],
            code_blocks=[],
            confidence_score=0.95,
            extraction_method="trafilatura",
        ),
        "metadata": ContentMetadata(
            url="https://example.com/test-article",
            title="Test Article",
            description="Test article description",
            author="Test Author",
            published_date=None,
            domain="example.com",
            domain_type=DomainType.GENERAL,
            schema_data={},
            social_shares={},
        ),
        "quality_score": QualityScore(
            overall_score=0.85,
            confidence=0.9,
            grammar_score=0.90,
            readability_score=0.80,
            coherence_score=0.85,
            information_density=0.75,
            domain_relevance=0.95,
            bias_score=0.1,
            toxicity_score=0.05,
        ),
    }


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for testing."""
    mock_client = AsyncMock()

    # Default successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.content = b"<html><body>Test</body></html>"
    mock_response.text = "<html><body>Test</body></html>"
    mock_response.url = "https://example.com"

    mock_client.get.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for crawler testing with async context manager support."""
    mock_client = AsyncMock()

    # Default successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html; charset=utf-8"}
    mock_response.content = b"<html><body><h1>Test Article</h1><p>This is test content.</p></body></html>"
    mock_response.text = "<html><body><h1>Test Article</h1><p>This is test content.</p></body></html>"
    mock_response.url = "https://example.com"
    mock_response.is_error = False
    mock_response.is_redirect = False

    # Make get method async
    async def async_get(*args, **kwargs):
        await asyncio.sleep(0.001)  # Simulate network delay
        return mock_response

    mock_client.get = async_get

    # Add async context manager support
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Add other HTTP methods if needed
    mock_client.head = async_get
    mock_client.post = async_get

    return mock_client


@pytest.fixture
def mock_gpu():
    """Mock GPU operations for testing."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.get_device_properties") as mock_props,
    ):
        mock_props.return_value = MagicMock(
            name="NVIDIA GeForce RTX 4090",
            total_memory=24 * 1024**3,  # 24GB
            major=8,
            minor=9,
        )
        yield


@pytest.fixture
def mock_sentence_transformers():
    """Mock sentence transformers for testing."""
    with patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_st.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_faiss():
    """Mock FAISS for testing."""
    with patch("faiss.IndexFlatIP") as mock_index:
        mock_idx = MagicMock()
        mock_idx.ntotal = 0
        mock_idx.d = 384
        mock_idx.search.return_value = (
            np.array([[0.95, 0.80]], dtype=np.float32),  # distances
            np.array([[0, 1]], dtype=np.int64),  # indices
        )
        mock_index.return_value = mock_idx
        yield mock_idx


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def test_database(temp_dir):
    """Provide test database with sample data."""
    db_path = temp_dir / "test.db"
    config = SQLiteConfig(db_path=db_path)
    manager = SQLiteManager(config)

    await manager.initialize()

    # Insert sample data
    await manager.store_batch(
        [
            {
                "content_id": str(uuid4()),
                "url": "https://example.com/1",
                "title": "Test Article 1",
                "content_hash": "hash1",
                "domain": "example.com",
                "quality_score": 0.85,
                "parquet_path": str(temp_dir / "sample.parquet"),
            }
        ]
    )

    yield manager
    await manager.close()


# ============================================================================
# Performance Testing Fixtures
# ============================================================================


@pytest.fixture
def performance_dataset():
    """Generate performance testing dataset."""
    urls = [f"https://example.com/article-{i}" for i in range(1000)]
    return urls


@pytest.fixture
def large_content_sample():
    """Generate large content sample for stress testing."""
    # 10KB of text content
    content = "This is a sample paragraph. " * 400
    return content


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    memory_usage = []
    stop_monitoring = threading.Event()

    def monitor():
        process = psutil.Process()
        while not stop_monitoring.is_set():
            memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
            time.sleep(0.1)

    thread = threading.Thread(target=monitor)
    thread.start()

    yield memory_usage

    stop_monitoring.set()
    thread.join()


# ============================================================================
# Chaos Engineering Fixtures
# ============================================================================


@pytest.fixture
def chaos_config():
    """Configuration for chaos engineering tests."""
    return {
        "failure_rate": 0.1,  # 10% failure rate
        "network_delay_ms": 100,
        "memory_pressure_mb": 100,
        "disk_full_threshold": 0.95,
        "cpu_stress_duration": 5.0,
    }


@pytest.fixture
def network_chaos():
    """Simulate network failures."""

    class NetworkChaos:
        def __init__(self):
            self.should_fail = False
            self.delay_ms = 0

        async def maybe_fail(self):
            if self.should_fail:
                raise httpx.ConnectTimeout("Chaos: Network failure")
            if self.delay_ms > 0:
                await asyncio.sleep(self.delay_ms / 1000)

    return NetworkChaos()


# ============================================================================
# Domain-Specific Test Data
# ============================================================================


@pytest.fixture
def medical_content():
    """Medical domain test content."""
    return {
        "html": """
        <article>
            <h1>Diabetes Management Guidelines</h1>
            <p>Type 2 diabetes mellitus affects approximately 422 million people worldwide.
               Treatment typically involves metformin as first-line therapy, with HbA1c
               targets of <7% for most adults.</p>
            <p>Contraindications include severe renal impairment (eGFR <30 mL/min/1.73m²)
               and metabolic acidosis.</p>
        </article>
        """,
        "expected_entities": ["diabetes", "metformin", "HbA1c"],
        "quality_threshold": 0.9,
    }


@pytest.fixture
def legal_content():
    """Legal domain test content."""
    return {
        "html": """
        <article>
            <h1>Contract Law Principles</h1>
            <p>In Smith v. Jones (2023) UKSC 45, the Supreme Court held that
               consideration must be sufficient but need not be adequate.</p>
            <p>This principle, established in Chappell & Co Ltd v Nestlé Co Ltd [1960] AC 87,
               remains fundamental to contract formation under English law.</p>
        </article>
        """,
        "expected_citations": [
            "Smith v. Jones (2023) UKSC 45",
            "Chappell & Co Ltd v Nestlé Co Ltd [1960] AC 87",
        ],
        "quality_threshold": 0.95,
    }


@pytest.fixture
def technical_content():
    """Technical domain test content."""
    return {
        "html": """
        <article>
            <h1>Async Python Patterns</h1>
            <p>Modern Python applications leverage asyncio for concurrent I/O operations.</p>
            <pre><code class="python">
async def fetch_data(session, url):
    async with session.get(url) as response:
        return await response.json()
            </code></pre>
            <p>This pattern ensures proper resource cleanup and exception handling.</p>
        </article>
        """,
        "expected_code_blocks": 1,
        "quality_threshold": 0.85,
    }


@pytest.fixture
def ecommerce_content():
    """E-commerce domain test content."""
    return {
        "html": """
        <article>
            <h1>Premium Wireless Headphones</h1>
            <p>Price: $299.99 (was $399.99) - Save 25%!</p>
            <ul>
                <li>Active Noise Cancellation</li>
                <li>30-hour battery life</li>
                <li>Bluetooth 5.0 connectivity</li>
            </ul>
            <p>Rating: 4.8/5 stars (2,847 reviews)</p>
        </article>
        """,
        "expected_price": 299.99,
        "expected_rating": 4.8,
        "quality_threshold": 0.80,
    }


# ============================================================================
# HTTP Client and Container Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def fresh_container(temp_dir):
    """Create a fresh dependency container with test configuration."""
    from quarrycore.config.config import Config
    from quarrycore.container import DependencyContainer

    # Create test config
    config = Config()
    config.storage.hot.db_path = temp_dir / "test.db"
    config.storage.warm.base_path = temp_dir / "parquet"
    config.storage.retention.cold_storage_path = temp_dir / "cold"
    config.storage.backup.path = temp_dir / "backups"
    config.monitoring.log_file = temp_dir / "test.log"
    config.monitoring.prometheus_port = None  # Disable prometheus for testing
    config.crawler.concurrent_requests = 2
    config.debug.test_mode = True
    config.debug.max_urls_to_process = 100

    container = DependencyContainer()
    container.config = config
    await container.initialize()

    yield container

    await container.shutdown()


@pytest_asyncio.fixture
async def http_client_with_config(temp_dir):
    """Create HTTP client with test configuration."""
    from quarrycore.config.config import Config
    from quarrycore.crawler.http_client import HttpClient

    config = Config()
    config.crawler.max_retries = 3
    config.crawler.max_concurrency_per_domain = 2
    config.crawler.backoff_cooldown_seconds = 1  # Short for testing
    config.crawler.timeout = 5.0
    config.crawler.respect_robots = True
    config.crawler.user_agent = "TestBot/1.0"

    async with HttpClient(config) as client:
        yield client


@pytest_asyncio.fixture
async def http_client(temp_dir):
    """Create simple HTTP client for testing."""
    from quarrycore.config.config import Config
    from quarrycore.crawler.http_client import HttpClient

    config = Config()
    config.crawler.max_retries = 3
    config.crawler.max_concurrency_per_domain = 2
    config.crawler.backoff_cooldown_seconds = 1  # Short for testing
    config.crawler.timeout = 5.0
    config.crawler.respect_robots = False  # Disable for simpler testing
    config.crawler.user_agent = "TestBot/1.0"
    config.debug.test_mode = True

    async with HttpClient(config) as client:
        yield client


@pytest.fixture(autouse=True)
def metrics_reset():
    """Reset metrics between tests using test mode to avoid registry conflicts."""
    import os
    import sys

    # Ensure test mode is enabled for testing
    original_test_mode = os.environ.get("QUARRY_TEST_MODE", "0")
    os.environ["QUARRY_TEST_MODE"] = "1"

    # Import (or reload) the metrics module so that QUARRY_TEST_MODE flag is picked up,
    # but **do not clear the Prometheus REGISTRY** – doing so invalidates existing
    # collector references held by already-imported test modules.  Instead we simply
    # reset the internal values of each collector to guarantee isolation.

    if "quarrycore.observability.metrics" in sys.modules:
        import importlib

        metrics_module = importlib.reload(sys.modules["quarrycore.observability.metrics"])
    else:
        from quarrycore.observability import metrics as metrics_module

    # Zero all metric values (counters, gauges, histograms) so each test starts fresh.
    for _metric in metrics_module.METRICS.values():
        # Counter / Gauge share the _value attribute with an underlying Value class.
        if hasattr(_metric, "_value"):
            try:
                _metric._value.set(0)  # type: ignore[attr-defined]
            except Exception:
                pass

        # Histogram has _sum, _count, and optionally _buckets.
        if hasattr(_metric, "_sum") and hasattr(_metric, "_count"):
            try:
                _metric._sum.set(0)  # type: ignore[attr-defined]
                _metric._count.set(0)  # type: ignore[attr-defined]

                if hasattr(_metric, "_buckets"):
                    for bucket in _metric._buckets:  # type: ignore[attr-defined]
                        bucket.set(0)
            except Exception:
                pass

    yield

    # Restore original test mode
    os.environ["QUARRY_TEST_MODE"] = original_test_mode


@pytest.fixture
def deterministic_jitter():
    """Make backoff jitter deterministic for testing."""
    import random
    from unittest.mock import patch

    # Fixed jitter values for predictable testing
    jitter_values = [1.0, 1.1, 0.9, 1.05, 0.95]
    jitter_index = 0

    def mock_uniform(a, b):
        nonlocal jitter_index
        # Return from fixed sequence
        value = jitter_values[jitter_index % len(jitter_values)]
        jitter_index += 1
        return value

    with patch("random.uniform", side_effect=mock_uniform):
        yield


@pytest.fixture
def mock_robots_txt():
    """Mock robots.txt responses for testing."""
    return {
        "allowed": "User-agent: *\nAllow: /",
        "disallowed": "User-agent: *\nDisallow: /",
        "specific_disallow": "User-agent: *\nDisallow: /private/\nAllow: /",
        "crawl_delay": "User-agent: *\nCrawl-delay: 1\nAllow: /",
    }


# ============================================================================
# Utility Functions
# ============================================================================


def create_test_urls(count: int, domain: str = "example.com") -> List[str]:
    """Create list of test URLs."""
    return [f"https://{domain}/article-{i}" for i in range(count)]


async def wait_for_condition(condition, timeout: float = 10.0, interval: float = 0.1):
    """Wait for a condition to become true."""
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < timeout:
        if await condition() if asyncio.iscoroutinefunction(condition) else condition():
            return True
        await asyncio.sleep(interval)
    return False


def assert_performance_threshold(actual: float, expected: float, tolerance: float = 0.1):
    """Assert performance is within acceptable threshold."""
    assert actual <= expected * (1 + tolerance), f"Performance degraded: {actual} > {expected * (1 + tolerance)}"


# ============================================================================
# Cleanup
# ============================================================================

# The autouse cleanup_tasks fixture above replaces the previous manual
# and flawed cleanup implementations. No further code is needed here.
