# Testing Guide for QuarryCore

This guide covers testing strategies, fixtures, and best practices for QuarryCore's test suite.

## Test Organization

### Test Categories

- **Unit Tests** (`tests/unit/`): Test individual components in isolation
- **Integration Tests** (`tests/integration/`): Test component interactions
- **End-to-End Tests** (`tests/e2e/`): Test complete workflows

### Coverage Requirements

- HTTP Client: ≥ 90% branch coverage
- Dependency Container: ≥ 90% branch coverage  
- Overall project: ≥ 80% coverage

## Key Testing Fixtures

### Container Management

```python
@pytest_asyncio.fixture
async def fresh_container(temp_dir):
    """Create a fresh dependency container with test configuration."""
    config = Config()
    config.storage.hot.db_path = temp_dir / "test.db"
    config.debug.test_mode = True
    
    container = DependencyContainer()
    container.config = config
    await container.initialize()
    
    yield container
    
    await container.shutdown()
```

### HTTP Client Testing

```python
@pytest_asyncio.fixture
async def http_client_with_config(temp_dir):
    """Create HTTP client with test configuration."""
    config = Config()
    config.crawler.max_retries = 3
    config.crawler.max_concurrency_per_domain = 2
    config.debug.test_mode = True  # Enables deterministic behavior
    
    client = HttpClient(config)
    await client.initialize()
    
    yield client
    
    await client.close()
```

### Metrics Validation

```python
from tests.helpers import metric_delta, histogram_observes

# Test counter increments
with metric_delta(METRICS["some_counter"]):
    # Code that should increment counter by 1
    pass

# Test histogram observations
with histogram_observes(METRICS["request_duration"]):
    # Code that should add observations
    pass
```

## Deterministic Testing

### Seeding Strategy

For reproducible test results, QuarryCore uses deterministic seeding:

```python
import os
import random

# Set environment variable for consistent hashing
os.environ["PYTHONHASHSEED"] = "1234"

# Seed random number generator
random.seed(1234)

# Enable test mode for deterministic behavior
config.debug.test_mode = True
```

### Property-Based Testing

QuarryCore uses Hypothesis for property-based testing:

```python
from hypothesis import given, strategies as st, settings

@given(st.integers(min_value=1, max_value=10))
@settings(max_examples=50, derandomize=True)
@pytest.mark.asyncio
async def test_retry_attempts_property(max_retries, http_client):
    """Property test: retry attempts should match configuration."""
    # Test implementation...
```

### Configuration

```python
hypothesis.settings(
    max_examples=50,      # Bounded for CI performance
    derandomize=True      # Reproducible test runs
)
```

## Testing Patterns

### Behavioral Contract Testing

Focus on externally observable behavior rather than implementation details:

```python
@pytest.mark.asyncio
async def test_robots_txt_compliance(http_client):
    """Test robots.txt compliance via public API."""
    with aioresponses() as m:
        m.get("https://example.com/robots.txt", 
              body="User-agent: *\nDisallow: /private")
        
        response = await http_client.fetch("https://example.com/private/page")
        
        # Assert behavior, not implementation
        assert response.status == 999  # Blocked status
```

### Error Handling Validation

```python
@pytest.mark.asyncio
async def test_malformed_url_handling(http_client):
    """Test graceful handling of malformed URLs."""
    malformed_urls = ["not-a-url", "://missing-scheme", ""]
    
    for url in malformed_urls:
        response = await http_client.fetch(url)
        # Should handle gracefully
        assert response.status in (0, 999)
        assert response.body == b""
```

### Concurrent Access Testing

```python
@pytest.mark.asyncio
async def test_concurrent_singleton_access():
    """Test thread-safe singleton access."""
    async def get_instance():
        return await container.get_http_client()
    
    # 100 concurrent accesses
    tasks = [get_instance() for _ in range(100)]
    instances = await asyncio.gather(*tasks)
    
    # All should be same instance
    assert all(instance is instances[0] for instance in instances)
```

## Running Tests

### Local Development

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/quarrycore --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run property-based tests
pytest -k "property" --hypothesis-show-statistics
```

### Parallel Execution

```bash
# Use pytest-xdist for parallel execution
pytest -n auto  # Auto-detect CPU cores
pytest -n 4     # Use 4 workers
```

### Coverage Analysis

```bash
# Generate detailed coverage report
pytest --cov=src/quarrycore --cov-report=html --cov-branch

# View HTML report
open htmlcov/index.html
```

## Best Practices

### Test Design

1. **Test Behavior, Not Implementation**: Assert on public APIs and observable outcomes
2. **Use Descriptive Names**: Test names should explain what is being validated
3. **Arrange-Act-Assert**: Structure tests clearly with setup, execution, and validation
4. **Deterministic Tests**: Use seeding and mocking for reproducible results

### Performance Testing

```python
@pytest.mark.asyncio
async def test_request_timing():
    """Test request completes within reasonable time."""
    start_time = time.time()
    
    response = await http_client.fetch("https://example.com")
    
    elapsed = time.time() - start_time
    assert elapsed < 5.0  # Reasonable timeout
    assert response.status == 200
```

### Error Scenarios

```python
@pytest.mark.asyncio
async def test_network_failure_handling():
    """Test graceful handling of network failures."""
    with aioresponses() as m:
        m.get("https://example.com", exception=aiohttp.ClientError("Network error"))
        
        response = await http_client.fetch("https://example.com")
        
        # Should handle gracefully
        assert response.status == 0
        assert response.attempts > 1  # Should have retried
```

## Fixtures Reference

### Core Fixtures

- `temp_dir`: Temporary directory for test files
- `fresh_container`: Clean dependency container
- `http_client_with_config`: Configured HTTP client
- `metrics_reset`: Reset Prometheus metrics between tests

### Configuration Fixtures

- `test_config`: Base test configuration
- `minimal_config`: Minimal configuration for unit tests
- `production_config`: Production-like configuration for integration tests

### Mock Fixtures

- `mock_aioresponses`: HTTP response mocking
- `mock_robots_txt`: Robots.txt response mocking
- `mock_metrics`: Metrics collection mocking

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run Tests
  run: |
    pytest -q tests/ \
      --cov=src/quarrycore \
      --cov-report=xml \
      --cov-fail-under=80 \
      -n auto
```

### Coverage Requirements

Tests must maintain:
- HTTP Client: ≥ 90% branch coverage
- Dependency Container: ≥ 90% branch coverage
- Overall: ≥ 80% coverage

### Performance Benchmarks

```bash
# Ensure no performance regression
python benchmarks/compare.py 95  # 95% retention threshold
```

## Troubleshooting

### Common Issues

1. **Timing-sensitive tests**: Use deterministic mocking instead of real delays
2. **Port conflicts**: Use dynamic port allocation in tests
3. **Resource leaks**: Ensure proper cleanup in teardown methods
4. **Flaky tests**: Add proper synchronization and avoid race conditions

### Debug Mode

```python
# Enable debug logging in tests
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export QUARRY_LOG_LEVEL=DEBUG
```

This testing strategy ensures comprehensive coverage while maintaining test reliability and development velocity. 