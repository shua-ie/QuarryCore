# Contributing to QuarryCore 🤝

Thank you for your interest in contributing to QuarryCore! We're building an advanced AI training data pipeline, and we welcome contributions from the community.

## 🌟 Why Contribute?

- **Impact**: Help shape a production-grade data pipeline for AI training
- **Learning**: Work with modern Python async patterns and ML technologies
- **Community**: Join developers working on data engineering challenges

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (Required for modern async features)
- **Git** for version control
- **Optional**: Docker for containerized development
- **Optional**: NVIDIA GPU for accelerated features

### Development Environment Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/shua-ie/quarrycore.git
cd quarrycore

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode with dev dependencies
pip install -e ".[dev]"

# 4. Verify installation
python -c "import quarrycore; print('✅ QuarryCore installed')"

# 5. Run tests to ensure everything works
pytest tests/ -v
```

### Optional: GPU Development

If you have an NVIDIA GPU and want to work on GPU-accelerated features:

```bash
# Install with GPU support
pip install -e ".[dev,gpu]"
```

## 🎯 How to Contribute

### 1. Find an Issue

- Browse [open issues](https://github.com/shua-ie/quarrycore/issues)
- Look for issues labeled `good first issue` or `help wanted`
- Check [discussions](https://github.com/shua-ie/quarrycore/discussions) for ideas
- Propose new features through feature requests

### 2. Create a Branch

```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Changes

Follow our development guidelines below.

### 4. Test Your Changes

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_crawler.py -v

# Run with coverage
pytest tests/ --cov=quarrycore --cov-report=html

# Run linting
ruff check src/ tests/
black --check src/ tests/
mypy src/quarrycore --strict
```

### 5. Submit a Pull Request

- Push your changes to your fork
- Create a pull request with a clear description
- Link to any related issues

## 📋 Development Guidelines

### Code Style

We use modern Python practices and maintain high code quality:

#### Python Code Standards

```python
# ✅ Good: Type hints, docstrings, async patterns
from typing import List, Optional
from quarrycore.protocols import CrawlerProtocol

async def process_urls(
    urls: List[str],
    crawler: CrawlerProtocol,
    *,
    max_concurrent: int = 10,
) -> List[CrawlResult]:
    """
    Process multiple URLs concurrently.
    
    Args:
        urls: List of URLs to process
        crawler: Crawler implementation
        max_concurrent: Maximum concurrent requests
        
    Returns:
        List of crawl results
    """
    # Implementation here...
```

#### Async/Await Patterns

QuarryCore is built on async/await. Use proper async patterns:

```python
# ✅ Good: Proper async context managers and error handling
async with httpx.AsyncClient() as client:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error(f"HTTP error: {e}")
        raise
```

### Testing Standards

#### Running Tests

```bash
# Unit tests
pytest tests/ -m "unit"

# Integration tests (may require services)
pytest tests/ -m "integration"

# Security tests
pytest tests/test_security_comprehensive.py -v

# Performance benchmarks
pytest tests/test_performance_benchmarks.py -v
```

#### Writing Tests

```python
# tests/test_example.py
import pytest
from quarrycore.example import process_data

class TestDataProcessing:
    """Test data processing functionality."""
    
    @pytest.mark.unit
    async def test_basic_processing(self):
        """Test basic data processing."""
        result = await process_data("test input")
        assert result.success is True
        assert len(result.output) > 0
```

### Documentation

#### Docstring Format

Use Google-style docstrings:

```python
def extract_content(html: str, url: str) -> ExtractedContent:
    """
    Extract content from HTML.
    
    Args:
        html: Raw HTML content
        url: Source URL for context
        
    Returns:
        Extracted content with metadata
        
    Raises:
        ExtractionError: If extraction fails
    """
```

### Project Structure

```
src/quarrycore/
├── __init__.py          # Package initialization
├── config/              # Configuration management
├── crawler/             # Web crawling components
├── extractor/           # Content extraction
├── deduplicator/        # Deduplication engine
├── quality/             # Quality assessment
├── metadata/            # Metadata extraction
├── storage/             # Storage backends
├── dataset/             # Dataset construction
├── auth/                # Authentication (JWT)
├── security/            # Security features
├── monitoring/          # Metrics and monitoring
├── observability/       # Logging and tracing
└── protocols.py         # Protocol definitions
```

## 🔧 Common Development Tasks

### Adding a New Feature

1. **Create a protocol** if adding a new component type
2. **Implement the feature** following existing patterns
3. **Add tests** with good coverage
4. **Update documentation** if needed

### Fixing a Bug

1. **Write a test** that reproduces the bug
2. **Fix the issue** with minimal changes
3. **Verify** all tests pass
4. **Document** the fix in your PR

### Improving Performance

1. **Profile first** to identify bottlenecks
2. **Benchmark** before and after changes
3. **Document** performance improvements
4. **Consider** hardware adaptation

## 🚫 What to Avoid

### Code Quality
- ❌ Hardcoded values - use configuration
- ❌ Synchronous I/O in async functions
- ❌ Missing type hints
- ❌ Untested code

### Security
- ❌ Hardcoded secrets or credentials
- ❌ Unvalidated user input
- ❌ SQL string concatenation
- ❌ Disabled security features

### Pull Requests
- ❌ Large, unfocused changes
- ❌ Missing tests
- ❌ Breaking changes without discussion
- ❌ Incomplete implementations

## 🤝 Code Review Process

### For Contributors
- Respond to feedback constructively
- Update your PR based on reviews
- Ask questions if something is unclear
- Be patient - reviews take time

### For Reviewers
- Be constructive and specific
- Focus on significant issues
- Suggest improvements
- Approve when ready

## 📚 Resources

### Documentation
- [Architecture Overview](ANALYSIS_REPORT.md)
- [Security Guide](SECURITY.md)
- [Deployment Guide](DEPLOYMENT.md)

### Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **ruff**: Fast linting
- **mypy**: Type checking

## ❓ Getting Help

- **GitHub Issues**: For bugs and features
- **Discussions**: For questions and ideas
- **Email**: josh.mcd31@gmail.com for other inquiries

## 📄 License

By contributing to QuarryCore, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

---

Thank you for contributing to QuarryCore! Your efforts help make this project better for everyone. 🙏 