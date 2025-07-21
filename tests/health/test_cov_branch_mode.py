"""Health check test to ensure branch coverage is always enabled."""

import coverage


def test_branch_coverage_enabled():
    """Test that branch coverage is enabled in the coverage configuration."""
    # Create a coverage instance to check configuration
    cov = coverage.Coverage()

    # Check that branch coverage is enabled - option format is "section:option"
    assert (
        cov.get_option("run:branch") is True
    ), "Branch coverage must be enabled. Ensure [tool.coverage.run] branch = true in pyproject.toml"


def test_parallel_coverage_enabled():
    """Test that parallel coverage is enabled for multi-process test runs."""
    # Create a coverage instance to check configuration
    cov = coverage.Coverage()

    # Check that parallel coverage is enabled - option format is "section:option"
    assert cov.get_option("run:parallel") is True, (
        "Parallel coverage must be enabled for multi-process tests. "
        "Ensure [tool.coverage.run] parallel = true in pyproject.toml"
    )
