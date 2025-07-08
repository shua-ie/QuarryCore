"""
Tests for atomic utilities to boost coverage.

These tests focus on atomic file operations and rollback scenarios
to increase global line coverage toward the 24% target.
"""

import tempfile
from pathlib import Path

import pytest
from quarrycore.utils.atomic import atomic_write_json


@pytest.mark.unit
class TestAtomicUtils:
    """Test atomic utilities functionality."""

    def test_atomic_write_json_success(self):
        """Test successful atomic JSON write."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "test.json"
            test_data = {"key": "value", "number": 42}

            # Perform atomic write
            atomic_write_json(target_file, test_data)

            # Verify file exists and contains correct data
            assert target_file.exists()

            import json

            with open(target_file, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data

    def test_atomic_write_json_nested_directory(self):
        """Test atomic write with nested directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "nested" / "deep" / "test.json"
            test_data = {"nested": True}

            # Perform atomic write (should create directories)
            atomic_write_json(target_file, test_data)

            # Verify file exists
            assert target_file.exists()
            assert target_file.parent.exists()

    def test_atomic_write_json_complex_data(self):
        """Test atomic write with complex data structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "complex.json"
            test_data = {
                "list": [1, 2, 3],
                "dict": {"nested": {"deep": "value"}},
                "boolean": True,
                "null": None,
                "string": "test",
            }

            # Perform atomic write
            atomic_write_json(target_file, test_data)

            # Verify file exists and data integrity
            assert target_file.exists()

            import json

            with open(target_file, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data

    def test_atomic_write_json_overwrite_existing(self):
        """Test atomic write overwrites existing file correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "overwrite.json"

            # Write initial data
            initial_data = {"version": 1}
            atomic_write_json(target_file, initial_data)

            # Verify initial write
            import json

            with open(target_file, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == initial_data

            # Overwrite with new data
            new_data = {"version": 2, "updated": True}
            atomic_write_json(target_file, new_data)

            # Verify overwrite
            with open(target_file, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == new_data
            assert loaded_data != initial_data

    def test_atomic_write_json_empty_data(self):
        """Test atomic write with empty data structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            target_file = Path(temp_dir) / "empty.json"

            # Test empty dict
            atomic_write_json(target_file, {})

            import json

            with open(target_file, "r") as f:
                loaded_data = json.load(f)
            assert loaded_data == {}

    def test_atomic_write_json_error_handling(self):
        """Test atomic write error handling with invalid file path."""
        # Test with invalid path (root directory which should fail)
        target_file = Path("/") / "should_fail.json"
        test_data = {"test": "data"}

        with pytest.raises(OSError):
            atomic_write_json(target_file, test_data)
