"""Unit tests for atomic file operations with cleanup verification."""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from quarrycore.utils.atomic import _cleanup_temp_files, atomic_json_dump, atomic_write_json


@pytest.mark.asyncio
async def test_atomic_json_cleanup():
    """Test that temp files are cleaned up after forced timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "test.json"
        test_data = {"key": "value", "data": "test"}

        # Don't create a manual temp file - let atomic_json_dump create its own
        # The pattern used by atomic_json_dump is .atomic_{filename}.{random}.tmp

        # Patch wait_for to simulate timeout
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            # Call with very short timeout
            result = await atomic_json_dump(test_data, target_path, timeout=0.1)

            # Should return False on timeout
            assert result is False

            # Check that no atomic temp files remain
            temp_files = list(Path(tmpdir).glob(".atomic_*"))
            assert len(temp_files) == 0, f"Found leftover temp files: {temp_files}"


@pytest.mark.asyncio
async def test_atomic_json_dump_success():
    """Test successful atomic JSON dump."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "success.json"
        test_data = {"status": "success", "count": 42}

        # Should succeed with reasonable timeout
        result = await atomic_json_dump(test_data, target_path, timeout=2.0)
        assert result is True

        # Verify file was created with correct content
        assert target_path.exists()
        with open(target_path) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

        # No temp files should remain
        temp_files = list(Path(tmpdir).glob(".success.json.*tmp"))
        assert len(temp_files) == 0


@pytest.mark.asyncio
async def test_atomic_json_dump_invalid_data():
    """Test atomic JSON dump with non-serializable data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "invalid.json"

        # Create non-serializable data
        class NonSerializable:
            pass

        test_data = {"obj": NonSerializable()}

        # Should return False for serialization error
        result = await atomic_json_dump(test_data, target_path, timeout=2.0)
        assert result is False

        # File should not exist
        assert not target_path.exists()

        # No temp files should remain
        temp_files = list(Path(tmpdir).glob(".invalid.json.*tmp"))
        assert len(temp_files) == 0


def test_cleanup_temp_files():
    """Test the _cleanup_temp_files utility function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create some temp files
        temp1 = tmpdir_path / ".target.json.12345.tmp"
        temp2 = tmpdir_path / ".target.json.67890.tmp"
        other_file = tmpdir_path / "other.txt"

        temp1.write_text("temp1")
        temp2.write_text("temp2")
        other_file.write_text("other")

        # Run cleanup
        _cleanup_temp_files(tmpdir_path, "target.json")

        # Check that only temp files were removed
        assert not temp1.exists()
        assert not temp2.exists()
        assert other_file.exists()


def test_atomic_write_json_cleanup_on_error():
    """Test that atomic_write_json cleans up on error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "error.json"
        test_data = {"test": "data"}

        # Mock os.replace to fail
        with patch("os.replace", side_effect=OSError("Mock replace error")):
            # Also mock shutil.move to fail
            with patch("shutil.move", side_effect=OSError("Mock move error")):
                # Should raise OSError
                with pytest.raises(OSError):
                    atomic_write_json(target_path, test_data)

        # No temp files should remain
        temp_files = list(Path(tmpdir).glob(".error.json.*tmp"))
        assert len(temp_files) == 0


def test_atomic_write_json_final_cleanup():
    """Test the finally block cleanup in atomic_write_json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "final.json"
        test_data = {"final": "test"}

        # Mock os.fsync to raise exception to trigger cleanup
        with patch("os.fsync", side_effect=OSError("Mock fsync error")):
            # Should raise OSError
            with pytest.raises(OSError):
                atomic_write_json(target_path, test_data)

        # No temp files should remain after the error
        temp_files = list(Path(tmpdir).glob(".final.json.*tmp"))
        assert len(temp_files) == 0, f"Found leftover temp files: {temp_files}"

        # Target file should not exist
        assert not target_path.exists()


@pytest.mark.asyncio
async def test_atomic_json_dump_directory_creation():
    """Test that atomic_json_dump creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested path that doesn't exist
        target_path = Path(tmpdir) / "nested" / "deep" / "test.json"
        test_data = {"nested": True}

        # Should succeed and create directories
        result = await atomic_json_dump(test_data, target_path, timeout=2.0)
        assert result is True

        # Verify directories were created
        assert target_path.parent.exists()
        assert target_path.exists()

        # Verify content
        with open(target_path) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data
