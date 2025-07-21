"""Unit tests for atomic JSON operations."""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from quarrycore.utils.atomic import _remove_stale_atomic_files, atomic_json_dump


@pytest.mark.asyncio
async def test_atomic_json_dump_success():
    """Test successful atomic JSON dump with no leftover files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "success.json"
        test_data = {"status": "success", "items": [1, 2, 3], "nested": {"key": "value"}}

        # Write should succeed
        result = await atomic_json_dump(test_data, target_path, timeout=2.0)
        assert result is True

        # Verify file was created with correct content
        assert target_path.exists()
        with open(target_path) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

        # Verify no temp files remain
        temp_files = list(Path(tmpdir).glob(".atomic_*"))
        assert len(temp_files) == 0

        # Directory should only contain the target file
        all_files = list(Path(tmpdir).iterdir())
        assert len(all_files) == 1
        assert all_files[0].name == "success.json"


@pytest.mark.asyncio
async def test_atomic_json_dump_timeout_cleanup():
    """Test that temp files are cleaned up on timeout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "timeout.json"
        test_data = {"will": "timeout"}

        # Mock wait_for to raise TimeoutError
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            # Should timeout and return False
            result = await atomic_json_dump(test_data, target_path, timeout=0.05)
            assert result is False

        # Target file should not exist
        assert not target_path.exists()

        # No temp files should remain - directory should be empty
        assert not any(Path(tmpdir).iterdir()), "Directory should be empty after timeout"


@pytest.mark.asyncio
async def test_atomic_json_dump_invalid_data():
    """Test handling of non-serializable data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "invalid.json"

        # Create non-serializable object
        class NonSerializable:
            pass

        test_data = {"obj": NonSerializable()}

        # Should return False for serialization failure
        result = await atomic_json_dump(test_data, target_path, timeout=2.0)
        assert result is False

        # File should not exist
        assert not target_path.exists()

        # No temp files should remain
        assert not any(Path(tmpdir).iterdir())


@pytest.mark.asyncio
async def test_atomic_json_dump_directory_creation():
    """Test that parent directories are created if needed."""
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


@pytest.mark.asyncio
async def test_atomic_json_dump_concurrent_writes():
    """Test concurrent writes to different files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple write tasks
        tasks = []
        for i in range(5):
            target_path = Path(tmpdir) / f"concurrent_{i}.json"
            test_data = {"index": i, "data": f"test_{i}"}
            tasks.append(atomic_json_dump(test_data, target_path, timeout=2.0))

        # Run all concurrently
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

        # Verify all files exist with correct content
        for i in range(5):
            target_path = Path(tmpdir) / f"concurrent_{i}.json"
            assert target_path.exists()
            with open(target_path) as f:
                data = json.load(f)
            assert data["index"] == i

        # No temp files should remain
        temp_files = list(Path(tmpdir).glob(".atomic_*"))
        assert len(temp_files) == 0


def test_remove_stale_atomic_files():
    """Test removal of stale atomic files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create some test files
        old_file = tmpdir_path / ".atomic_old.tmp"
        old_file.write_text("old")

        new_file = tmpdir_path / ".atomic_new.tmp"
        new_file.write_text("new")

        other_file = tmpdir_path / "other.txt"
        other_file.write_text("other")

        # Make old_file appear old by modifying its mtime
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(old_file, (old_time, old_time))

        # Run cleanup
        _remove_stale_atomic_files(tmpdir_path)

        # Old file should be removed
        assert not old_file.exists()

        # New file and other file should remain
        assert new_file.exists()
        assert other_file.exists()


@pytest.mark.asyncio
async def test_atomic_json_dump_exception_cleanup():
    """Test cleanup when os.replace fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "error.json"
        test_data = {"test": "data"}

        # Mock os.replace to fail
        with patch("os.replace", side_effect=OSError("Mock replace error")):
            result = await atomic_json_dump(test_data, target_path, timeout=2.0)
            assert result is False

        # No files should remain
        assert not any(Path(tmpdir).iterdir())


def test_atomic_write_json_sync():
    """Test the synchronous atomic_write_json function."""
    from quarrycore.utils.atomic import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "sync.json"
        test_data = {"sync": True, "method": "atomic_write_json"}

        # Should succeed
        atomic_write_json(target_path, test_data)

        # Verify file exists with correct content
        assert target_path.exists()
        with open(target_path) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data

        # No temp files should remain
        temp_files = list(Path(tmpdir).glob(".*tmp"))
        assert len(temp_files) == 0


def test_atomic_write_json_invalid_json():
    """Test atomic_write_json with non-serializable data."""
    from quarrycore.utils.atomic import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "invalid.json"

        class NonSerializable:
            pass

        test_data = {"obj": NonSerializable()}

        # Should raise ValueError
        with pytest.raises(ValueError, match="Cannot serialize data to JSON"):
            atomic_write_json(target_path, test_data)

        # No files should exist
        assert not target_path.exists()
        assert not any(Path(tmpdir).iterdir())


def test_atomic_write_text():
    """Test atomic_write_text function."""
    from quarrycore.utils.atomic import atomic_write_text

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "text.txt"
        test_content = "Hello, World!\nThis is a test.\n"

        # Write text
        atomic_write_text(target_path, test_content)

        # Verify file exists with correct content
        assert target_path.exists()
        assert target_path.read_text() == test_content

        # No temp files should remain
        temp_files = list(Path(tmpdir).glob(".*tmp"))
        assert len(temp_files) == 0


def test_cleanup_temp_files():
    """Test the _cleanup_temp_files function."""
    from quarrycore.utils.atomic import _cleanup_temp_files

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create test files
        temp1 = tmpdir_path / ".target.json.12345.tmp"
        temp2 = tmpdir_path / ".target.json.67890.tmp"
        other = tmpdir_path / "other.txt"

        temp1.write_text("temp1")
        temp2.write_text("temp2")
        other.write_text("other")

        # Run cleanup
        _cleanup_temp_files(tmpdir_path, "target.json")

        # Temp files should be removed
        assert not temp1.exists()
        assert not temp2.exists()

        # Other file should remain
        assert other.exists()


@pytest.mark.asyncio
async def test_atomic_json_dump_write_error():
    """Test atomic_json_dump when write fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "write_error.json"
        test_data = {"test": "data"}

        # Mock Path.write_text to fail
        def failing_write(*args, **kwargs):
            raise OSError("Mock write error")

        with patch.object(Path, "write_text", side_effect=failing_write):
            result = await atomic_json_dump(test_data, target_path, timeout=2.0)
            assert result is False

        # No files should remain
        assert not any(Path(tmpdir).iterdir())


@pytest.mark.asyncio
async def test_atomic_json_dump_replace_error():
    """Test atomic_json_dump when os.replace fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "replace_error.json"
        test_data = {"test": "data"}

        # Mock os.replace to fail
        original_replace = os.replace

        def failing_replace(src, dst):
            if "atomic_" in src:
                raise OSError("Mock replace error")
            return original_replace(src, dst)

        with patch("os.replace", side_effect=failing_replace):
            result = await atomic_json_dump(test_data, target_path, timeout=2.0)
            assert result is False

        # No temp files should remain after cleanup
        assert not any(Path(tmpdir).glob(".atomic_*"))


def test_atomic_write_json_fallback_to_shutil():
    """Test atomic_write_json fallback to shutil.move."""
    from quarrycore.utils.atomic import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "fallback.json"
        test_data = {"fallback": True}

        # Mock os.replace to fail
        with patch("os.replace", side_effect=OSError("Mock replace error")):
            # Should succeed using shutil.move
            atomic_write_json(target_path, test_data)

        # Verify file exists with correct content
        assert target_path.exists()
        with open(target_path) as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data


def test_atomic_write_json_both_operations_fail():
    """Test atomic_write_json when both os.replace and shutil.move fail."""
    from quarrycore.utils.atomic import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "both_fail.json"
        test_data = {"will": "fail"}

        # Mock both operations to fail
        with patch("os.replace", side_effect=OSError("Mock replace error")):
            with patch("shutil.move", side_effect=OSError("Mock move error")):
                # Should raise OSError
                with pytest.raises(OSError, match="Failed to atomically write"):
                    atomic_write_json(target_path, test_data)

        # No files should exist
        assert not target_path.exists()
        assert not any(Path(tmpdir).iterdir())


def test_atomic_write_text_with_encoding():
    """Test atomic_write_text with custom encoding."""
    from quarrycore.utils.atomic import atomic_write_text

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "utf16.txt"
        test_content = "Unicode: 你好世界 🌍"

        # Write with UTF-16 encoding
        atomic_write_text(target_path, test_content, encoding="utf-16")

        # Read back with same encoding
        assert target_path.exists()
        assert target_path.read_text(encoding="utf-16") == test_content


def test_atomic_write_text_with_error():
    """Test atomic_write_text error handling."""
    from quarrycore.utils.atomic import atomic_write_text

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "error.txt"

        # Mock os.replace to fail
        with patch("os.replace", side_effect=OSError("Mock replace error")):
            with patch("shutil.move", side_effect=OSError("Mock move error")):
                # Should raise OSError
                with pytest.raises(OSError):
                    atomic_write_text(target_path, "test content")


def test_atomic_write_json_unexpected_error():
    """Test atomic_write_json with unexpected error type."""
    from quarrycore.utils.atomic import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "unexpected.json"
        test_data = {"test": "data"}

        # Mock flush to raise unexpected error
        with patch("os.fsync", side_effect=RuntimeError("Unexpected error")):
            # Should raise OSError wrapping the RuntimeError
            with pytest.raises(OSError, match="Unexpected error"):
                atomic_write_json(target_path, test_data)


def test_atomic_write_json_directory_creation():
    """Test that atomic_write_json creates parent directories."""
    from quarrycore.utils.atomic import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        # Nested path that doesn't exist
        target_path = Path(tmpdir) / "nested" / "deep" / "file.json"
        test_data = {"nested": "test"}

        # Should create directories and succeed
        atomic_write_json(target_path, test_data)

        # Verify file exists
        assert target_path.exists()
        assert target_path.parent.exists()


def test_cleanup_error_handling():
    """Test that cleanup errors are handled gracefully."""
    from quarrycore.utils.atomic import atomic_write_json

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "cleanup_test.json"
        test_data = {"test": "data"}

        # Write successfully first
        atomic_write_json(target_path, test_data)

        # Even with mock unlink errors during cleanup, should not raise
        with patch("pathlib.Path.unlink", side_effect=OSError("Mock unlink error")):
            # Write again - should succeed despite cleanup errors
            atomic_write_json(target_path, {"new": "data"})

        # File should exist with new data
        assert target_path.exists()
        with open(target_path) as f:
            data = json.load(f)
        assert data == {"new": "data"}


def test_remove_stale_atomic_files_with_errors():
    """Test _remove_stale_atomic_files error handling."""
    from quarrycore.utils.atomic import _remove_stale_atomic_files

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Mock glob to raise error
        with patch.object(Path, "glob", side_effect=OSError("Permission denied")):
            # Should not raise, just log
            _remove_stale_atomic_files(tmpdir_path)


def test_remove_stale_atomic_files_unlink_error():
    """Test _remove_stale_atomic_files when unlink fails."""
    from quarrycore.utils.atomic import _remove_stale_atomic_files

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create an old file
        old_file = tmpdir_path / ".atomic_old.tmp"
        old_file.write_text("old")

        # Make it appear old
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(old_file, (old_time, old_time))

        # Mock unlink to fail
        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
            # Should not raise
            _remove_stale_atomic_files(tmpdir_path)

        # File should still exist since unlink failed
        assert old_file.exists()


@pytest.mark.asyncio
async def test_atomic_json_dump_stale_cleanup():
    """Test that stale files are cleaned up on successful write."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "cleanup.json"
        test_data = {"test": "cleanup"}

        # Create a stale atomic file
        stale_file = Path(tmpdir) / ".atomic_stale.tmp"
        stale_file.write_text("stale")

        # Make it old
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(stale_file, (old_time, old_time))

        # Successful write should trigger cleanup
        result = await atomic_json_dump(test_data, target_path, timeout=2.0)
        assert result is True

        # Target should exist
        assert target_path.exists()

        # Stale file should be removed
        assert not stale_file.exists()
