"""
Cross-platform atomic file writing utilities.

Provides atomic file operations that work reliably on both Linux and Windows
with proper fallback mechanisms for different filesystem scenarios.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


async def atomic_json_dump(data: Any, path: Path, timeout: float = 2.0) -> bool:
    """
    Asynchronously write JSON data to a file atomically with timeout.

    Args:
        data: Data to serialize as JSON
        path: Target file path
        timeout: Maximum time to wait for write operation

    Returns:
        bool: True if write succeeded, False otherwise

    Notes:
        - Guarantees cleanup of temporary files even on timeout
        - Logs warnings on failure but doesn't raise exceptions
    """
    path = Path(path)
    temp_file_path = None

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize data first to catch JSON errors early
        try:
            json_content = json.dumps(data, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"Cannot serialize data to JSON: {e}", path=str(path))
            return False

        # Create temp file with a unique name
        temp_fd, temp_file_path = tempfile.mkstemp(dir=path.parent, prefix=f".atomic_{path.name}.", suffix=".tmp")

        # Close the file descriptor immediately - we'll write using path
        os.close(temp_fd)
        temp_file_path = Path(temp_file_path)

        # Write content asynchronously with timeout
        loop = asyncio.get_event_loop()

        async def write_and_replace():
            # Write content
            await loop.run_in_executor(None, temp_file_path.write_text, json_content, "utf-8")
            # Atomic replace
            await loop.run_in_executor(None, os.replace, str(temp_file_path), str(path))

        await asyncio.wait_for(write_and_replace(), timeout=timeout)

        # Clean up any stale atomic files older than 1 hour
        _remove_stale_atomic_files(path.parent)

        return True

    except asyncio.TimeoutError:
        logger.warning(f"Atomic write timed out after {timeout}s", path=str(path))
        return False
    except Exception as e:
        logger.warning(f"Atomic write failed: {e}", path=str(path))
        return False
    finally:
        # Always clean up temp file if it exists
        if temp_file_path and isinstance(temp_file_path, Path) and temp_file_path.exists():
            try:
                temp_file_path.unlink(missing_ok=True)
            except Exception:
                pass


def _remove_stale_atomic_files(dir_: Path, pattern: str = ".atomic_*") -> None:
    """
    Remove stale atomic temporary files older than 1 hour.

    Args:
        dir_: Directory to clean
        pattern: Glob pattern for atomic temp files
    """
    try:
        current_time = time.time()
        for temp_file in dir_.glob(pattern):
            try:
                # Check if file is older than 1 hour
                if temp_file.is_file():
                    file_age = current_time - temp_file.stat().st_mtime
                    if file_age > 3600:  # 1 hour
                        temp_file.unlink()
                        logger.debug(f"Removed stale atomic file: {temp_file}")
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Error during stale file cleanup: {e}")


def _cleanup_temp_files(directory: Path, target_name: str) -> None:
    """Clean up any temporary files left over from failed atomic writes."""
    try:
        pattern = f".{target_name}.*tmp"
        for temp_file in directory.glob(pattern):
            try:
                temp_file.unlink()
                logger.debug(f"Cleaned up orphaned temp file: {temp_file}")
            except OSError:
                pass
    except Exception as e:
        logger.debug(f"Error during temp file cleanup: {e}")


def atomic_write_json(target_path: Path, data: Dict[str, Any]) -> None:
    """
    Atomically write JSON data to a file with cross-platform compatibility.

    Uses same-filesystem temporary file creation followed by atomic rename
    operation on POSIX systems, with fallback to shutil.move for Windows
    compatibility when cross-filesystem operations are needed.

    Args:
        target_path: Target file path to write to
        data: Dictionary data to serialize as JSON

    Raises:
        OSError: If writing fails after all retry attempts
        ValueError: If data cannot be serialized to JSON
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize data first to catch JSON errors early
    try:
        json_content = json.dumps(data, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.error("Failed to serialize data to JSON", error=str(e))
        raise ValueError(f"Cannot serialize data to JSON: {e}") from e

    # Use same directory as target to ensure same filesystem
    temp_dir = target_path.parent
    temp_file_path = None

    try:
        # Create temporary file in same directory as target
        with tempfile.NamedTemporaryFile(
            mode="w", dir=temp_dir, prefix=f".{target_path.name}.", suffix=".tmp", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file_path = Path(temp_file.name)
            temp_file.write(json_content)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk

        # Attempt atomic rename (works on POSIX, may fail on Windows)
        try:
            # os.replace is atomic on both POSIX and Windows when on same filesystem
            os.replace(str(temp_file_path), str(target_path))
            logger.debug("Atomic write completed successfully", target=str(target_path), method="os.replace")

        except OSError as rename_error:
            logger.warning(
                "Atomic rename failed, falling back to shutil.move", error=str(rename_error), target=str(target_path)
            )

            # Fallback for Windows or cross-filesystem scenarios
            try:
                shutil.move(str(temp_file_path), str(target_path))
                logger.debug("Atomic write completed with fallback", target=str(target_path), method="shutil.move")
            except (OSError, shutil.Error) as move_error:
                logger.error(
                    "Both atomic rename and fallback move failed",
                    rename_error=str(rename_error),
                    move_error=str(move_error),
                    target=str(target_path),
                )
                raise OSError(
                    f"Failed to atomically write {target_path}: "
                    f"rename failed ({rename_error}), move failed ({move_error})"
                ) from move_error

    except Exception as e:
        # Clean up temporary file if something went wrong
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
                logger.debug("Cleaned up temporary file", temp_file=str(temp_file_path))
            except OSError as cleanup_error:
                logger.warning(
                    "Failed to clean up temporary file", temp_file=str(temp_file_path), error=str(cleanup_error)
                )

        if isinstance(e, (OSError, ValueError)):
            raise
        else:
            logger.error("Unexpected error during atomic write", error=str(e))
            raise OSError(f"Unexpected error writing {target_path}: {e}") from e

    finally:
        # Final cleanup attempt for any remaining temp files
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError:
                pass


def atomic_write_text(target_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Atomically write text content to a file.

    Args:
        target_path: Target file path to write to
        content: Text content to write
        encoding: Text encoding to use (default: utf-8)

    Raises:
        OSError: If writing fails after all retry attempts
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = target_path.parent
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=temp_dir, prefix=f".{target_path.name}.", suffix=".tmp", delete=False, encoding=encoding
        ) as temp_file:
            temp_file_path = Path(temp_file.name)
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        try:
            os.replace(str(temp_file_path), str(target_path))
        except OSError:
            shutil.move(str(temp_file_path), str(target_path))

    except Exception as e:
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except OSError:
                pass
        raise OSError(f"Failed to atomically write {target_path}: {e}") from e
