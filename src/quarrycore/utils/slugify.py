"""
Simple, dependency-free string slugification for safe filenames.

Converts strings to filesystem-safe names by replacing unsafe characters
with safe alternatives while preserving readability.
"""

import re
from typing import Optional

# Regex pattern for characters that are unsafe in filenames
# Matches anything that's not alphanumeric or hyphen
# Note: Dots and underscores are considered unsafe for consistency in slugification
UNSAFE_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9\-]")

# Pattern to collapse multiple consecutive hyphens
MULTIPLE_HYPHENS_PATTERN = re.compile(r"-+")

# Windows reserved filenames
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def slugify(text: str, replacement: str = "-", max_length: Optional[int] = 200, lowercase: bool = True) -> str:
    """
    Convert a string to a filesystem-safe slug.

    Replaces unsafe characters with the replacement character and optionally
    converts to lowercase. Handles edge cases like leading/trailing separators
    and consecutive separators. Also handles Windows reserved names.

    Args:
        text: Input string to slugify
        replacement: Character to replace unsafe chars with (default: '-')
        max_length: Maximum length of result (default: 200, None for unlimited)
        lowercase: Whether to convert to lowercase (default: True)

    Returns:
        Slugified string safe for use as filename

    Examples:
        >>> slugify("Hello World!")
        'hello-world'

        >>> slugify("file/path\\name:test")
        'file-path-name-test'

        >>> slugify("My File (v2.1).txt")
        'my-file-v2-1-txt'

        >>> slugify("CON")
        'con-reserved'

        >>> slugify("")
        ''
    """
    if not text or not text.strip():
        return ""

    # Start with the input text
    result = text.strip()

    # Replace unsafe characters with replacement
    result = UNSAFE_CHARS_PATTERN.sub(replacement, result)

    # Collapse multiple consecutive replacement characters
    if len(replacement) == 1:
        pattern = re.compile(f"{re.escape(replacement)}+")
        result = pattern.sub(replacement, result)

    # Remove leading and trailing replacement characters
    result = result.strip(replacement)

    # Convert to lowercase if requested
    if lowercase:
        result = result.lower()

    # Check for Windows reserved names (case-insensitive)
    # Check the base name (before any extension-like part)
    parts = result.split(replacement)
    if parts and parts[0].upper() in WINDOWS_RESERVED_NAMES:
        # Append -reserved to make it safe
        parts.append("reserved")
        result = replacement.join(parts)

    # Apply length limit if specified
    if max_length and len(result) > max_length:
        result = result[:max_length].rstrip(replacement)

    # Handle edge case where everything was stripped away
    if not result:
        return ""

    return result


def slugify_job_id(job_id: str) -> str:
    """
    Slugify a job ID specifically for checkpoint filename usage.

    This is a specialized version of slugify optimized for job IDs that may
    contain UUIDs, timestamps, or other identifier formats that need to be
    made filesystem-safe.

    Args:
        job_id: Job identifier string

    Returns:
        Filesystem-safe job ID

    Examples:
        >>> slugify_job_id('job:2024/01/01-12:30:45')
        'job-2024-01-01-12-30-45'

        >>> slugify_job_id('pipeline\\batch#123')
        'pipeline-batch-123'
    """
    return slugify(job_id, replacement="-", max_length=100, lowercase=True)  # Reasonable limit for job IDs


def is_safe_filename(filename: str) -> bool:
    """
    Check if a filename is already safe for filesystem use.

    Args:
        filename: Filename to check

    Returns:
        True if filename is safe, False otherwise
    """
    if not filename or filename in (".", ".."):
        return False

    # Check for unsafe characters
    if UNSAFE_CHARS_PATTERN.search(filename):
        return False

    # Check for reserved names on Windows
    base_name = filename.split(".")[0].upper()
    if base_name in WINDOWS_RESERVED_NAMES:
        return False

    # Check for trailing spaces or dots (problematic on Windows)
    if filename.endswith(" ") or filename.endswith("."):
        return False

    return True
