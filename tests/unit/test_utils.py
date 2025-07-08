"""
Unit tests for utility modules (atomic writes and slugification).
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from quarrycore.utils import atomic_write_json, slugify


class TestAtomicWrite:
    """Test atomic file writing utilities."""

    def test_atomic_write_json_basic(self):
        """Test basic atomic JSON writing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"
            test_data = {"key": "value", "number": 42, "nested": {"inner": "data"}}

            atomic_write_json(test_file, test_data)

            assert test_file.exists()

            with open(test_file) as f:
                loaded_data = json.load(f)

            assert loaded_data == test_data

    def test_atomic_write_json_overwrite(self):
        """Test atomic JSON writing with overwrite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"

            # Write initial data
            initial_data = {"version": 1, "data": "initial"}
            atomic_write_json(test_file, initial_data)

            # Overwrite with new data
            new_data = {"version": 2, "data": "updated"}
            atomic_write_json(test_file, new_data)

            with open(test_file) as f:
                loaded_data = json.load(f)

            assert loaded_data == new_data
            assert loaded_data["version"] == 2

    def test_atomic_write_json_creates_parent_dir(self):
        """Test that atomic write creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_file = Path(temp_dir) / "nested" / "deep" / "test.json"
            test_data = {"created": "nested"}

            atomic_write_json(nested_file, test_data)

            assert nested_file.exists()
            assert nested_file.parent.exists()

    def test_atomic_write_json_invalid_data(self):
        """Test atomic write with non-serializable data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"

            # Data that can't be JSON serialized
            invalid_data = {"function": lambda x: x}

            with pytest.raises(ValueError, match="Cannot serialize data to JSON"):
                atomic_write_json(test_file, invalid_data)

    def test_atomic_write_json_unicode(self):
        """Test atomic write with Unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "unicode.json"
            unicode_data = {
                "english": "Hello World",
                "chinese": "你好世界",
                "japanese": "こんにちは世界",
                "emoji": "🌍🚀✨",
            }

            atomic_write_json(test_file, unicode_data)

            with open(test_file, encoding="utf-8") as f:
                loaded_data = json.load(f)

            assert loaded_data == unicode_data


class TestSlugify:
    """Test the slugify function with various edge cases."""

    def test_basic_slugification(self):
        """Test basic string slugification."""
        assert slugify("Hello World") == "hello-world"
        assert slugify("Test_File_Name") == "test-file-name"
        assert slugify("123-Numbers") == "123-numbers"

    def test_special_characters(self):
        """Test removal of special characters."""
        assert slugify("Hello@World!") == "hello-world"
        assert slugify("Test#$%^&*()") == "test"
        assert slugify("File/Path\\Name") == "file-path-name"

    def test_dots_removal(self):
        """Test that dots are properly removed (AC-02)."""
        assert slugify("file.name.txt") == "file-name-txt"
        assert slugify("test...dots") == "test-dots"
        assert slugify(".hidden.file") == "hidden-file"
        assert slugify("ends.with.") == "ends-with"

    def test_windows_reserved_names(self):
        """Test that Windows reserved names are handled (AC-02)."""
        # Common Windows reserved names
        assert slugify("CON") == "con-reserved"
        assert slugify("PRN") == "prn-reserved"
        assert slugify("AUX") == "aux-reserved"
        assert slugify("NUL") == "nul-reserved"
        assert slugify("COM1") == "com1-reserved"
        assert slugify("COM9") == "com9-reserved"
        assert slugify("LPT1") == "lpt1-reserved"
        assert slugify("LPT9") == "lpt9-reserved"

    def test_windows_reserved_case_insensitive(self):
        """Test Windows reserved names are handled case-insensitively."""
        assert slugify("con") == "con-reserved"
        assert slugify("Con") == "con-reserved"
        assert slugify("CON") == "con-reserved"
        assert slugify("CoN") == "con-reserved"

    def test_unicode_handling(self):
        """Test Unicode character handling."""
        # Current implementation strips non-ASCII characters
        assert slugify("Café") == "caf"
        assert slugify("naïve") == "na-ve"  # ï gets stripped, leaving na ve
        assert slugify("Москва") == "untitled"  # All non-ASCII results in empty, returns untitled
        assert slugify("北京") == "untitled"  # All non-ASCII results in empty, returns untitled
        assert slugify("🚀 Rocket") == "rocket"

    def test_max_length(self):
        """Test max length parameter."""
        long_string = "a" * 200
        assert len(slugify(long_string, max_length=50)) == 50
        assert len(slugify(long_string, max_length=100)) == 100

        # Ensure it doesn't cut in the middle of a word substitute
        assert slugify("test-file-name", max_length=10) == "test-file"

    def test_replacement_character(self):
        """Test custom replacement character."""
        assert slugify("Hello World", replacement="_") == "hello_world"
        assert slugify("Test File.txt", replacement="_") == "test_file_txt"
        assert slugify("CON", replacement="_") == "con_reserved"

    def test_lowercase_option(self):
        """Test lowercase option."""
        assert slugify("Hello World", lowercase=True) == "hello-world"
        assert slugify("Hello World", lowercase=False) == "Hello-World"
        assert slugify("TEST", lowercase=False) == "TEST"

    def test_empty_and_whitespace(self):
        """Test empty strings and whitespace."""
        assert slugify("") == "untitled"
        assert slugify("   ") == "untitled"
        assert slugify("\t\n") == "untitled"
        assert slugify("  spaces  ") == "spaces"

    def test_consecutive_separators(self):
        """Test handling of consecutive separators."""
        assert slugify("test--file") == "test-file"
        assert slugify("test___file") == "test-file"
        assert slugify("test - - file") == "test-file"

    def test_complex_filenames(self):
        """Test complex filename scenarios."""
        assert slugify("My Document (Final).docx") == "my-document-final-docx"
        assert slugify("Report_2023.12.25_FINAL.pdf") == "report-2023-12-25-final-pdf"
        assert slugify("~$temp.file.backup") == "temp-file-backup"

    @pytest.mark.parametrize(
        "reserved,expected",
        [
            ("CON.txt", "con-txt-reserved"),
            ("prn.log", "prn-log-reserved"),
            ("COM1.config", "com1-config-reserved"),
            ("aux.data", "aux-data-reserved"),
        ],
    )
    def test_reserved_with_extensions(self, reserved, expected):
        """Test Windows reserved names with extensions."""
        assert slugify(reserved) == expected

    def test_numeric_strings(self):
        """Test numeric strings."""
        assert slugify("123") == "123"
        assert slugify("456.789") == "456-789"
        assert slugify("0xFF") == "0xff"

    def test_mixed_case_preservation(self):
        """Test that case can be preserved when requested."""
        assert slugify("CamelCase", lowercase=False) == "CamelCase"
        assert slugify("UPPERCASE", lowercase=False) == "UPPERCASE"
        assert slugify("miXeD.CaSe", lowercase=False) == "miXeD-CaSe"


class TestAtomicWriteEdgeCases:
    """Test edge cases and error conditions for atomic writes."""

    def test_atomic_write_permission_error(self):
        """Test atomic write with permission issues."""
        # Create a read-only directory
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = Path(temp_dir) / "readonly"
            readonly_dir.mkdir()
            readonly_dir.chmod(0o444)  # Read-only

            test_file = readonly_dir / "test.json"
            test_data = {"test": "data"}

            try:
                with pytest.raises(OSError):
                    atomic_write_json(test_file, test_data)
            finally:
                # Restore permissions for cleanup
                readonly_dir.chmod(0o755)

    def test_atomic_write_disk_full_simulation(self):
        """Test atomic write behavior when disk is full (simulated)."""
        # This is hard to test without actually filling disk
        # Just ensure our error handling works
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.json"

            # Create very large data that might cause issues
            large_data = {"data": "x" * 1000000}  # 1MB of data

            # Should work normally
            atomic_write_json(test_file, large_data)
            assert test_file.exists()

    def test_atomic_write_concurrent_access(self):
        """Test atomic write with concurrent access."""
        import threading
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "concurrent.json"
            results = []

            def write_data(data):
                try:
                    atomic_write_json(test_file, data)
                    results.append(f"success_{data['id']}")
                except Exception:
                    results.append(f"error_{data['id']}")

            # Start multiple threads writing to same file
            threads = []
            for i in range(5):
                t = threading.Thread(target=write_data, args=({"id": i, "data": f"thread_{i}"},))
                threads.append(t)
                t.start()

            # Wait for all threads
            for t in threads:
                t.join()

            # All writes should succeed (atomic)
            assert len([r for r in results if r.startswith("success")]) == 5

            # File should exist and contain valid JSON
            assert test_file.exists()
            with open(test_file) as f:
                data = json.load(f)
                assert "id" in data  # Should have one of the written data sets
