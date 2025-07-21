"""
Unit tests for HTML canonicalization processor.

Tests the production-grade HTML canonicalization that ensures
consistent hashing for exact duplicate detection.
"""

import pytest
from quarrycore.dedup.canonical import CanonicalHTMLProcessor, canonicalize_html


class TestCanonicalHTMLProcessor:
    """Test the CanonicalHTMLProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = CanonicalHTMLProcessor()

    def test_basic_html_canonicalization(self):
        """Test basic HTML canonicalization."""
        html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Hello World</h1>
                <p>This is a test page.</p>
            </body>
        </html>
        """

        result = self.processor.canonicalize(html)

        # Should extract text content and normalize whitespace
        assert "Test Page" in result
        assert "Hello World" in result
        assert "This is a test page." in result

        # Should not contain HTML tags
        assert "<html>" not in result
        assert "<body>" not in result
        assert "<p>" not in result

    def test_script_and_style_removal(self):
        """Test that script and style tags are completely removed."""
        html = """
        <html>
            <head>
                <title>Test</title>
                <script>
                    alert('This should be removed');
                    var malicious = 'code';
                </script>
                <style>
                    body { color: red; }
                    .hidden { display: none; }
                </style>
            </head>
            <body>
                <h1>Visible Content</h1>
                <script>document.write('More script')</script>
                <p>Good content here</p>
                <style>.more-css { font-size: 12px; }</style>
            </body>
        </html>
        """

        result = self.processor.canonicalize(html)

        # Should contain visible content
        assert "Test" in result
        assert "Visible Content" in result
        assert "Good content here" in result

        # Should NOT contain script or style content
        assert "alert" not in result
        assert "malicious" not in result
        assert "color: red" not in result
        assert "display: none" not in result
        assert "document.write" not in result
        assert "font-size" not in result

    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized."""
        html = """
        <html>
            <body>
                <p>   Multiple    spaces   between    words   </p>
                <div>
                    Line breaks
                    and    tabs		should    be
                    normalized
                </div>
            </body>
        </html>
        """

        result = self.processor.canonicalize(html)

        # Should normalize multiple spaces to single spaces
        assert "Multiple spaces between words" in result
        assert "Line breaks and tabs should be normalized" in result

        # Should not have leading/trailing whitespace
        assert not result.startswith(" ")
        assert not result.endswith(" ")

        # Should not contain multiple consecutive spaces
        assert "  " not in result

    def test_idempotent_processing(self):
        """Test that processing the same content twice gives same result."""
        html = """
        <div>
            <script>var x = 1;</script>
            <p>   Content   with   spaces   </p>
            <style>p { color: blue; }</style>
        </div>
        """

        result1 = self.processor.canonicalize(html)
        result2 = self.processor.canonicalize(html)

        assert result1 == result2

    def test_consistent_output_for_equivalent_content(self):
        """Test that semantically equivalent HTML produces identical output."""
        html1 = """
        <html><head><title>Test</title></head>
        <body><p>Hello World</p></body></html>
        """

        html2 = """
        <html>
            <head>
                <title>Test</title>
            </head>
            <body>
                <p>Hello    World</p>
            </body>
        </html>
        """

        html3 = """
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Test</title>
                <style>body { margin: 0; }</style>
            </head>
            <body>
                <script>console.log('test');</script>
                <p>Hello World</p>
            </body>
        </html>
        """

        result1 = self.processor.canonicalize(html1)
        result2 = self.processor.canonicalize(html2)
        result3 = self.processor.canonicalize(html3)

        # All should produce the same canonical output
        assert result1 == result2 == result3
        assert "Test Hello World" == result1.strip()

    def test_empty_content_handling(self):
        """Test handling of empty or invalid content."""
        # Empty string
        with pytest.raises(ValueError):
            self.processor.canonicalize("")

        # Whitespace only
        with pytest.raises(ValueError):
            self.processor.canonicalize("   \n\t  ")

        # Only script/style content (results in empty after processing)
        html_only_scripts = "<script>alert('test');</script><style>body{}</style>"
        result = self.processor.canonicalize(html_only_scripts)
        assert result == ""

    def test_html_entities_handling(self):
        """Test that HTML entities are properly decoded."""
        html = """
        <p>Test &amp; example with &lt;entities&gt; and &quot;quotes&quot;</p>
        <div>Non-breaking&nbsp;space and &#39;apostrophe&#39;</div>
        """

        result = self.processor.canonicalize(html)

        # Should decode common HTML entities
        assert "Test & example" in result
        assert "<entities>" in result
        assert '"quotes"' in result
        assert "Non-breaking space" in result
        assert "'apostrophe'" in result

    def test_malformed_html_resilience(self):
        """Test that processor handles malformed HTML gracefully."""
        malformed_html = """
        <html>
            <head><title>Test</title>
            <body>
                <p>Unclosed paragraph
                <div>Nested without closing
                <script>alert('unclosed script'
                <p>More content</p>
            </body>
        </html>
        """

        # Should not raise an exception
        result = self.processor.canonicalize(malformed_html)

        # Should still extract visible content
        assert "Test" in result
        assert "More content" in result

        # Should not contain script content
        assert "alert" not in result

    def test_performance_stats(self):
        """Test that performance statistics are tracked."""
        # Process some content
        html = "<p>Test content</p>"
        self.processor.canonicalize(html)
        self.processor.canonicalize(html)

        stats = self.processor.get_stats()

        assert stats["processed_count"] == 2
        assert "fallback_rate" in stats
        assert "using_selectolax" in stats
        assert isinstance(stats["fallback_rate"], float)

    def test_convenience_function(self):
        """Test the module-level convenience function."""
        html = "<p>Test content</p>"
        result = canonicalize_html(html)

        assert "Test content" in result
        assert "<p>" not in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_content(self):
        """Test processing of very large HTML content."""
        # Create large HTML content
        large_content = "<div>" + "Test content. " * 10000 + "</div>"

        processor = CanonicalHTMLProcessor()
        result = processor.canonicalize(large_content)

        # Should handle large content without errors
        assert "Test content." in result
        assert len(result) > 0

    def test_nested_script_style_tags(self):
        """Test deeply nested and complex script/style removal."""
        html = """
        <div>
            <script>
                var x = '<style>body{}</style>';
                document.write('<script>alert("nested")</script>');
            </script>
            <div>
                <style>
                    /* Comment with <script> tag */
                    .class { content: "</style>"; }
                </style>
                <p>Actual content</p>
            </div>
        </div>
        """

        processor = CanonicalHTMLProcessor()
        result = processor.canonicalize(html)

        # Should only contain actual content
        assert "Actual content" in result
        assert "var x" not in result
        assert "alert" not in result
        assert "Comment with" not in result
        assert ".class" not in result

    def test_unicode_content(self):
        """Test handling of Unicode content."""
        html = """
        <div>
            <p>Unicode test: 你好世界 🌍 émojis</p>
            <p>Mathematical symbols: ∑ ∏ ∫ ∞</p>
        </div>
        """

        processor = CanonicalHTMLProcessor()
        result = processor.canonicalize(html)

        # Should preserve Unicode characters
        assert "你好世界" in result
        assert "🌍" in result
        assert "émojis" in result
        assert "∑ ∏ ∫ ∞" in result
