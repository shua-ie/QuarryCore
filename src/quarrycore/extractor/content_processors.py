"""
Specialized Content Processors for Multi-Modal Extraction

Implements dedicated processors for different content types:
- Text: Clean text with boilerplate removal
- Tables: Structure preservation with data extraction
- Code: Language detection and syntax highlighting
- Images: Alt-text and caption extraction
- Links: Classification and metadata extraction
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

# Third-party imports with graceful fallbacks
try:
    from selectolax.lexbor import LexborHTMLParser  # type: ignore[import-not-found]
    from selectolax.parser import HTMLParser  # type: ignore[import-not-found]

    HAS_SELECTOLAX = True
except ImportError:
    HTMLParser = None
    LexborHTMLParser = None
    HAS_SELECTOLAX = False

try:
    import pygments
    from pygments.lexers import get_all_lexers, get_lexer_by_name, guess_lexer
    from pygments.util import ClassNotFound

    HAS_PYGMENTS = True
except ImportError:
    pygments = None  # type: ignore[assignment]
    HAS_PYGMENTS = False

logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Advanced text processor with boilerplate removal and cleaning.
    """

    def __init__(self) -> None:
        # Boilerplate patterns to remove
        self.boilerplate_patterns: List[str] = [
            # Navigation and UI elements
            r"\b(home|about|contact|menu|navigation|nav|sidebar)\b.*?(?:\n|$)",
            r"\b(login|register|sign up|sign in|logout)\b.*?(?:\n|$)",
            r"\b(previous|next|back|forward|continue)\b.*?(?:\n|$)",
            # Advertisement patterns
            r"\b(advertisement|sponsored|ads?|promotion)\b.*?(?:\n|$)",
            r"\b(click here|read more|learn more|view all)\b.*?(?:\n|$)",
            # Footer and legal
            r"\b(copyright|©|\(c\)|privacy|terms|conditions|policy)\b.*?(?:\n|$)",
            r"\b(all rights reserved|disclaimer|legal)\b.*?(?:\n|$)",
            # Social media and sharing
            r"\b(share|tweet|like|follow|subscribe)\b.*?(?:\n|$)",
            r"\b(facebook|twitter|instagram|linkedin|youtube)\b.*?(?:\n|$)",
            # Technical/meta content
            r"\b(loading|please wait|error|404|not found)\b.*?(?:\n|$)",
            r"\b(javascript|cookies?|enable|disable)\b.*?(?:\n|$)",
        ]

        # Patterns for content structure improvement
        self.structure_patterns: Dict[str, str] = {
            "list_items": r"^[-•*]\s+(.+)$",
            "numbered_lists": r"^\d+\.\s+(.+)$",
            "headers": r"^#{1,6}\s+(.+)$",
            "quotes": r"^>\s+(.+)$",
        }

        logger.info("TextProcessor initialized")

    async def clean_text(
        self,
        text: str,
        *,
        remove_boilerplate: bool = True,
        normalize_whitespace: bool = True,
        preserve_structure: bool = True,
    ) -> str:
        """
        Clean and process text content.

        Args:
            text: Raw text to clean
            remove_boilerplate: Whether to remove boilerplate content
            normalize_whitespace: Whether to normalize whitespace
            preserve_structure: Whether to preserve text structure

        Returns:
            Cleaned text content
        """
        if not text:
            return ""

        cleaned_text = text

        # Remove boilerplate patterns
        if remove_boilerplate:
            cleaned_text = await self._remove_boilerplate(cleaned_text)

        # Normalize whitespace while preserving structure
        if normalize_whitespace:
            cleaned_text = await self._normalize_whitespace(cleaned_text, preserve_structure)

        # Remove excessive line breaks
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text.strip()

    async def _remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate content using pattern matching."""
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

        return text

    async def _normalize_whitespace(self, text: str, preserve_structure: bool) -> str:
        """Normalize whitespace while optionally preserving structure."""
        if preserve_structure:
            # Preserve paragraph breaks (double newlines)
            lines = text.split("\n")
            normalized_lines: List[str] = []

            for line in lines:
                # Clean up whitespace within lines
                cleaned_line = re.sub(r"\s+", " ", line.strip())
                normalized_lines.append(cleaned_line)

            # Rejoin with single newlines, then restore paragraph breaks
            text = "\n".join(normalized_lines)
            text = re.sub(r"\n\s*\n", "\n\n", text)
        else:
            # Simple whitespace normalization
            text = re.sub(r"\s+", " ", text)

        return text

    async def extract_structured_content(self, text: str) -> Dict[str, List[str]]:
        """Extract structured content elements like lists and headers."""
        structured: Dict[str, List[str]] = {
            "headers": [],
            "lists": [],
            "quotes": [],
            "paragraphs": [],
        }

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for headers
            header_match = re.match(self.structure_patterns["headers"], line)
            if header_match:
                structured["headers"].append(header_match.group(1))
                continue

            # Check for list items
            list_match = re.match(self.structure_patterns["list_items"], line)
            if not list_match:
                list_match = re.match(self.structure_patterns["numbered_lists"], line)

            if list_match:
                structured["lists"].append(list_match.group(1))
                continue

            # Check for quotes
            quote_match = re.match(self.structure_patterns["quotes"], line)
            if quote_match:
                structured["quotes"].append(quote_match.group(1))
                continue

            # Regular paragraph
            if len(line) > 20:  # Minimum length for paragraph
                structured["paragraphs"].append(line)

        return structured

    def _format_list_item(self, item: Any) -> str:
        """Format a list item with proper indentation."""
        if not item:
            return ""

        # Handle different list item types
        if hasattr(item, "get_text"):
            text = str(item.get_text(strip=True))
        elif hasattr(item, "text"):
            text = str(item.text).strip()
        else:
            text = str(item).strip()

        return f"• {text}" if text else ""

    def _format_blockquote(self, quote: Any) -> str:
        """Format a blockquote with proper styling."""
        if not quote:
            return ""

        # Extract text content
        if hasattr(quote, "get_text"):
            text = str(quote.get_text(strip=True))
        elif hasattr(quote, "text"):
            text = str(quote.text).strip()
        else:
            text = str(quote).strip()

        if not text:
            return ""

        # Format as blockquote
        lines = text.split("\n")
        formatted_lines = [f"> {line.strip()}" for line in lines if line.strip()]
        return "\n".join(formatted_lines)


class TableProcessor:
    """
    HTML table processor with structure preservation.
    """

    def __init__(self) -> None:
        logger.info("TableProcessor initialized")

    async def extract_tables(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Extract tables from HTML with structure preservation.

        Args:
            html_content: HTML content to process

        Returns:
            List of table dictionaries with structure and data
        """
        if not HAS_SELECTOLAX or not html_content:
            return []

        try:
            parser = LexborHTMLParser(html_content) if LexborHTMLParser else HTMLParser(html_content)
            tables = parser.css("table")

            extracted_tables: List[Dict[str, Any]] = []

            for i, table in enumerate(tables):
                table_data = await self._process_table(table, i)
                if table_data:
                    extracted_tables.append(table_data)

            return extracted_tables

        except Exception as e:
            logger.error(f"Table extraction error: {e}")
            return []

    async def _process_table(self, table_element: Any, table_index: int) -> Optional[Dict[str, Any]]:
        """Process individual table element."""
        try:
            # Extract table metadata
            table_data: Dict[str, Any] = {
                "index": table_index,
                "headers": [],
                "rows": [],
                "caption": "",
                "summary": "",
                "structure": {
                    "row_count": 0,
                    "column_count": 0,
                    "has_header": False,
                    "has_footer": False,
                },
            }

            # Extract caption
            caption = table_element.css_first("caption")
            if caption:
                table_data["caption"] = str(caption.text(strip=True))

            # Extract summary attribute
            attrs = getattr(table_element, "attrs", {})
            summary = attrs.get("summary", "") if attrs else ""
            if summary:
                table_data["summary"] = str(summary)

            # Extract headers
            headers: List[str] = []
            header_rows = table_element.css("thead tr, tr:first-child")

            if header_rows:
                first_row = header_rows[0]
                header_cells = first_row.css("th, td")

                for cell in header_cells:
                    header_text = str(cell.text(strip=True))
                    headers.append(header_text)

                if headers and any(headers):  # Has meaningful headers
                    table_data["headers"] = headers
                    table_data["structure"]["has_header"] = True

            # Extract data rows
            rows: List[List[Dict[str, Any]]] = []
            data_rows = table_element.css("tbody tr, tr")

            # Skip header row if we identified headers
            start_index = 1 if table_data["structure"]["has_header"] else 0

            for row_element in data_rows[start_index:]:
                cells = row_element.css("td, th")
                row_data: List[Dict[str, Any]] = []

                for cell in cells:
                    cell_text = str(cell.text(strip=True))

                    # Extract additional cell metadata
                    cell_attrs = getattr(cell, "attrs", {})
                    cell_data: Dict[str, Any] = {
                        "text": cell_text,
                        "colspan": (int(cell_attrs.get("colspan", 1)) if cell_attrs else 1),
                        "rowspan": (int(cell_attrs.get("rowspan", 1)) if cell_attrs else 1),
                    }

                    row_data.append(cell_data)

                if row_data:
                    rows.append(row_data)

            table_data["rows"] = rows

            # Update structure information
            table_data["structure"]["row_count"] = len(rows)
            table_data["structure"]["column_count"] = (
                len(headers) if headers else (max((len(row) for row in rows), default=0) if rows else 0)
            )

            # Check for footer
            tfoot = table_element.css_first("tfoot")
            if tfoot:
                table_data["structure"]["has_footer"] = True

            # Only return tables with meaningful content
            if table_data["structure"]["row_count"] > 0 or table_data["caption"]:
                return table_data

            return None

        except Exception as e:
            logger.error(f"Table processing error: {e}")
            return None

    async def convert_table_to_text(self, table_data: Dict[str, Any]) -> str:
        """Convert table data to readable text format."""
        lines: List[str] = []

        # Add caption
        if table_data.get("caption"):
            lines.append(f"Table: {table_data['caption']}")
            lines.append("")

        # Add headers
        headers = table_data.get("headers")
        if headers:
            header_line = " | ".join(str(h) for h in headers)
            lines.append(header_line)
            lines.append("-" * len(header_line))

        # Add rows
        for row in table_data.get("rows", []):
            row_texts: List[str] = []
            for cell in row:
                if isinstance(cell, dict):
                    row_texts.append(str(cell.get("text", "")))
                else:
                    row_texts.append(str(cell))

            lines.append(" | ".join(row_texts))

        return "\n".join(lines)


class CodeProcessor:
    """
    Code block processor with language detection and syntax analysis.
    """

    def __init__(self) -> None:
        # Common code patterns and indicators
        self.code_indicators: Dict[str, List[str]] = {
            "function_patterns": [
                r"\bdef\s+\w+\s*\(",  # Python
                r"\bfunction\s+\w+\s*\(",  # JavaScript
                r"\b\w+\s*\([^)]*\)\s*{",  # C/Java/etc
            ],
            "variable_patterns": [
                r"\bvar\s+\w+",  # JavaScript
                r"\blet\s+\w+",  # JavaScript/TypeScript
                r"\bconst\s+\w+",  # JavaScript/TypeScript
                r"\bint\s+\w+",  # C/Java
            ],
            "import_patterns": [
                r"\bimport\s+",  # Python/JavaScript
                r"\bfrom\s+\w+\s+import",  # Python
                r"\brequire\s*\(",  # Node.js
                r"\b#include\s*<",  # C/C++
            ],
        }

        # Language-specific patterns
        self.language_patterns: Dict[str, List[str]] = {
            "python": [
                r"\bdef\s+",
                r"\bclass\s+",
                r"\bimport\s+",
                r"\bfrom\s+.*import",
                r"print\s*\(",
                r'if\s+__name__\s*==\s*["\']__main__["\']',
            ],
            "javascript": [
                r"\bfunction\s+",
                r"\bvar\s+",
                r"\blet\s+",
                r"\bconst\s+",
                r"console\.log\s*\(",
                r"document\.",
                r"window\.",
            ],
            "java": [
                r"\bpublic\s+class\s+",
                r"\bpublic\s+static\s+void\s+main",
                r"\bSystem\.out\.",
                r"\bprivate\s+",
                r"\bprotected\s+",
            ],
            "c": [
                r"#include\s*<",
                r"\bint\s+main\s*\(",
                r"\bprintf\s*\(",
                r"\bmalloc\s*\(",
                r"\bfree\s*\(",
            ],
            "sql": [
                r"\bSELECT\s+",
                r"\bFROM\s+",
                r"\bWHERE\s+",
                r"\bINSERT\s+",
                r"\bUPDATE\s+",
                r"\bDELETE\s+",
                r"\bCREATE\s+TABLE",
            ],
            "html": [
                r"<html[^>]*>",
                r"<head[^>]*>",
                r"<body[^>]*>",
                r"<div[^>]*>",
                r"<script[^>]*>",
                r"<style[^>]*>",
            ],
            "css": [
                r"\.[a-zA-Z][a-zA-Z0-9_-]*\s*{",
                r"#[a-zA-Z][a-zA-Z0-9_-]*\s*{",
                r"@media\s+",
                r"@import\s+",
            ],
        }

        logger.info("CodeProcessor initialized")

    async def extract_code_blocks(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from HTML content.

        Args:
            html_content: HTML content to process

        Returns:
            List of code block dictionaries
        """
        if not HAS_SELECTOLAX or not html_content:
            return []

        try:
            parser = LexborHTMLParser(html_content) if LexborHTMLParser else HTMLParser(html_content)
            code_blocks: List[Dict[str, Any]] = []

            # Extract from <pre><code> blocks
            pre_code_blocks = parser.css("pre code, pre, code")
            for i, block in enumerate(pre_code_blocks):
                code_data = await self._process_code_block(block, i, "pre_code")
                if code_data:
                    code_blocks.append(code_data)

            # Extract from syntax highlighting divs (common in documentation)
            syntax_blocks = parser.css(".highlight, .code, .codehilite, .language-*")
            for i, block in enumerate(syntax_blocks):
                code_data = await self._process_code_block(block, i, "syntax_highlight")
                if code_data:
                    code_blocks.append(code_data)

            return code_blocks

        except Exception as e:
            logger.error(f"Code block extraction error: {e}")
            return []

    async def _process_code_block(
        self, code_element: Any, block_index: int, block_type: str
    ) -> Optional[Dict[str, Any]]:
        """Process individual code block element."""
        try:
            code_text = str(code_element.text())
            if not code_text or len(code_text.strip()) < 10:
                return None

            # Extract metadata
            code_data: Dict[str, Any] = {
                "index": block_index,
                "type": block_type,
                "content": code_text.strip(),
                "language": "unknown",
                "confidence": 0.0,
                "features": {
                    "line_count": len(code_text.split("\n")),
                    "char_count": len(code_text),
                    "has_functions": False,
                    "has_imports": False,
                    "has_variables": False,
                },
            }

            # Try to detect language from class attributes
            attrs = getattr(code_element, "attrs", {})
            class_attr = str(attrs.get("class", "")) if attrs else ""
            detected_lang = await self._extract_language_from_class(class_attr)

            if detected_lang:
                code_data["language"] = detected_lang
                code_data["confidence"] = 0.8
            else:
                # Use pattern-based detection
                detected_lang, confidence = await self._detect_language_by_patterns(code_text)
                code_data["language"] = detected_lang
                code_data["confidence"] = confidence

            # Use Pygments for enhanced detection if available
            if HAS_PYGMENTS and code_data["confidence"] < 0.7:
                pygments_lang = await self._detect_with_pygments(code_text)
                if pygments_lang:
                    code_data["language"] = pygments_lang
                    code_data["confidence"] = 0.9

            # Extract code features
            code_data["features"] = await self._analyze_code_features(code_text)

            return code_data

        except Exception as e:
            logger.error(f"Code block processing error: {e}")
            return None

    async def _extract_language_from_class(self, class_attr: str) -> Optional[str]:
        """Extract language from CSS class attributes."""
        if not class_attr:
            return None

        # Common class patterns for language identification
        class_lower = class_attr.lower()

        # Direct language indicators
        for lang in self.language_patterns.keys():
            if f"language-{lang}" in class_lower or f"lang-{lang}" in class_lower:
                return lang
            if f" {lang} " in f" {class_lower} " or class_lower == lang:
                return lang

        # Check for common highlighting library patterns
        if "highlight" in class_lower:
            # Extract language from patterns like "highlight-python"
            match = re.search(r"highlight-(\w+)", class_lower)
            if match:
                return match.group(1)

        return None

    async def _detect_language_by_patterns(self, code_text: str) -> Tuple[str, float]:
        """Detect programming language using pattern matching."""
        code_text.lower()
        language_scores: Dict[str, float] = {}

        # Score each language based on pattern matches
        for language, patterns in self.language_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = len(re.findall(pattern, code_text, re.IGNORECASE))
                score += matches

            if score > 0:
                # Normalize by code length
                normalized_score = score / max(1, len(code_text) / 100)
                language_scores[language] = normalized_score

        if not language_scores:
            return "unknown", 0.0

        # Get best match
        best_language = max(language_scores, key=lambda k: language_scores[k])
        confidence = min(1.0, language_scores[best_language] / 10.0)  # Scale confidence

        return best_language, confidence

    async def _detect_with_pygments(self, code_text: str) -> Optional[str]:
        """Use Pygments for language detection."""
        if not HAS_PYGMENTS:
            return None

        try:
            lexer = guess_lexer(code_text)
            aliases = getattr(lexer, "aliases", [])
            return str(aliases[0]) if aliases else None
        except ClassNotFound:
            return None
        except Exception as e:
            logger.debug(f"Pygments detection error: {e}")
            return None

    async def _analyze_code_features(self, code_text: str) -> Dict[str, Any]:
        """Analyze code features and characteristics."""
        features: Dict[str, Any] = {
            "line_count": len(code_text.split("\n")),
            "char_count": len(code_text),
            "has_functions": False,
            "has_imports": False,
            "has_variables": False,
            "has_comments": False,
            "indentation_style": "unknown",
        }

        # Check for functions
        for pattern in self.code_indicators["function_patterns"]:
            if re.search(pattern, code_text, re.IGNORECASE):
                features["has_functions"] = True
                break

        # Check for imports
        for pattern in self.code_indicators["import_patterns"]:
            if re.search(pattern, code_text, re.IGNORECASE):
                features["has_imports"] = True
                break

        # Check for variables
        for pattern in self.code_indicators["variable_patterns"]:
            if re.search(pattern, code_text, re.IGNORECASE):
                features["has_variables"] = True
                break

        # Check for comments
        comment_patterns = [r"//.*$", r"/\*.*?\*/", r"#.*$", r"<!--.*?-->"]
        for pattern in comment_patterns:
            if re.search(pattern, code_text, re.MULTILINE | re.DOTALL):
                features["has_comments"] = True
                break

        # Detect indentation style
        lines = code_text.split("\n")
        indent_counts = {"spaces": 0, "tabs": 0}

        for line in lines:
            if line.startswith("    "):  # 4 spaces
                indent_counts["spaces"] += 1
            elif line.startswith("\t"):  # Tab
                indent_counts["tabs"] += 1

        if indent_counts["spaces"] > indent_counts["tabs"]:
            features["indentation_style"] = "spaces"
        elif indent_counts["tabs"] > 0:
            features["indentation_style"] = "tabs"

        return features

    def _extract_with_pygments(self, text: str, language: str) -> str:
        """Extract code with Pygments syntax highlighting."""
        try:
            from pygments import highlight
            from pygments.formatters import get_formatter_by_name
            from pygments.lexers import get_lexer_by_name

            lexer = get_lexer_by_name(language, stripall=True)
            formatter = get_formatter_by_name("text")  # type: ignore[no-untyped-call]

            highlighted = highlight(text, lexer, formatter)
            return str(highlighted) if highlighted else text

        except Exception as e:
            logger.warning(f"Pygments highlighting failed: {e}")
            return text

    def _clean_code_text(self, text: str) -> str:
        """Clean and format code text."""
        if not text:
            return ""

        # Remove excessive whitespace while preserving structure
        lines = text.split("\n")
        cleaned_lines: List[str] = []

        for line in lines:
            # Keep indentation but clean up extra spaces
            stripped = line.rstrip()
            if stripped or (cleaned_lines and cleaned_lines[-1]):  # Keep empty lines between content
                cleaned_lines.append(stripped)

        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1]:
            cleaned_lines.pop()

        result = "\n".join(cleaned_lines)
        return str(result) if result else ""


class ImageProcessor:
    """
    Image processor for extracting alt-text, captions, and metadata.
    """

    def __init__(self) -> None:
        logger.info("ImageProcessor initialized")

    async def extract_images(self, html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract image information from HTML content.

        Args:
            html_content: HTML content to process
            base_url: Base URL for resolving relative URLs

        Returns:
            List of image dictionaries with metadata
        """
        if not HAS_SELECTOLAX or not html_content:
            return []

        try:
            parser = LexborHTMLParser(html_content) if LexborHTMLParser else HTMLParser(html_content)
            images = parser.css("img")

            extracted_images: List[Dict[str, Any]] = []

            for i, img in enumerate(images):
                image_data = await self._process_image(img, i, base_url)
                if image_data:
                    extracted_images.append(image_data)

            return extracted_images

        except Exception as e:
            logger.error(f"Image extraction error: {e}")
            return []

    async def _process_image(self, img_element: Any, image_index: int, base_url: str) -> Optional[Dict[str, Any]]:
        """Process individual image element."""
        try:
            attrs = getattr(img_element, "attrs", {})
            src = str(attrs.get("src", "")) if attrs else ""
            if not src:
                return None

            # Resolve relative URLs
            if src.startswith("//"):
                src = "https:" + src
            elif src.startswith("/"):
                src = urljoin(base_url, src)
            elif not src.startswith(("http://", "https://")):
                src = urljoin(base_url, src)

            image_data: Dict[str, Any] = {
                "index": image_index,
                "src": src,
                "alt": str(attrs.get("alt", "")) if attrs else "",
                "title": str(attrs.get("title", "")) if attrs else "",
                "width": str(attrs.get("width", "")) if attrs else "",
                "height": str(attrs.get("height", "")) if attrs else "",
                "caption": "",
                "context": "",
                "type": await self._classify_image_type(img_element, src),
            }

            # Extract caption from nearby elements
            caption = await self._extract_image_caption(img_element)
            if caption:
                image_data["caption"] = caption

            # Extract context from surrounding text
            context = await self._extract_image_context(img_element)
            if context:
                image_data["context"] = context

            return image_data

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return None

    async def _extract_image_caption(self, img_element: Any) -> str:
        """Extract caption from nearby elements."""
        # Check parent figure element
        parent = getattr(img_element, "parent", None)
        if parent and hasattr(parent, "tag") and parent.tag == "figure":
            figcaption = parent.css_first("figcaption")
            if figcaption:
                return str(figcaption.text(strip=True))

        # Check for caption class nearby
        if parent:
            caption_element = parent.css_first(".caption, .image-caption, .figure-caption")
            if caption_element:
                return str(caption_element.text(strip=True))

        return ""

    async def _extract_image_context(self, img_element: Any) -> str:
        """Extract context from surrounding text."""
        context_parts: List[str] = []

        # Get parent element
        parent = getattr(img_element, "parent", None)
        if not parent:
            return ""

        # Extract text from parent
        parent_text = str(parent.text(strip=True)) if hasattr(parent, "text") else ""
        if parent_text and len(parent_text) > 20:
            context_parts.append(parent_text[:200])  # Limit context length

        return " ".join(context_parts)

    async def _classify_image_type(self, img_element: Any, src: str) -> str:
        """Classify image type based on attributes and source."""
        # Check file extension
        parsed_url = urlparse(src.lower())
        path = parsed_url.path

        if any(ext in path for ext in [".svg"]):
            return "icon"
        elif any(ext in path for ext in [".gif"]):
            return "animation"
        elif any(ext in path for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            # Check size attributes to classify further
            attrs = getattr(img_element, "attrs", {})
            width = attrs.get("width", "") if attrs else ""
            height = attrs.get("height", "") if attrs else ""

            try:
                w = int(width) if width else 0
                h = int(height) if height else 0

                if w > 0 and h > 0:
                    if w < 50 or h < 50:
                        return "icon"
                    elif w > 800 or h > 600:
                        return "photo"
                    else:
                        return "graphic"
            except ValueError:
                pass

            return "photo"

        return "unknown"


class LinkProcessor:
    """
    Link processor for classification and metadata extraction.
    """

    def __init__(self) -> None:
        logger.info("LinkProcessor initialized")

    async def extract_links(self, html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """
        Extract links from HTML content with classification.

        Args:
            html_content: HTML content to process
            base_url: Base URL for resolving relative URLs

        Returns:
            List of link dictionaries with metadata
        """
        if not HAS_SELECTOLAX or not html_content:
            return []

        try:
            parser = LexborHTMLParser(html_content) if LexborHTMLParser else HTMLParser(html_content)
            links = parser.css("a[href]")

            extracted_links: List[Dict[str, Any]] = []
            base_domain = urlparse(base_url).netloc

            for i, link in enumerate(links):
                link_data = await self._process_link(link, i, base_url, base_domain)
                if link_data:
                    extracted_links.append(link_data)

            return extracted_links

        except Exception as e:
            logger.error(f"Link extraction error: {e}")
            return []

    async def _process_link(
        self, link_element: Any, link_index: int, base_url: str, base_domain: str
    ) -> Optional[Dict[str, Any]]:
        """Process individual link element."""
        try:
            attrs = getattr(link_element, "attrs", {})
            href = str(attrs.get("href", "")) if attrs else ""
            if not href:
                return None

            # Resolve relative URLs
            if href.startswith("//"):
                href = "https:" + href
            elif href.startswith("/"):
                href = urljoin(base_url, href)
            elif not href.startswith(("http://", "https://", "mailto:", "tel:")):
                href = urljoin(base_url, href)

            link_text = str(link_element.text(strip=True)) if hasattr(link_element, "text") else ""

            link_data: Dict[str, Any] = {
                "index": link_index,
                "href": href,
                "text": link_text,
                "title": str(attrs.get("title", "")) if attrs else "",
                "rel": str(attrs.get("rel", "")) if attrs else "",
                "target": str(attrs.get("target", "")) if attrs else "",
                "type": await self._classify_link(link_element, href, link_text),
                "category": await self._classify_link(
                    link_element, href, link_text
                ),  # Add category field for test compatibility
                "is_external": urlparse(href).netloc != base_domain,
                "context": await self._extract_link_context(link_element),
            }

            return link_data

        except Exception as e:
            logger.error(f"Link processing error: {e}")
            return None

    async def _classify_link(self, link_element: Any, href: str, link_text: str) -> str:
        """Classify link type based on href and content."""
        href_lower = href.lower()
        text_lower = link_text.lower()

        # Email links
        if href_lower.startswith("mailto:"):
            return "email"

        # Phone links
        if href_lower.startswith("tel:"):
            return "phone"

        # File downloads
        file_extensions = [
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".zip",
            ".rar",
        ]
        if any(ext in href_lower for ext in file_extensions):
            return "download"

        # Source code repositories
        source_code_domains = [
            "github.com",
            "gitlab.com",
            "bitbucket.org",
            "sourceforge.net",
        ]
        if any(domain in href_lower for domain in source_code_domains):
            return "source_code"

        # Social media
        social_domains = [
            "facebook.com",
            "twitter.com",
            "linkedin.com",
            "instagram.com",
            "youtube.com",
        ]
        if any(domain in href_lower for domain in social_domains):
            return "social"

        # Documentation
        if "documentation" in text_lower or "/docs/" in href_lower or "manual" in text_lower:
            return "documentation"

        # Navigation based on text
        nav_keywords = ["home", "about", "contact", "menu", "next", "previous", "back"]
        if any(keyword in text_lower for keyword in nav_keywords):
            return "navigation"

        # Internal anchors
        if href.startswith("#"):
            return "anchor"

        return "content"

    async def _extract_link_context(self, link_element: Any) -> str:
        """Extract context from surrounding text."""
        # Get parent element
        parent = getattr(link_element, "parent", None)
        if not parent:
            return ""

        # Extract text from parent, excluding the link text itself
        parent_text = str(parent.text(strip=True)) if hasattr(parent, "text") else ""
        link_text = str(link_element.text(strip=True)) if hasattr(link_element, "text") else ""

        # Remove link text from context
        context = parent_text.replace(link_text, "").strip()

        # Limit context length
        return context[:200] if context else ""
