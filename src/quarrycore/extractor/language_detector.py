"""
Language Detection with fastText and Fallback Methods

Provides accurate language detection for 50+ languages with multiple
detection strategies and confidence scoring.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports with graceful fallbacks
try:
    import fasttext  # type: ignore[import-not-found]

    HAS_FASTTEXT = True
except ImportError:
    fasttext = None
    HAS_FASTTEXT = False

try:
    import langdetect  # type: ignore[import-not-found]
    from langdetect import LangDetectException, detect, detect_langs

    HAS_LANGDETECT = True
except ImportError:
    langdetect = None
    detect = None
    detect_langs = None
    LangDetectException = Exception
    HAS_LANGDETECT = False

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Multi-strategy language detection with high accuracy.

    Features:
    - Primary: fastText model for high accuracy
    - Secondary: langdetect library as fallback
    - Tertiary: Pattern-based detection for common languages
    - Support for 50+ languages with confidence scoring
    """

    # Common language patterns for fallback detection
    LANGUAGE_PATTERNS: Dict[str, List[str]] = {
        "en": [
            r"\b(the|and|or|but|in|on|at|to|for|of|with|by)\b",
            r"\b(this|that|these|those|what|where|when|why|how)\b",
        ],
        "es": [
            r"\b(el|la|los|las|y|o|pero|en|de|con|por|para)\b",
            r"\b(que|como|cuando|donde|quien|cual)\b",
        ],
        "fr": [
            r"\b(le|la|les|et|ou|mais|dans|de|avec|par|pour)\b",
            r"\b(que|comme|quand|où|qui|quel)\b",
        ],
        "de": [
            r"\b(der|die|das|und|oder|aber|in|auf|mit|von|für)\b",
            r"\b(was|wie|wann|wo|wer|welch)\b",
        ],
        "it": [
            r"\b(il|la|lo|gli|e|o|ma|in|di|con|per)\b",
            r"\b(che|come|quando|dove|chi|quale)\b",
        ],
        "pt": [
            r"\b(o|a|os|as|e|ou|mas|em|de|com|por|para)\b",
            r"\b(que|como|quando|onde|quem|qual)\b",
        ],
        "ru": [
            r"\b(и|или|но|в|на|с|по|для|от|до)\b",
            r"\b(что|как|когда|где|кто|какой)\b",
        ],
        "zh": [
            r"[的|和|或|但是|在|与|由|为]",
            r"[什么|如何|何时|哪里|谁|哪个]",
        ],
        "ja": [
            r"[の|と|や|が|を|に|で|から|まで]",
            r"[何|どう|いつ|どこ|誰|どの]",
        ],
        "ar": [
            r"\b(في|من|إلى|على|مع|عن|بعد|قبل)\b",
            r"\b(ما|كيف|متى|أين|من|أي)\b",
        ],
    }

    # Language code to full name mapping
    LANGUAGE_NAMES: Dict[str, str] = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ar": "Arabic",
        "hi": "Hindi",
        "ko": "Korean",
        "th": "Thai",
        "vi": "Vietnamese",
        "tr": "Turkish",
        "pl": "Polish",
        "nl": "Dutch",
        "sv": "Swedish",
        "da": "Danish",
        "no": "Norwegian",
        "fi": "Finnish",
        "cs": "Czech",
        "sk": "Slovak",
        "hu": "Hungarian",
        "ro": "Romanian",
        "bg": "Bulgarian",
        "hr": "Croatian",
        "sr": "Serbian",
        "sl": "Slovenian",
        "et": "Estonian",
        "lv": "Latvian",
        "lt": "Lithuanian",
        "el": "Greek",
        "he": "Hebrew",
        "fa": "Persian",
        "ur": "Urdu",
        "bn": "Bengali",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "kn": "Kannada",
        "gu": "Gujarati",
        "pa": "Punjabi",
        "mr": "Marathi",
        "ne": "Nepali",
        "si": "Sinhala",
        "my": "Burmese",
        "km": "Khmer",
        "lo": "Lao",
        "ka": "Georgian",
        "am": "Amharic",
        "sw": "Swahili",
        "zu": "Zulu",
        "af": "Afrikaans",
    }

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = model_path
        self._fasttext_model: Optional[Any] = None
        self._model_loaded = False

        # Statistics
        self._detection_stats: Dict[str, Any] = {
            "total_detections": 0,
            "fasttext_used": 0,
            "langdetect_used": 0,
            "pattern_used": 0,
            "language_distribution": {},
        }

        logger.info("LanguageDetector initialized")

    async def detect_language(
        self,
        text: str,
        *,
        min_confidence: float = 0.5,
        fallback_to_patterns: bool = True,
    ) -> str:
        """
        Detect language of given text.

        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold
            fallback_to_patterns: Whether to use pattern-based fallback

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr')
        """
        if not text or len(text.strip()) < 10:
            return "en"  # Default to English for short texts

        self._detection_stats["total_detections"] += 1

        # Clean text for detection
        clean_text = self._clean_text_for_detection(text)

        # Strategy 1: fastText (most accurate)
        if HAS_FASTTEXT:
            result = await self._detect_with_fasttext(clean_text, min_confidence)
            if result:
                self._detection_stats["fasttext_used"] += 1
                self._update_language_stats(result)
                return result

        # Strategy 2: langdetect library
        if HAS_LANGDETECT:
            result = await self._detect_with_langdetect(clean_text, min_confidence)
            if result:
                self._detection_stats["langdetect_used"] += 1
                self._update_language_stats(result)
                return result

        # Strategy 3: Pattern-based detection
        if fallback_to_patterns:
            result = await self._detect_with_patterns(clean_text)
            if result:
                self._detection_stats["pattern_used"] += 1
                self._update_language_stats(result)
                return result

        # Default fallback
        self._update_language_stats("en")
        return "en"

    async def detect_language_with_confidence(
        self,
        text: str,
        *,
        return_top_n: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Detect language with confidence scores.

        Args:
            text: Text to analyze
            return_top_n: Number of top predictions to return

        Returns:
            List of (language_code, confidence) tuples
        """
        if not text or len(text.strip()) < 10:
            return [("en", 1.0)]

        clean_text = self._clean_text_for_detection(text)
        results: List[Tuple[str, float]] = []

        # Try fastText first
        if HAS_FASTTEXT:
            results = await self._detect_multiple_with_fasttext(clean_text, return_top_n)
            if results:
                return results

        # Try langdetect
        if HAS_LANGDETECT:
            results = await self._detect_multiple_with_langdetect(clean_text, return_top_n)
            if results:
                return results

        # Pattern-based fallback
        pattern_result = await self._detect_with_patterns(clean_text)
        if pattern_result:
            return [(pattern_result, 0.7)]

        return [("en", 0.5)]

    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection."""
        # Remove URLs
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )

        # Remove email addresses
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # Remove numbers and special characters for better detection
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    async def _detect_with_fasttext(self, text: str, min_confidence: float) -> Optional[str]:
        """Detect language using fastText model."""
        if not HAS_FASTTEXT or not fasttext:
            return None

        try:
            if not self._model_loaded:
                await self._load_fasttext_model()

            if not self._fasttext_model:
                return None

            # Predict language
            predictions = self._fasttext_model.predict(text, k=1)

            if predictions and len(predictions[0]) > 0:
                # Extract language code from fastText label (e.g., '__label__en' -> 'en')
                lang_label = str(predictions[0][0])
                lang_code = lang_label.replace("__label__", "")
                confidence = float(predictions[1][0])

                if confidence >= min_confidence:
                    return lang_code

        except Exception as e:
            logger.warning(f"fastText detection error: {e}")

        return None

    async def _detect_multiple_with_fasttext(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Get multiple language predictions from fastText."""
        if not HAS_FASTTEXT or not fasttext:
            return []

        try:
            if not self._model_loaded:
                await self._load_fasttext_model()

            if not self._fasttext_model:
                return []

            predictions = self._fasttext_model.predict(text, k=top_n)

            results: List[Tuple[str, float]] = []
            for label, confidence in zip(predictions[0], predictions[1], strict=False):
                lang_code = str(label).replace("__label__", "")
                results.append((lang_code, float(confidence)))

            return results

        except Exception as e:
            logger.warning(f"fastText multiple detection error: {e}")
            return []

    async def _load_fasttext_model(self) -> None:
        """Load fastText language identification model."""
        if not HAS_FASTTEXT or not fasttext:
            self._model_loaded = True
            return

        try:
            if self.model_path:
                # Load custom model
                self._fasttext_model = fasttext.load_model(self.model_path)
            else:
                # Try to load pre-trained model (would need to be downloaded)
                # For now, we'll skip this and use other methods
                pass

            self._model_loaded = True
            logger.info("fastText model loaded successfully")

        except Exception as e:
            logger.warning(f"Failed to load fastText model: {e}")
            self._model_loaded = True  # Mark as attempted

    async def _detect_with_langdetect(self, text: str, min_confidence: float) -> Optional[str]:
        """Detect language using langdetect library."""
        if not HAS_LANGDETECT or not detect or not detect_langs:
            return None

        try:
            # Simple detection
            detected_lang = detect(text)

            # Get confidence by trying detect_langs
            try:
                lang_probs = detect_langs(text)
                if lang_probs:
                    confidence = float(lang_probs[0].prob)
                    if confidence >= min_confidence:
                        return str(detected_lang)
            except LangDetectException:
                # Fallback to simple detection with lower confidence requirement
                if min_confidence <= 0.7:
                    return str(detected_lang)

        except Exception as e:
            logger.debug(f"langdetect error: {e}")

        return None

    async def _detect_multiple_with_langdetect(self, text: str, top_n: int) -> List[Tuple[str, float]]:
        """Get multiple language predictions from langdetect."""
        if not HAS_LANGDETECT or not detect_langs:
            return []

        try:
            lang_probs = detect_langs(text)
            results: List[Tuple[str, float]] = []

            for lang_prob in lang_probs[:top_n]:
                results.append((str(lang_prob.lang), float(lang_prob.prob)))

            return results

        except Exception as e:
            logger.debug(f"langdetect multiple detection error: {e}")
            return []

    async def _detect_with_patterns(self, text: str) -> Optional[str]:
        """Detect language using pattern matching."""
        text_lower = text.lower()

        # Score each language based on pattern matches
        language_scores: Dict[str, float] = {}

        for lang_code, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0.0
            text_length = len(text_lower)

            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                # Normalize by text length
                score += matches / max(1, text_length / 1000)

            if score > 0:
                language_scores[lang_code] = score

        # Return language with highest score if above threshold
        if language_scores:
            best_lang = max(language_scores, key=lambda k: language_scores[k])
            if language_scores[best_lang] > 0.1:  # Minimum pattern threshold
                return best_lang

        return None

    def _update_language_stats(self, lang_code: str) -> None:
        """Update language detection statistics."""
        lang_dist = self._detection_stats["language_distribution"]
        if lang_code in lang_dist:
            lang_dist[lang_code] = int(lang_dist[lang_code]) + 1
        else:
            lang_dist[lang_code] = 1

    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code."""
        return self.LANGUAGE_NAMES.get(lang_code, lang_code.upper())

    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.LANGUAGE_NAMES.keys())

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get language detection statistics."""
        stats = self._detection_stats.copy()

        # Add method usage percentages
        total = int(stats["total_detections"])
        if total > 0:
            stats["method_usage"] = {
                "fasttext_percentage": (int(stats["fasttext_used"]) / total) * 100,
                "langdetect_percentage": (int(stats["langdetect_used"]) / total) * 100,
                "pattern_percentage": (int(stats["pattern_used"]) / total) * 100,
            }

        return stats

    async def is_language_supported(self, lang_code: str) -> bool:
        """Check if language is supported."""
        return lang_code in self.LANGUAGE_NAMES

    async def detect_encoding_and_language(self, content: bytes) -> Tuple[str, str]:
        """Detect both encoding and language from raw content."""
        # Try to decode with common encodings
        encodings = ["utf-8", "latin-1", "windows-1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                text = content.decode(encoding)
                lang_code = await self.detect_language(text)
                return encoding, lang_code
            except UnicodeDecodeError:
                continue

        # Fallback
        try:
            text = content.decode("utf-8", errors="replace")
            lang_code = await self.detect_language(text)
            return "utf-8", lang_code
        except Exception:
            return "utf-8", "en"
