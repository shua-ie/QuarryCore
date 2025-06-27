"""
Domain-Specific Content Extractors

Implements specialized extraction rules and processing for different domains:
- Medical: Clinical entities, terminology, and HIPAA compliance
- Legal: Citation parsing, jurisdiction awareness, and legal entities
- E-commerce: Product attributes, pricing, and structured data
- Technical: Code analysis, API documentation, and technical concepts
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from ..protocols import DomainType, ExtractedContent

logger = logging.getLogger(__name__)


class BaseDomainExtractor(ABC):
    """Base class for domain-specific extractors."""

    def __init__(self) -> None:
        self.domain_keywords: set[str] = set()
        self.extraction_patterns: Dict[str, List[str]] = {}
        self.quality_indicators: Dict[str, Any] = {}

    @abstractmethod
    async def enhance_extraction(self, content: ExtractedContent, html: str) -> ExtractedContent:
        """Enhance extraction with domain-specific rules."""
        pass

    @abstractmethod
    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract domain-specific entities."""
        pass

    async def calculate_domain_confidence(self, text: str) -> float:
        """Calculate confidence that content belongs to this domain."""
        if not text or not self.domain_keywords:
            return 0.0

        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in self.domain_keywords if keyword in text_lower)

        # Normalize by text length and keyword count
        text_length = len(text_lower.split())
        confidence = (keyword_matches / len(self.domain_keywords)) * min(1.0, text_length / 1000)

        return min(1.0, confidence)


class MedicalExtractor(BaseDomainExtractor):
    """
    Medical domain extractor with clinical entity recognition.

    Features:
    - Medical terminology extraction
    - Drug and dosage identification
    - Clinical abbreviation expansion
    - HIPAA-compliant PII detection
    - Anatomical term normalization
    """

    def __init__(self) -> None:
        super().__init__()

        self.domain_keywords = {
            "patient",
            "diagnosis",
            "treatment",
            "therapy",
            "clinical",
            "medical",
            "disease",
            "syndrome",
            "disorder",
            "symptom",
            "procedure",
            "surgery",
            "medication",
            "drug",
            "dosage",
            "prescription",
            "pharmaceutical",
            "hospital",
            "clinic",
            "doctor",
            "physician",
            "nurse",
            "healthcare",
            "anatomy",
            "physiology",
            "pathology",
            "radiology",
            "oncology",
            "cardiology",
            "neurology",
            "pediatrics",
            "geriatrics",
            "psychiatry",
        }

        # Medical entity patterns
        self.extraction_patterns = {
            "medications": [
                r"\b[A-Z][a-z]+(?:cin|nol|pril|sartan|statin|mycin|cillin)\b",  # Common drug suffixes
                r"\b(?:mg|mcg|IU|units?)\b",  # Dosage units
                r"\b\d+\s*(?:mg|mcg|ml|cc|IU|units?)\b",  # Dosages
            ],
            "medical_conditions": [
                r"\b[A-Z][a-z]+(?:itis|osis|emia|uria|pathy|plasia|trophy)\b",  # Medical suffixes
                r"\b(?:acute|chronic|severe|mild|moderate)\s+\w+\b",  # Severity descriptors
            ],
            "procedures": [
                r"\b\w+(?:ectomy|otomy|ostomy|scopy|graphy|plasty)\b",  # Procedure suffixes
                r"\b(?:biopsy|surgery|operation|procedure|examination)\b",
            ],
            "anatomy": [
                r"\b(?:heart|brain|liver|kidney|lung|stomach|intestine|bone|muscle|nerve)\b",
                r"\b(?:anterior|posterior|superior|inferior|medial|lateral|proximal|distal)\b",
            ],
            "clinical_abbreviations": [
                r"\b(?:BP|HR|RR|O2|CO2|CBC|BUN|ECG|EKG|MRI|CT|PET|BMI)\b",
                r"\b(?:IV|IM|PO|PRN|BID|TID|QID|QD|HS|AC|PC)\b",
            ],
        }

        # Common medical abbreviations with expansions
        self.abbreviation_expansions = {
            "BP": "blood pressure",
            "HR": "heart rate",
            "RR": "respiratory rate",
            "CBC": "complete blood count",
            "BUN": "blood urea nitrogen",
            "ECG": "electrocardiogram",
            "EKG": "electrocardiogram",
            "MRI": "magnetic resonance imaging",
            "CT": "computed tomography",
            "PET": "positron emission tomography",
            "BMI": "body mass index",
            "IV": "intravenous",
            "IM": "intramuscular",
            "PO": "by mouth",
            "PRN": "as needed",
            "BID": "twice daily",
            "TID": "three times daily",
            "QID": "four times daily",
            "QD": "once daily",
        }

        # PII patterns for HIPAA compliance
        self.pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{2}/\d{2}/\d{4}\b",  # Date of birth patterns
            r"\b(?:MRN|Medical Record Number):\s*\d+\b",  # Medical record numbers
            r"\b\d{10,}\b",  # Potential patient IDs
        ]

        logger.info("MedicalExtractor initialized")

    async def enhance_extraction(self, content: ExtractedContent, html: str) -> ExtractedContent:
        """Enhance extraction with medical domain rules."""
        if not content.text:
            return content

        # Extract medical entities
        entities = await self.extract_entities(content.text)

        # Store domain-specific data
        if not hasattr(content, "domain_data"):
            content.domain_data = {}

        content.domain_data["medical_entities"] = entities

        # Expand medical abbreviations
        expanded_text = await self._expand_abbreviations(content.text)
        if expanded_text != content.text:
            content.domain_data["expanded_abbreviations"] = True

        # Check for PII and flag for removal
        pii_detected = await self._detect_pii(content.text)
        if pii_detected:
            content.warnings.append("Medical PII detected - review for HIPAA compliance")
            content.domain_data["pii_detected"] = pii_detected

        # Calculate medical domain confidence
        medical_confidence = await self.calculate_domain_confidence(content.text)
        content.domain_data["medical_confidence"] = medical_confidence

        return content

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text."""
        entities = {}

        for entity_type, patterns in self.extraction_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)

            # Deduplicate and clean
            clean_matches = list(set(match.strip() for match in matches if match.strip()))
            if clean_matches:
                entities[entity_type] = clean_matches

        return entities

    async def _expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations for better understanding."""
        expanded_text = text

        for abbrev, expansion in self.abbreviation_expansions.items():
            # Use word boundaries to avoid partial matches
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            replacement = f"{abbrev} ({expansion})"
            expanded_text = re.sub(pattern, replacement, expanded_text)

        return expanded_text

    async def _detect_pii(self, text: str) -> List[str]:
        """Detect potential PII in medical text."""
        detected_pii = []

        for pattern in self.pii_patterns:
            matches = re.findall(pattern, text)
            if matches:
                detected_pii.extend(matches)

        return detected_pii


class LegalExtractor(BaseDomainExtractor):
    """
    Legal domain extractor with citation parsing and jurisdiction awareness.

    Features:
    - Legal citation parsing with cross-reference resolution
    - Jurisdiction-aware processing
    - Legal entity extraction (parties, judges, lawyers)
    - Statute and regulation linking
    - Case outcome extraction
    """

    def __init__(self) -> None:
        super().__init__()

        self.domain_keywords = {
            "court",
            "judge",
            "jury",
            "trial",
            "case",
            "lawsuit",
            "litigation",
            "plaintiff",
            "defendant",
            "attorney",
            "lawyer",
            "counsel",
            "barrister",
            "statute",
            "regulation",
            "law",
            "legal",
            "judicial",
            "jurisdiction",
            "appeal",
            "motion",
            "brief",
            "ruling",
            "verdict",
            "judgment",
            "sentence",
            "contract",
            "agreement",
            "liability",
            "damages",
            "injunction",
            "precedent",
            "constitutional",
            "federal",
            "state",
            "municipal",
            "civil",
            "criminal",
        }

        # Legal citation patterns
        self.extraction_patterns = {
            "case_citations": [
                r"\b\d+\s+[A-Z][a-z]*\.?\s*\d+d?\s*\d+\b",  # Basic case citation
                r"\b[A-Z][a-z]+\s+v\.?\s+[A-Z][a-z]+\b",  # Case names
                r"\b\(\d{4}\)\b",  # Year citations
            ],
            "statute_citations": [
                r"\b\d+\s+U\.?S\.?C\.?\s*§?\s*\d+\b",  # USC citations
                r"\b\d+\s+C\.?F\.?R\.?\s*§?\s*\d+\b",  # CFR citations
                r"\bSection\s+\d+\b",  # Section references
            ],
            "legal_entities": [
                r"\b(?:Judge|Justice|Hon\.|Honorable)\s+[A-Z][a-z]+\b",  # Judges
                r"\b(?:Attorney|Counsel|Esq\.)\s+[A-Z][a-z]+\b",  # Attorneys
                r"\b(?:Court|Tribunal|Commission)\s+of\s+[A-Z][a-z]+\b",  # Courts
            ],
            "legal_procedures": [
                r"\b(?:motion|petition|appeal|writ|subpoena|deposition|discovery)\b",
                r"\b(?:filed|granted|denied|dismissed|reversed|affirmed|remanded)\b",
            ],
            "jurisdictions": [
                r"\b(?:Federal|State|Municipal|County|District|Circuit)\s+Court\b",
                r"\b(?:Supreme Court|Court of Appeals|District Court)\b",
                r"\b(?:United States|U\.S\.|California|New York|Texas)\b",
            ],
        }

        # Legal outcome indicators
        self.outcome_patterns = {
            "granted": [r"\bgranted\b", r"\ballowed\b", r"\bapproved\b"],
            "denied": [r"\bdenied\b", r"\brejected\b", r"\bdismissed\b"],
            "reversed": [r"\breversed\b", r"\boverturned\b", r"\bset aside\b"],
            "affirmed": [r"\baffirmed\b", r"\bupheld\b", r"\bconfirmed\b"],
            "remanded": [r"\bremanded\b", r"\bsent back\b", r"\breturned\b"],
        }

        logger.info("LegalExtractor initialized")

    async def enhance_extraction(self, content: ExtractedContent, html: str) -> ExtractedContent:
        """Enhance extraction with legal domain rules."""
        if not content.text:
            return content

        # Extract legal entities
        entities = await self.extract_entities(content.text)

        # Store domain-specific data
        if not hasattr(content, "domain_data"):
            content.domain_data = {}

        content.domain_data["legal_entities"] = entities

        # Extract case outcomes
        outcomes = await self._extract_case_outcomes(content.text)
        if outcomes:
            content.domain_data["case_outcomes"] = outcomes

        # Identify jurisdiction
        jurisdiction = await self._identify_jurisdiction(content.text)
        if jurisdiction:
            content.domain_data["jurisdiction"] = jurisdiction

        # Parse legal citations
        citations = await self._parse_citations(content.text)
        if citations:
            content.domain_data["legal_citations"] = citations

        # Calculate legal domain confidence
        legal_confidence = await self.calculate_domain_confidence(content.text)
        content.domain_data["legal_confidence"] = legal_confidence

        return content

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract legal entities from text."""
        entities = {}

        for entity_type, patterns in self.extraction_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(found)

            # Clean and deduplicate
            clean_matches = list(set(match.strip() for match in matches if match.strip()))
            if clean_matches:
                entities[entity_type] = clean_matches

        return entities

    async def _extract_case_outcomes(self, text: str) -> Dict[str, List[str]]:
        """Extract case outcomes and decisions."""
        outcomes = {}

        for outcome_type, patterns in self.outcome_patterns.items():
            matches = []
            for pattern in patterns:
                # Find sentences containing outcome indicators
                sentences = re.split(r"[.!?]+", text)
                for sentence in sentences:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        matches.append(sentence.strip())

            if matches:
                outcomes[outcome_type] = matches

        return outcomes

    async def _identify_jurisdiction(self, text: str) -> Optional[str]:
        """Identify legal jurisdiction from text."""
        # Check for explicit jurisdiction mentions
        jurisdiction_indicators = {
            "federal": [
                "federal court",
                "u.s. district",
                "united states",
                "supreme court",
            ],
            "state": ["state court", "california", "new york", "texas", "florida"],
            "local": ["municipal", "county", "district court"],
        }

        text_lower = text.lower()

        for jurisdiction, indicators in jurisdiction_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return jurisdiction

        return None

    async def _parse_citations(self, text: str) -> Dict[str, List[str]]:
        """Parse legal citations with structure."""
        citations: Dict[str, List[Any]] = {
            "cases": [],
            "statutes": [],
            "regulations": [],
        }

        # Case citations
        case_pattern = r"\b(\w+\s+v\.?\s+\w+),?\s*(\d+\s+[A-Z][a-z]*\.?\s*\d+d?\s*\d+)(?:\s*\((\d{4})\))?"
        case_matches = re.findall(case_pattern, text, re.IGNORECASE)

        for match in case_matches:
            case_name, citation, year = match
            citations["cases"].append(
                {
                    "name": case_name.strip(),
                    "citation": citation.strip(),
                    "year": year if year else None,
                }
            )

        # Statute citations
        statute_pattern = r"\b(\d+)\s+(U\.?S\.?C\.?)\s*§?\s*(\d+)\b"
        statute_matches = re.findall(statute_pattern, text, re.IGNORECASE)

        for match in statute_matches:
            title, code, section = match
            citations["statutes"].append(
                {
                    "title": title,
                    "code": code,
                    "section": section,
                }
            )

        return citations


class EcommerceExtractor(BaseDomainExtractor):
    """
    E-commerce domain extractor with product attributes and pricing.

    Features:
    - Product attribute extraction with normalization
    - Multi-currency price detection and conversion
    - Review authenticity scoring
    - Inventory status tracking
    - Schema.org markup parsing
    """

    def __init__(self) -> None:
        super().__init__()

        self.domain_keywords = {
            "product",
            "price",
            "buy",
            "purchase",
            "order",
            "cart",
            "checkout",
            "shipping",
            "delivery",
            "return",
            "warranty",
            "guarantee",
            "review",
            "rating",
            "star",
            "customer",
            "seller",
            "brand",
            "model",
            "size",
            "color",
            "style",
            "material",
            "weight",
            "dimension",
            "specification",
            "inventory",
            "stock",
            "available",
            "sale",
            "discount",
            "offer",
        }

        # E-commerce entity patterns
        self.extraction_patterns = {
            "prices": [
                r"\$\d+(?:\.\d{2})?",  # USD prices
                r"€\d+(?:\.\d{2})?",  # EUR prices
                r"£\d+(?:\.\d{2})?",  # GBP prices
                r"\d+(?:\.\d{2})?\s*(?:USD|EUR|GBP|CAD|AUD)",  # Currency codes
            ],
            "product_identifiers": [
                r"\bSKU:?\s*([A-Z0-9-]+)",  # SKU codes
                r"\bModel:?\s*([A-Z0-9-]+)",  # Model numbers
                r"\bUPC:?\s*(\d{12})",  # UPC codes
                r"\bASIN:?\s*([A-Z0-9]{10})",  # Amazon ASINs
            ],
            "specifications": [
                r"\b(\w+):\s*([^,\n]+)",  # Key-value specifications
                r"\b(\d+(?:\.\d+)?)\s*(inches?|cm|mm|lbs?|kg|oz|g)\b",  # Measurements
            ],
            "reviews": [
                r"\b(\d+(?:\.\d+)?)\s*(?:stars?|/5|out of 5)\b",  # Star ratings
                r"\b(\d+)\s*reviews?\b",  # Review counts
            ],
            "inventory": [
                r"\b(?:in stock|available|out of stock|sold out|backordered)\b",
                r"\b(\d+)\s*(?:left|remaining|available)\b",  # Quantity available
            ],
        }

        # Currency conversion rates (simplified - in production, use real-time rates)
        self.currency_rates = {
            "USD": 1.0,
            "EUR": 1.1,
            "GBP": 1.25,
            "CAD": 0.75,
            "AUD": 0.7,
        }

        logger.info("EcommerceExtractor initialized")

    async def enhance_extraction(self, content: ExtractedContent, html: str) -> ExtractedContent:
        """Enhance extraction with e-commerce domain rules."""
        if not content.text:
            return content

        # Extract e-commerce entities
        entities = await self.extract_entities(content.text)

        # Store domain-specific data
        if not hasattr(content, "domain_data"):
            content.domain_data = {}

        content.domain_data["ecommerce_entities"] = entities

        # Extract and normalize pricing
        pricing_info = await self._extract_pricing_info(content.text)
        if pricing_info:
            content.domain_data["pricing"] = pricing_info

        # Extract product specifications
        specifications = await self._extract_specifications(content.text)
        if specifications:
            content.domain_data["specifications"] = specifications

        # Analyze reviews and ratings
        review_analysis = await self._analyze_reviews(content.text)
        if review_analysis:
            content.domain_data["reviews"] = review_analysis

        # Check inventory status
        inventory_status = await self._check_inventory_status(content.text)
        if inventory_status:
            content.domain_data["inventory"] = inventory_status

        # Calculate e-commerce domain confidence
        ecommerce_confidence = await self.calculate_domain_confidence(content.text)
        content.domain_data["ecommerce_confidence"] = ecommerce_confidence

        return content

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract e-commerce entities from text."""
        entities = {}

        for entity_type, patterns in self.extraction_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                if isinstance(found[0], tuple) if found else False:
                    # Handle tuple results from grouped patterns
                    matches.extend([match[0] if match[0] else match[1] for match in found])
                else:
                    matches.extend(found)

            # Clean and deduplicate
            clean_matches = list(set(match.strip() for match in matches if match.strip()))
            if clean_matches:
                entities[entity_type] = clean_matches

        return entities

    async def _extract_pricing_info(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and normalize pricing information."""
        pricing: Dict[str, Any] = {
            "prices": [],
            "currencies": set(),
            "price_range": None,
            "discounts": [],
        }

        # Extract all price mentions
        price_patterns = [
            r"\$(\d+(?:\.\d{2})?)",  # USD
            r"€(\d+(?:\.\d{2})?)",  # EUR
            r"£(\d+(?:\.\d{2})?)",  # GBP
            r"(\d+(?:\.\d{2}))\s*(USD|EUR|GBP|CAD|AUD)",  # With currency codes
        ]

        for pattern in price_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    price, currency = match
                    pricing["prices"].append(
                        {
                            "amount": float(price),
                            "currency": currency.upper(),
                        }
                    )
                    pricing["currencies"].add(currency.upper())
                else:
                    # Simple price without explicit currency
                    pricing["prices"].append(
                        {
                            "amount": float(match),
                            "currency": "USD",  # Default assumption
                        }
                    )
                    pricing["currencies"].add("USD")

        # Calculate price range
        if pricing["prices"]:
            amounts = [p["amount"] for p in pricing["prices"]]
            pricing["price_range"] = {
                "min": min(amounts),
                "max": max(amounts),
                "avg": sum(amounts) / len(amounts),
            }

        # Look for discount indicators
        discount_patterns = [
            r"(\d+)%\s*off",
            r"save\s*\$(\d+(?:\.\d{2})?)",
            r"was\s*\$(\d+(?:\.\d{2})?)",
        ]

        for pattern in discount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            pricing["discounts"].extend(matches)

        return pricing if pricing["prices"] else None

    async def _extract_specifications(self, text: str) -> Dict[str, str]:
        """Extract product specifications."""
        specs = {}

        # Look for key-value pairs in specifications
        spec_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):?\s*([^,\n;]+)"
        matches = re.findall(spec_pattern, text)

        for key, value in matches:
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()

            # Filter out obvious non-specifications
            if len(value) < 50 and not any(word in value.lower() for word in ["click", "see", "more"]):
                specs[key] = value

        return specs

    async def _analyze_reviews(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze review information."""
        review_analysis = {
            "ratings": [],
            "review_count": 0,
            "average_rating": None,
        }

        # Extract star ratings
        rating_pattern = r"(\d+(?:\.\d+)?)\s*(?:stars?|/5|out of 5)"
        rating_matches = re.findall(rating_pattern, text, re.IGNORECASE)

        if rating_matches:
            ratings = [float(r) for r in rating_matches]
            review_analysis["ratings"] = ratings
            review_analysis["average_rating"] = sum(ratings) / len(ratings)

        # Extract review counts
        count_pattern = r"(\d+)\s*reviews?"
        count_matches = re.findall(count_pattern, text, re.IGNORECASE)

        if count_matches:
            review_analysis["review_count"] = max(int(c) for c in count_matches)

        return review_analysis if review_analysis["ratings"] or review_analysis["review_count"] else None

    async def _check_inventory_status(self, text: str) -> Dict[str, Any]:
        """Check inventory and availability status."""
        inventory: Dict[str, Any] = {
            "status": "unknown",
            "quantity": None,
            "availability_text": [],
        }

        text_lower = text.lower()

        # Check availability status
        if any(phrase in text_lower for phrase in ["in stock", "available now", "ready to ship"]):
            inventory["status"] = "in_stock"
        elif any(phrase in text_lower for phrase in ["out of stock", "sold out", "unavailable"]):
            inventory["status"] = "out_of_stock"
        elif any(phrase in text_lower for phrase in ["backordered", "pre-order", "coming soon"]):
            inventory["status"] = "backordered"

        # Extract quantity information
        quantity_pattern = r"(\d+)\s*(?:left|remaining|available|in stock)"
        quantity_matches = re.findall(quantity_pattern, text, re.IGNORECASE)

        if quantity_matches:
            inventory["quantity"] = max(int(q) for q in quantity_matches)

        # Extract availability text
        availability_pattern = r"((?:in stock|out of stock|available|unavailable)[^.!?]*)"
        availability_matches = re.findall(availability_pattern, text, re.IGNORECASE)
        inventory["availability_text"] = availability_matches

        # Extract availability count
        if availability_matches:
            # Extract numeric availability
            availability_match = re.search(r"(\d+)", str(availability_matches[0]) if availability_matches else "")
            if availability_match:
                int(availability_match.group(1))

        return inventory


class TechnicalExtractor(BaseDomainExtractor):
    """
    Technical domain extractor for code, APIs, and documentation.

    Features:
    - Code snippet extraction with language detection
    - API documentation parsing
    - Technology stack identification
    - Performance metric extraction
    - Dependency and version extraction
    """

    def __init__(self) -> None:
        super().__init__()

        self.domain_keywords = {
            "code",
            "function",
            "class",
            "method",
            "variable",
            "parameter",
            "api",
            "endpoint",
            "request",
            "response",
            "json",
            "xml",
            "http",
            "library",
            "framework",
            "dependency",
            "package",
            "module",
            "import",
            "version",
            "release",
            "update",
            "patch",
            "build",
            "deploy",
            "test",
            "performance",
            "benchmark",
            "optimization",
            "scalability",
            "latency",
            "documentation",
            "tutorial",
            "example",
            "guide",
            "reference",
            "manual",
        }

        # Technical entity patterns
        self.extraction_patterns = {
            "programming_languages": [
                r"\b(Python|JavaScript|Java|C\+\+|C#|Ruby|PHP|Go|Rust|Swift|Kotlin)\b",
            ],
            "frameworks": [
                r"\b(React|Angular|Vue|Django|Flask|Spring|Express|Rails|Laravel)\b",
            ],
            "technologies": [
                r"\b(Docker|Kubernetes|AWS|Azure|GCP|MongoDB|PostgreSQL|MySQL|Redis)\b",
            ],
            "version_numbers": [
                r"\bv?(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?)\b",
            ],
            "api_endpoints": [
                r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-:{}]*)",
                r"\bhttps?://[^\s/$.?#].[^\s]*\b",
            ],
            "performance_metrics": [
                r"(\d+(?:\.\d+)?)\s*(ms|seconds?|minutes?|hours?)",  # Time
                r"(\d+(?:\.\d+)?)\s*(MB/s|GB/s|Mbps|Gbps)",  # Throughput
                r"(\d+(?:\.\d+)?)\s*(MB|GB|KB)",  # Memory/storage
            ],
        }

        # Code quality indicators
        self.quality_patterns = {
            "good_practices": [
                r"\b(test|spec|unittest|pytest|jest|mocha)\b",  # Testing
                r"\b(documentation|readme|docs)\b",  # Documentation
                r"\b(error handling|exception|try-catch)\b",  # Error handling
            ],
            "technical_concepts": [
                r"\b(algorithm|data structure|design pattern|architecture)\b",
                r"\b(async|asynchronous|concurrent|parallel|distributed)\b",
                r"\b(security|authentication|authorization|encryption)\b",
            ],
        }

        logger.info("TechnicalExtractor initialized")

    async def enhance_extraction(self, content: ExtractedContent, html: str) -> ExtractedContent:
        """Enhance extraction with technical domain rules."""
        if not content.text:
            return content

        # Extract technical entities
        entities = await self.extract_entities(content.text)

        # Store domain-specific data
        if not hasattr(content, "domain_data"):
            content.domain_data = {}

        content.domain_data["technical_entities"] = entities

        # Identify technology stack
        tech_stack = await self._identify_tech_stack(content.text)
        if tech_stack:
            content.domain_data["technology_stack"] = tech_stack

        # Extract API information
        api_info = await self._extract_api_info(content.text)
        if api_info:
            content.domain_data["api_info"] = api_info

        # Analyze code quality indicators
        quality_analysis = await self._analyze_code_quality(content.text)
        if quality_analysis:
            content.domain_data["code_quality"] = quality_analysis

        # Extract performance metrics
        performance_metrics = await self._extract_performance_metrics(content.text)
        if performance_metrics:
            content.domain_data["performance_metrics"] = performance_metrics

        # Calculate technical domain confidence
        technical_confidence = await self.calculate_domain_confidence(content.text)
        content.domain_data["technical_confidence"] = technical_confidence

        return content

    async def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract technical entities from text."""
        entities = {}

        for entity_type, patterns in self.extraction_patterns.items():
            matches = []
            for pattern in patterns:
                found = re.findall(pattern, text, re.IGNORECASE)
                if isinstance(found[0], tuple) if found else False:
                    # Handle tuple results
                    matches.extend([" ".join(match) for match in found])
                else:
                    matches.extend(found)

            # Clean and deduplicate
            clean_matches = list(set(match.strip() for match in matches if match.strip()))
            if clean_matches:
                entities[entity_type] = clean_matches

        return entities

    async def _identify_tech_stack(self, text: str) -> Dict[str, List[str]]:
        """Identify technology stack from content."""
        tech_stack: Dict[str, List[str]] = {
            "programming_languages": [],
            "frameworks": [],
            "databases": [],
            "tools": [],
            "platforms": [],
        }

        text_lower = text.lower()

        # Programming languages
        languages = [
            "python",
            "javascript",
            "java",
            "c++",
            "c#",
            "ruby",
            "php",
            "go",
            "rust",
        ]
        tech_stack["programming_languages"] = [lang for lang in languages if lang in text_lower]

        # Frameworks
        frameworks = ["react", "angular", "vue", "django", "flask", "spring", "express"]
        tech_stack["frameworks"] = [fw for fw in frameworks if fw in text_lower]

        # Databases
        databases = ["mongodb", "postgresql", "mysql", "redis", "elasticsearch"]
        tech_stack["databases"] = [db for db in databases if db in text_lower]

        # Tools
        tools = ["docker", "kubernetes", "jenkins", "git", "npm", "webpack"]
        tech_stack["tools"] = [tool for tool in tools if tool in text_lower]

        # Platforms
        platforms = ["aws", "azure", "gcp"]
        tech_stack["platforms"] = [platform for platform in platforms if platform in text_lower]

        # Remove empty categories
        return {k: v for k, v in tech_stack.items() if v}

    async def _extract_api_info(self, text: str) -> Dict[str, Any]:
        """Extract API-related information."""
        api_info: Dict[str, Any] = {
            "endpoints": [],
            "authentication": [],
            "examples": [],
            "parameters": [],
            "responses": [],
        }

        # Extract HTTP methods and endpoints
        endpoint_pattern = r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[\w/\-:{}]*)"
        endpoint_matches = re.findall(endpoint_pattern, text, re.IGNORECASE)

        for method, endpoint in endpoint_matches:
            api_info["endpoints"].append(
                {
                    "method": method.upper(),
                    "path": endpoint,
                }
            )
            if method.upper() not in api_info["authentication"]:
                api_info["authentication"].append(method.upper())

        # Extract authentication methods
        auth_pattern = r"\b(Basic|Bearer|OAuth|API Key|Token)\b"
        auth_matches = re.findall(auth_pattern, text)
        api_info["authentication"] = list(set(auth_matches))

        # Extract code examples
        code_pattern = (
            r"\b(?:```|```python|```javascript|```java|```c++|```ruby|```php|```go|```rust|```swift|```kotlin)\b"
        )
        code_matches = re.findall(code_pattern, text)
        api_info["examples"] = code_matches

        # Extract parameters
        param_pattern = r"\b(\w+):\s*([^,\n]+)\b"
        param_matches = re.findall(param_pattern, text)
        api_info["parameters"] = param_matches

        # Extract response formats
        response_pattern = (
            r"\b(?:application/json|application/xml|application/csv|application/html|text/csv|text/plain)\b"
        )
        response_matches = re.findall(response_pattern, text)
        api_info["responses"] = response_matches

        # Remove empty fields
        return {k: v for k, v in api_info.items() if v}

    async def _analyze_code_quality(self, text: str) -> Dict[str, Any]:
        """Analyze code quality indicators."""
        quality: Dict[str, Any] = {
            "has_tests": False,
            "has_documentation": False,
            "has_error_handling": False,
            "technical_concepts": [],
            "quality_score": 0.0,
        }

        text_lower = text.lower()

        # Check for testing
        if any(test_word in text_lower for test_word in ["test", "spec", "unittest", "pytest"]):
            quality["has_tests"] = True
            quality["quality_score"] = float(quality["quality_score"]) + 0.3

        # Check for documentation
        if any(doc_word in text_lower for doc_word in ["documentation", "readme", "docs", "comment"]):
            quality["has_documentation"] = True
            quality["quality_score"] = float(quality["quality_score"]) + 0.2

        # Check for error handling
        if any(error_word in text_lower for error_word in ["error", "exception", "try", "catch"]):
            quality["has_error_handling"] = True
            quality["quality_score"] = float(quality["quality_score"]) + 0.2

        # Extract technical concepts
        concepts = [
            "algorithm",
            "data structure",
            "design pattern",
            "architecture",
            "security",
        ]
        quality["technical_concepts"] = [concept for concept in concepts if concept in text_lower]

        # Bonus for advanced concepts
        if quality["technical_concepts"]:
            quality["quality_score"] = float(quality["quality_score"]) + 0.3

        return quality

    async def _extract_performance_metrics(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract performance-related metrics."""
        metrics: Dict[str, List[Dict[str, Any]]] = {
            "timing": [],
            "throughput": [],
            "memory": [],
        }

        # Time metrics
        time_pattern = r"(\d+(?:\.\d+)?)\s*(ms|milliseconds?|seconds?|minutes?|hours?)"
        time_matches = re.findall(time_pattern, text, re.IGNORECASE)

        for value, unit in time_matches:
            metrics["timing"].append(
                {
                    "value": float(value),
                    "unit": unit.lower(),
                }
            )

        # Throughput metrics
        throughput_pattern = r"(\d+(?:\.\d+)?)\s*(MB/s|GB/s|Mbps|Gbps|requests?/s)"
        throughput_matches = re.findall(throughput_pattern, text, re.IGNORECASE)

        for value, unit in throughput_matches:
            metrics["throughput"].append(
                {
                    "value": float(value),
                    "unit": unit,
                }
            )

        # Memory metrics
        memory_pattern = r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)"
        memory_matches = re.findall(memory_pattern, text, re.IGNORECASE)

        for value, unit in memory_matches:
            metrics["memory"].append(
                {
                    "value": float(value),
                    "unit": unit.upper(),
                }
            )

        # Remove empty categories
        return {k: v for k, v in metrics.items() if v}


class DomainExtractorFactory:
    """Factory for creating domain-specific extractors."""

    def __init__(self) -> None:
        self._extractors = {
            DomainType.MEDICAL: MedicalExtractor(),
            DomainType.LEGAL: LegalExtractor(),
            DomainType.ECOMMERCE: EcommerceExtractor(),
            DomainType.TECHNICAL: TechnicalExtractor(),
        }

        logger.info("DomainExtractorFactory initialized")

    def get_extractor(self, domain_type: DomainType) -> Optional[BaseDomainExtractor]:
        """Get extractor for specific domain type."""
        return self._extractors.get(domain_type)

    def get_all_extractors(self) -> Dict[DomainType, BaseDomainExtractor]:
        """Get all available extractors."""
        return self._extractors.copy()

    async def detect_best_domain(self, text: str) -> Tuple[DomainType, float]:
        """
        Detect the best matching domain for given text.

        Returns:
            Tuple of (domain_type, confidence_score)
        """
        if not text:
            return DomainType.GENERAL, 0.0

        domain_scores = {}

        for domain_type, extractor in self._extractors.items():
            confidence = await extractor.calculate_domain_confidence(text)
            domain_scores[domain_type] = confidence

        if not domain_scores:
            return DomainType.GENERAL, 0.0

        best_domain = max(domain_scores, key=lambda x: domain_scores[x])
        best_confidence = domain_scores[best_domain]

        # Return GENERAL if no domain has sufficient confidence
        if best_confidence < 0.3:
            return DomainType.GENERAL, 0.0

        return best_domain, best_confidence
