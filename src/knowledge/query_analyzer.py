# src/knowledge/query_analyzer.py
import re
import logging
from typing import Dict, List, Set, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Analyzes MTG queries to determine optimal retrieval strategies.

    This analyzer examines query characteristics to decide whether to use
    vector-based search, knowledge graph traversal, or a hybrid approach,
    and identifies specific entities mentioned in the query.
    """

    def __init__(self, data_loader=None):
        """
        Initialize the query analyzer.

        Args:
            data_loader: Optional MTG data loader for entity recognition
        """
        self.data_loader = data_loader

        # Define common query patterns for classification
        self.patterns = {
            # Complex interaction queries - put this first to ensure it matches before mechanic_lookup
            "complex_interaction": [
                r"\binteract(?:ion|s)?\b",
                r"\bstack\b",
                r"\btrigger(?:ed|s)?\b",
                r"\bresolve\b",
                r"\bwhen\s+(.+?)\s+and\s+(.+?)\b",
                r"\bif\s+(.+?)\s+(?:then|and)\s+(.+?)\b",
                r"(.+?)\s+(?:with|and)\s+(.+?)\s+interact",
                r"(.+?)\s+respond(?:s|ed)?\s+with\s+(.+?)",
            ],
            # Card lookup queries
            "card_lookup": [
                r"\b(?:what|find|show|get|tell me about)\s+(?:the\s+)?card\s+(.+?)(?:\?|$)",
                r"\bcard\s+(?:named|called)\s+(?:\")?([^\"]+)(?:\")?\b",
                r"\bwhat\s+does\s+(?:the\s+)?card\s+(.+?)\s+do\b",
            ],
            # Rules lookup queries
            "rule_lookup": [
                r"\brule\s+(\d+\.\d+[a-z]?)\b",
                r"\bsection\s+(\d+\.\d+[a-z]?)\b",
                r"\b(\d{3}\.\d+[a-z]?)\b",  # Direct rule number
            ],
            # Mechanics queries
            "mechanic_lookup": [
                r"\bhow\s+does\s+(.+?)\s+(?:work|function)\b",
                r"\bexplain\s+(.+?)\s+(?:mechanic|keyword|ability)\b",
                r"\b(?:what|how)\s+(?:does|is)\s+(.+?)\b",
            ],
            # Strategic/gameplay queries
            "strategic": [
                r"\bstrategy\b",
                r"\bplay\b",
                r"\bdecision\b",
                r"\battack\b",
                r"\bblock\b",
                r"\bmulligan\b",
                r"\bbest\s+(?:way|approach|option)\b",
            ],
        }

        # Common MTG entity types to search for
        self.entity_types = {
            "card": [
                r"\"([^\"]+?)\"",  # Text in quotes
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b",  # Proper case words
                r"\s([^,\.!?]+?)\scard",  # Something followed by "card"
            ],
            "mechanic": [
                r"\b(Flying|First strike|Double strike|Deathtouch|Vigilance|Trample|Haste|Lifelink)\b",
                r"\b(Flash|Reach|Defender|Hexproof|Indestructible|Menace|Prowess|Ward)\b",
                r"\b(Equip|Cascade|Convoke|Delve|Emerge|Fortify|Madness|Morph|Miracle)\b",
                r"\b(Buyback|Dredge|Fading|Flashback|Kicker|Replicate|Retrace|Storm)\b",
            ],
            "keyword": [
                r"\b(Landfall|Constellation|Heroic|Ferocious|Spectacle|Undergrowth|Ascend)\b",
                r"\b(Adventure|Aftermath|Companion|Devotion|Escape|Foretell|Strive|Threshold)\b",
            ],
            "phase": [
                r"\b(Beginning|Untap|Upkeep|Draw|Main|Combat|Declare attackers|Declare blockers|Damage|End|Cleanup)\b",
            ],
        }

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine type, entities, and optimal retrieval strategy.

        Args:
            query: The user's query text

        Returns:
            Dictionary with analysis results including:
            - query_type: The classified query type
            - entities: List of entities mentioned in the query
            - requires_structured_knowledge: Whether graph traversal is recommended
            - relationship_types: Specific relationships to use in graph traversal
            - prioritize_graph: Whether to prioritize graph over vector retrieval
        """
        # Initialize default result
        result = {
            "query_type": "general",
            "entities": self._extract_entities(query),
            "requires_structured_knowledge": False,
            "relationship_types": [],
            "prioritize_graph": False,
        }

        # Determine query type based on patterns
        for query_type, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, query, re.IGNORECASE):
                    result["query_type"] = query_type
                    break
            if result["query_type"] != "general":
                break

        # Analyze if the query would benefit from graph-based retrieval
        result = self._determine_retrieval_strategy(query, result)

        # Extract specific relationships that may be relevant
        result["relationship_types"] = self._extract_relationships(
            query, result["query_type"]
        )

        # Log analysis results
        logger.debug(f"Query analysis: {result}")

        return result

    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """
        Extract mentioned entities from a query.

        Args:
            query: Query text

        Returns:
            List of entity dictionaries with type and name
        """
        entities = []

        # Extract entities using patterns
        for entity_type, patterns in self.entity_types.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(1) if match.lastindex else match.group(0)
                    entity_name = entity_name.strip()

                    # Skip if too short or already found
                    if len(entity_name) < 3 or any(
                        e["name"].lower() == entity_name.lower() for e in entities
                    ):
                        continue

                    entities.append({"type": entity_type, "name": entity_name})

        # Use data loader for better card recognition if available
        if self.data_loader and hasattr(self.data_loader, "cards"):
            self._enhance_card_recognition(query, entities)

        return entities

    def _enhance_card_recognition(
        self, query: str, entities: List[Dict[str, str]]
    ) -> None:
        """
        Enhance card entity recognition using the data loader.

        Args:
            query: Query text
            entities: List of entities to enhance (modified in-place)
        """
        # Get all card names from data loader
        query_lower = query.lower()
        card_names = getattr(self.data_loader, "cards", {}).keys()

        for card_name in sorted(card_names, key=len, reverse=True):
            # Skip very short names to avoid false positives
            if len(card_name) < 4:
                continue

            if card_name.lower() in query_lower:
                # Check if we already have this card
                if not any(
                    e["type"] == "card" and e["name"].lower() == card_name.lower()
                    for e in entities
                ):
                    entities.append({"type": "card", "name": card_name})

    def _determine_retrieval_strategy(
        self, query: str, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine the optimal retrieval strategy based on query characteristics.

        Args:
            query: Query text
            result: Current analysis result

        Returns:
            Updated analysis result
        """
        # Default to using vector retrieval (semantic search)
        result["requires_structured_knowledge"] = False
        result["prioritize_graph"] = False

        # Apply strategy based on query type
        self._apply_query_type_strategy(result)

        # Apply strategy based on relationship indicators
        self._apply_relationship_strategy(query, result)

        return result

    def _apply_query_type_strategy(self, result: Dict[str, Any]) -> None:
        """Apply retrieval strategy based on query type."""
        query_type = result["query_type"]
        entities = result["entities"]

        # Types that benefit from graph traversal with high priority
        if query_type in ["rule_lookup", "complex_interaction"]:
            result["requires_structured_knowledge"] = True
            result["prioritize_graph"] = True

        # Types that benefit from structured knowledge but work with either approach
        elif query_type == "card_lookup" and len(entities) == 1:
            result["requires_structured_knowledge"] = True
            # prioritize_graph remains False

        elif query_type == "mechanic_lookup":
            result["requires_structured_knowledge"] = True
            # prioritize_graph remains False

    def _apply_relationship_strategy(self, query: str, result: Dict[str, Any]) -> None:
        """Apply retrieval strategy based on relationship indicators in the query."""
        entities = result["entities"]

        # Calculate relationship score
        relationship_score = self._calculate_relationship_score(query)

        # If multiple entities and strong relationship indicators, prioritize graph
        if len(entities) >= 2 and relationship_score > 1.0:
            result["requires_structured_knowledge"] = True
            result["prioritize_graph"] = True

    def _calculate_relationship_score(self, query: str) -> float:
        """Calculate a score based on relationship indicators in the query."""
        relationship_indicators = {
            "between": 0.7,  # "interaction between X and Y"
            "relationship": 0.8,  # "relationship between X and Y"
            "affect": 0.7,  # "how does X affect Y"
            "combo": 0.9,  # "combo with X and Y"
            "interact": 0.8,  # "how do X and Y interact"
            "together": 0.7,  # "using X and Y together"
            "when": 0.6,  # "when X happens, Y does Z"
        }

        query_lower = query.lower()
        return sum(
            weight
            for indicator, weight in relationship_indicators.items()
            if indicator in query_lower
        )

    def _extract_relationships(self, query: str, query_type: str) -> List[str]:
        """
        Extract specific relationship types that may be needed for this query.

        Args:
            query: Query text
            query_type: Type of query

        Returns:
            List of relationship types that may be relevant
        """
        relationships = []

        # Map query types to likely relevant relationships
        type_to_relationships = {
            "card_lookup": ["card_uses_mechanic", "keyword_appears_on_card"],
            "rule_lookup": ["rule_references_rule", "mechanic_governed_by_rule"],
            "mechanic_lookup": ["card_uses_mechanic", "mechanic_governed_by_rule"],
            "complex_interaction": ["card_referenced_in_rule", "rule_references_rule"],
        }

        # Add relationships based on query type
        if query_type in type_to_relationships:
            relationships.extend(type_to_relationships[query_type])

        # Define shared patterns to avoid duplication
        rule_about_pattern = [r"\brule(?:s)?\s+(?:about|governing|for)\s+(.+?)\b"]
        card_reference_patterns = [
            r"\bcard(?:s)?\s+(?:affected by|mentioned in)\s+(.+?)\b",
            r"\bwhat\s+card(?:s)?\s+(?:are|is|get|gets)\s+(?:mentioned|referenced|cited)\s+in\s+(.+?)\b",
        ]

        # Detect specific relationship mentions using shared patterns where appropriate
        relationship_patterns = {
            "card_uses_mechanic": [
                r"\bcard(?:s)?\s+with\s+(.+?)\b",
                r"\b(.+?)\s+card(?:s)?\b",
            ],
            "rule_references_rule": rule_about_pattern,
            "mechanic_governed_by_rule": rule_about_pattern,
            "card_referenced_in_rule": card_reference_patterns,
        }

        for rel_type, patterns in relationship_patterns.items():
            for pattern in patterns:
                if (
                    re.search(pattern, query, re.IGNORECASE)
                    and rel_type not in relationships
                ):
                    relationships.append(rel_type)

        return relationships

    def get_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze the complexity of a query to help with retrieval planning.

        Args:
            query: Query text

        Returns:
            Dictionary with complexity metrics
        """
        # Analyze core complexity indicators
        words = query.split()

        # Count entities
        entities = self._extract_entities(query)

        # Count rule references
        rule_pattern = r"\b(\d+\.\d+[a-z]?)\b"
        rule_references = len(re.findall(rule_pattern, query))

        # Count relationship indicators
        relationship_words = [
            "between",
            "affect",
            "interact",
            "when",
            "if",
            "then",
            "because",
        ]
        relationship_count = sum(
            1 for word in words if word.lower() in relationship_words
        )

        # Add special case for "how does X work with Y" queries - they are inherently moderate
        special_patterns = {
            "moderate": [
                r"how\s+does\s+(.+?)\s+work\s+with\s+(.+?)",
                r"(.+?)\s+interact\s+with\s+(.+?)",
                r"when\s+(.+?)\s+and\s+(.+?)",
            ],
        }

        # Check special patterns first
        for category, patterns in special_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    # This is a special case that we always want to classify as specified
                    return {
                        "score": (
                            10.0 if category == "moderate" else 16.0
                        ),  # Just above the thresholds
                        "category": category,
                        "word_count": len(words),
                        "entity_count": len(entities),
                        "rule_references": rule_references,
                        "relationship_indicators": relationship_count,
                    }

        # Calculate complexity score for regular cases
        complexity_score = (
            len(words) * 0.1  # Base on length
            + len(entities) * 2.0  # Entities increase complexity
            + rule_references * 3.0  # Rule references significantly increase complexity
            + relationship_count * 2.5  # Increased weight for relationship indicators
        )

        complexity_category = "simple"
        if complexity_score > 15:
            complexity_category = "complex"
        elif complexity_score > 7:  # Slightly lowered threshold
            complexity_category = "moderate"

        return {
            "score": complexity_score,
            "category": complexity_category,
            "word_count": len(words),
            "entity_count": len(entities),
            "rule_references": rule_references,
            "relationship_indicators": relationship_count,
        }
