# src/data/mtg_data_loader.py
import json
import os
from typing import Dict, List, Optional, Any, TypedDict, Union
import logging
from pathlib import Path

# Configure module-level logger
logger = logging.getLogger(__name__)


class RulingType(TypedDict):
    """Type definition for card rulings."""

    date: str
    text: str


class CardType(TypedDict, total=False):
    """Type definition for card data with optional fields."""

    name: str
    mana_cost: str
    cmc: float
    type_line: str
    oracle_text: str
    colors: List[str]
    color_identity: List[str]
    power: str
    toughness: str
    loyalty: str
    keywords: List[str]
    legalities: Dict[str, str]
    rulings: List[RulingType]
    types: List[str]
    subtypes: List[str]
    supertypes: List[str]


class DocumentType(TypedDict):
    """Type definition for retrieval documents."""

    id: str
    type: str
    text: str
    metadata: dict


# Field name mappings for standardization
FIELD_MAPPINGS = {
    # Mana and cost information
    "mana_cost": ["mana_cost", "manaCost"],
    "cmc": ["cmc", "convertedManaCost", "manaValue"],
    # Type information
    "type_line": ["type_line", "type"],
    "types": ["types"],
    "subtypes": ["subtypes"],
    "supertypes": ["supertypes"],
    # Text and rules information
    "oracle_text": ["oracle_text", "text"],
    # Color information
    "colors": ["colors"],
    "color_identity": ["color_identity", "colorIdentity"],
    # Stats information
    "power": ["power"],
    "toughness": ["toughness"],
    "loyalty": ["loyalty"],
    # Additional information
    "keywords": ["keywords"],
    "legalities": ["legalities"],
    "rulings": ["rulings"],
}


class MTGDataLoader:
    """
    Load and manage Magic: The Gathering card and rules data from JSON files.

    This class handles loading card data and comprehensive rules from JSON files,
    normalizing field names for consistency, and providing methods to search and
    retrieve the loaded data.
    """

    def __init__(self, data_dir: str = "data") -> None:
        """
        Initialize the MTG data loader.

        Args:
            data_dir: Directory path to store and load data files
        """
        self.data_dir = data_dir
        self.cards_file = os.path.join(data_dir, "cards.json")
        self.rules_file = os.path.join(data_dir, "rules.json")
        self.glossary_file = os.path.join(data_dir, "glossary.json")

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Initialize data containers
        self.cards: Dict[str, CardType] = {}
        self.rules: Dict[str, Union[str, List, Dict]] = (
            {}
        )  # Updated to handle different formats
        self.rules_hierarchy: List[Dict[str, Any]] = []

    def load_data(self) -> List[DocumentType]:
        """
        Load card data and rules from local JSON files.

        Returns:
            List of documents prepared for retrieval system
        """
        # Load cards
        self.load_cards()

        # Load rules
        self.load_rules()

        # Create documents for retrieval
        documents = self._create_documents_for_retrieval()
        return documents

    def _should_update_card(
        self, existing_card: CardType, new_card_data: Dict[str, Any]
    ) -> bool:
        """
        Determine if existing card should be updated with new data.

        Args:
            existing_card: Current card data
            new_card_data: New card data to consider

        Returns:
            True if should update, False otherwise
        """
        # Update if new card has text and existing doesn't
        if not existing_card.get("oracle_text") and not existing_card.get("text"):
            if new_card_data.get("oracle_text") or new_card_data.get("text"):
                return True

        # Update if new card has rulings and existing doesn't
        if not existing_card.get("rulings") and "rulings" in new_card_data:
            return True

        # Update if new card has more complete type information
        if not existing_card.get("type_line") and not existing_card.get("type"):
            if new_card_data.get("type_line") or new_card_data.get("type"):
                return True

        # Update if new card has mana cost and existing doesn't
        if not existing_card.get("mana_cost") and new_card_data.get("mana_cost"):
            return True

        return False

    def _load_cards_from_sets_data(self, sets_data: Dict[str, Any]) -> int:
        """
        Load cards from a data structure organised by sets.

        Args:
            sets_data: Dictionary of set code -> set data

        Returns:
            Number of cards loaded
        """
        self.cards = {}
        total_cards = 0
        card_count = 0

        for set_code, set_data in sets_data.items():
            if not (isinstance(set_data, dict) and "cards" in set_data):
                continue

            cards_in_set = set_data["cards"]
            total_cards += len(cards_in_set)
            card_count += self._process_cards_in_set(cards_in_set)

        logger.info(
            f"Found {total_cards} total cards across all sets, loaded {card_count} unique cards"
        )
        return card_count

    def _process_cards_in_set(self, cards_in_set: list) -> int:
        """
        Process the list of cards for a given set.

        Args:
            cards_in_set: List of card data dictionaries.

        Returns:
            Number of unique cards loaded from this set.
        """
        count = 0
        for card_data in cards_in_set:
            # Skip non-English cards
            if card_data.get("language", "English") != "English":
                continue

            card_name = card_data.get("name")
            if not card_name:
                continue

            # Process card: add new card or update if needed
            if card_name not in self.cards:
                self.cards[card_name] = self._process_card_data(card_data)
                count += 1
            elif self._should_update_card(self.cards[card_name], card_data):
                self.cards[card_name] = self._process_card_data(card_data)
        return count

    def load_cards(self) -> int:
        """
        Load card data from a local JSON file.

        Returns:
            Number of cards loaded
        """
        if not os.path.exists(self.cards_file):
            logger.warning(f"Cards file not found: {self.cards_file}")
            return 0

        logger.info(f"Loading cards from: {self.cards_file}")
        try:
            with open(self.cards_file, "r", encoding="utf-8") as f:
                cards_data = json.load(f)

            # Reset cards dictionary
            self.cards = {}

            # Validate the basic structure of the JSON
            if not isinstance(cards_data, dict) or "data" not in cards_data:
                logger.error(
                    "Invalid cards file format: missing 'data' field or top-level dict"
                )
                return 0

            sets_data = cards_data["data"]
            if not isinstance(sets_data, dict):
                logger.error(
                    f"Expected dictionary in 'data' section, got {type(sets_data)}"
                )
                return 0

            total_cards_processed = 0
            unique_cards_found = 0

            # Iterate through each set and process cards using a helper method
            for set_code, set_data in sets_data.items():
                if set_code == "meta":
                    continue

                unique, total = self._process_set_cards(set_code, set_data)
                unique_cards_found += unique
                total_cards_processed += total

            logger.info(
                f"Processed {total_cards_processed} cards from all sets, loaded {unique_cards_found} unique cards"
            )
            return unique_cards_found

        except json.JSONDecodeError:
            logger.error(f"Failed to parse cards file: {self.cards_file}")
            return 0
        except Exception as e:
            logger.error(f"Error loading cards file: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return 0

    def _process_set_cards(self, set_code: str, set_data: Any) -> tuple:
        """
        Process card data for a specific set.

        Args:
            set_code: Identifier for the set.
            set_data: The set data, expected to contain a "cards" field.

        Returns:
            A tuple (unique_cards_found, total_cards_processed) for the set.
        """
        if not isinstance(set_data, dict) or "cards" not in set_data:
            logger.debug(f"Set {set_code} has no cards field")
            return 0, 0

        cards_in_set = set_data["cards"]
        if not isinstance(cards_in_set, list):
            logger.warning(
                f"Expected list in 'cards' field of set {set_code}, got {type(cards_in_set)}"
            )
            return 0, 0

        total_cards = len(cards_in_set)
        unique_cards = 0

        for card_data in cards_in_set:
            if not isinstance(card_data, dict):
                continue
            # Skip non-English cards
            if card_data.get("language", "English") != "English":
                continue

            card_name = card_data.get("name")
            if not card_name:
                continue

            # Process card data and update the dictionary as needed
            if card_name not in self.cards:
                self.cards[card_name] = self._process_card_data(card_data)
                unique_cards += 1
            elif self._should_update_card(self.cards[card_name], card_data):
                self.cards[card_name] = self._process_card_data(card_data)

        return unique_cards, total_cards

    def _process_card_data(self, card_data: Dict[str, Any]) -> CardType:
        """
        Convert raw card data to standardized format.

        Args:
            card_data: Raw card data from JSON

        Returns:
            Processed card data
        """
        # Initialize with basic properties
        card: CardType = {"name": card_data.get("name", "")}

        # Process card data by category
        self._add_mana_info(card, card_data)
        self._add_type_info(card, card_data)
        self._add_text_info(card, card_data)
        self._add_color_info(card, card_data)
        self._add_stat_info(card, card_data)
        self._add_additional_info(card, card_data)

        return card

    def _add_mana_info(self, card: CardType, card_data: Dict[str, Any]) -> None:
        """Add mana cost and cmc information to card data."""
        # Use field mappings for standardization
        for target_field, source_fields in FIELD_MAPPINGS.items():
            if target_field in ["mana_cost", "cmc"]:
                for field in source_fields:
                    if field in card_data:
                        card[target_field] = card_data[field]
                        break

    def _add_type_info(self, card: CardType, card_data: Dict[str, Any]) -> None:
        """Add type information to card data."""
        # Use field mappings for standardization
        for target_field, source_fields in FIELD_MAPPINGS.items():
            if target_field in ["type_line", "types", "subtypes", "supertypes"]:
                for field in source_fields:
                    if field in card_data:
                        card[target_field] = card_data[field]
                        break

    def _add_text_info(self, card: CardType, card_data: Dict[str, Any]) -> None:
        """Add oracle text to card data."""
        # Use field mappings for standardization
        for field in FIELD_MAPPINGS["oracle_text"]:
            if field in card_data:
                card["oracle_text"] = card_data[field]
                break

    def _add_color_info(self, card: CardType, card_data: Dict[str, Any]) -> None:
        """Add color information to card data."""
        # Use field mappings for standardization
        for target_field, source_fields in FIELD_MAPPINGS.items():
            if target_field in ["colors", "color_identity"]:
                for field in source_fields:
                    if field in card_data:
                        card[target_field] = card_data[field]
                        break

    def _add_stat_info(self, card: CardType, card_data: Dict[str, Any]) -> None:
        """Add power, toughness and loyalty information to card data."""
        # Use field mappings for standardization
        for target_field, source_fields in FIELD_MAPPINGS.items():
            if target_field in ["power", "toughness", "loyalty"]:
                for field in source_fields:
                    if field in card_data:
                        card[target_field] = card_data[field]
                        break

    def _add_additional_info(self, card: CardType, card_data: Dict[str, Any]) -> None:
        """Add additional card information like keywords and rulings."""
        # Use field mappings for standardization
        for target_field, source_fields in FIELD_MAPPINGS.items():
            if target_field in ["keywords", "legalities", "rulings"]:
                for field in source_fields:
                    if field in card_data:
                        card[target_field] = card_data[field]
                        break

    def load_rules(self) -> int:
        """
        Load comprehensive rules from local JSON file.

        Returns:
            Number of rule entries loaded
        """
        if not os.path.exists(self.rules_file):
            logger.warning(f"Rules file not found: {self.rules_file}")
            return 0

        logger.info(f"Loading rules from: {self.rules_file}")
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                rules_data = json.load(f)

                # Reset rules dictionary
                self.rules = {}

                # Check if the data is in the expected format
                if not isinstance(rules_data, dict) or "categories" not in rules_data:
                    logger.error(
                        "Rules file does not contain expected 'categories' field"
                    )
                    return 0

                # Process hierarchical rules
                self._process_hierarchical_rules(rules_data)

                # Log results
                logger.info(f"Loaded {len(self.rules)} rule entries")
                return len(self.rules)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse rules file: {self.rules_file}")
            return 0
        except Exception as e:
            logger.error(f"Error loading rules file: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return 0

    def _process_hierarchical_rules(self, rules_data: Dict[str, Any]) -> None:
        """
        Process hierarchical rules data into a flat dictionary.

        Args:
            rules_data: Hierarchical rules data with categories, sections, rules, and subrules
        """
        # Get the categories list
        categories = rules_data.get("categories", [])

        # Process each category
        for category in categories:
            # Process sections in each category
            for section in category.get("sections", []):
                # Process rules in each section
                for rule in section.get("rules", []):
                    # Add the main rule
                    rule_number = rule.get("rule_number", "")
                    rule_text = rule.get("text", "")

                    if rule_number:
                        self.rules[rule_number] = rule_text

                    # Process subrules recursively
                    if "subrules" in rule:
                        self._process_subrules(rule.get("subrules", []))

    def _process_subrules(self, subrules: List[Dict[str, Any]]) -> None:
        """
        Recursively process subrules, adding them to the rules dictionary.

        Args:
            subrules: List of subrule objects with rule_number, text, and potentially more subrules
        """
        for subrule in subrules:
            # Add this subrule
            rule_number = subrule.get("rule_number", "")
            rule_text = subrule.get("text", "")

            if rule_number:
                self.rules[rule_number] = rule_text

            # Process any nested subrules
            if "subrules" in subrule:
                self._process_subrules(subrule.get("subrules", []))

    def _create_documents_for_retrieval(self) -> List[DocumentType]:
        """
        Create documents for retrieval system from loaded data.

        Returns:
            List of documents formatted for retrieval
        """
        documents: List[DocumentType] = []

        # Add card documents
        self._add_card_documents(documents)

        # Add rule documents
        self._add_rule_documents(documents)

        logger.info(f"Created {len(documents)} documents for retrieval")
        return documents

    def _add_card_documents(self, documents: List[DocumentType]) -> None:
        for card_name, card in self.cards.items():
            doc_text = self._format_card_document_text(card_name, card)
            documents.append(
                {
                    "id": f"card_{card_name.lower().replace(' ', '_')}",
                    "type": "card",
                    "text": doc_text,
                    "metadata": {  # Add structured metadata
                        "name": card_name,
                        "types": card.get("types", []),
                        "mana_cost": card.get("mana_cost", ""),
                        "keywords": card.get("keywords", []),
                    },
                }
            )

    def _add_rule_documents(self, documents: List[DocumentType]) -> None:
        for rule_id, rule_text in self.rules.items():
            if isinstance(rule_text, str):
                documents.append(
                    {
                        "id": f"rule_{rule_id}",
                        "type": "rule",
                        "text": f"Rule {rule_id}: {rule_text}",
                        "metadata": {  # Add rule metadata
                            "rule_id": rule_id,
                            "category": rule_id.split(".")[0],
                        },
                    }
                )

    def _format_card_document_text(self, card_name: str, card: CardType) -> str:
        """
        Format card data as text for a document by assembling parts from helper functions.

        Args:
            card_name: Name of the card
            card: Card data dictionary

        Returns:
            Formatted document text as a string.
        """
        parts = [f"Card Name: {card_name}"]
        parts.extend(self._format_basic_info(card))
        parts.extend(self._format_types(card))

        rulings_text = self._format_rulings(card)
        if rulings_text:
            parts.append(rulings_text)

        return "\n".join(parts)

    def _format_basic_info(self, card: CardType) -> list:
        """Return basic card information as a list of strings."""
        info = []

        mana_cost = card.get("mana_cost")
        if mana_cost:
            info.append(f"Mana Cost: {mana_cost}")

        card_type = card.get("type_line") or card.get("type")
        if card_type:
            info.append(f"Type: {card_type}")

        text = card.get("oracle_text") or card.get("text")
        if text:
            info.append(f"Text: {text}")

        power = card.get("power")
        toughness = card.get("toughness")
        if power is not None and toughness is not None:
            info.append(f"Power/Toughness: {power}/{toughness}\n")

        loyalty = card.get("loyalty")
        if loyalty:
            info.append(f"Loyalty: {loyalty}")

        keywords = card.get("keywords")
        if keywords:
            info.append(f"Keywords: {', '.join(keywords)}")

        return info

    def _format_types(self, card: CardType) -> list:
        """Return concatenated type information as a list of strings."""
        type_parts = []
        for key in ["supertypes", "types", "subtypes"]:
            if card.get(key):
                type_parts.extend(card[key])
        if type_parts:
            return [f"All Types: {', '.join(type_parts)}"]
        return []

    def _format_rulings(self, card: CardType) -> str:
        """Return formatted rulings if available, else an empty string."""
        rulings = card.get("rulings")
        if not rulings:
            return ""
        formatted = ["Rulings:"]
        for ruling in rulings:
            formatted.append(f"- [{ruling['date']}] {ruling['text']}")
        return "\n".join(formatted)

    def get_card(self, name: str) -> Optional[CardType]:
        """
        Get card data by name.

        Args:
            name: Card name

        Returns:
            Card data dictionary or None if not found
        """
        # Try exact match first
        if name in self.cards:
            return self.cards[name]

        # Try case-insensitive match
        name_lower = name.lower()
        for card_name, card_data in self.cards.items():
            if card_name.lower() == name_lower:
                return card_data

        return None

    def get_rule(self, rule_id: str) -> Optional[str]:
        """
        Get rule text by ID.

        Args:
            rule_id: Rule identifier (e.g., "101.1")

        Returns:
            Rule text or None if not found
        """
        rule = self.rules.get(rule_id)
        # Only return if the rule is a string (not a list or dict)
        return rule if isinstance(rule, str) else None

    def search_cards(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for cards by name, text, or type.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching cards with scores
        """
        query_lower = query.lower()
        scored_cards = []

        for card_name, card in self.cards.items():
            score = self._calculate_card_match_score(card_name, card, query_lower)
            if score > 0:
                # Create a copy with name included and add score
                card_copy = dict(card)
                card_copy["name"] = card_name
                card_copy["score"] = score
                scored_cards.append(card_copy)

        # Sort by score and return top matches
        scored_cards.sort(key=lambda x: x["score"], reverse=True)
        return scored_cards[:limit]

    def _calculate_card_match_score(
        self, card_name: str, card: CardType, query_lower: str
    ) -> int:
        """
        Calculate match score for a card based on query.

        Args:
            card_name: Name of the card
            card: Card data
            query_lower: Lowercase search query

        Returns:
            Score representing how well the card matches the query
        """
        score = 0

        # Check name match
        score += self._get_name_match_score(card_name, query_lower)

        # Check text match
        score += self._get_text_match_score(card, query_lower)

        # Check type match
        score += self._get_type_match_score(card, query_lower)

        return score

    def _get_name_match_score(self, card_name: str, query_lower: str) -> int:
        """Calculate score based on card name match."""
        if query_lower == card_name.lower():
            return 110  # Exact match gets highest priority
        elif query_lower in card_name.lower():
            return 10
        return 0

    def _get_text_match_score(self, card: CardType, query_lower: str) -> int:
        """Calculate score based on card text match."""
        # Get card text from appropriate field
        card_text = ""
        if "oracle_text" in card and card["oracle_text"]:
            card_text = card["oracle_text"]
        elif "text" in card and card["text"]:
            card_text = card["text"]

        if card_text and query_lower in card_text.lower():
            return 5
        return 0

    def _get_type_match_score(self, card: CardType, query_lower: str) -> int:
        """Calculate score based on card type match."""
        # Get type information from appropriate field
        card_type = ""
        if "type_line" in card and card["type_line"]:
            card_type = card["type_line"]
        elif "type" in card and card["type"]:
            card_type = card["type"]

        if card_type and query_lower in card_type.lower():
            return 3
        return 0

    def search_rules(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for rules by text or rule number.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching rules with score and text
        """
        results = []
        query_lower = query.lower()

        for rule_id, rule_text in self.rules.items():
            # Skip if rule_text is not a string
            if not isinstance(rule_text, str):
                continue

            score = 0

            # Check rule ID match
            if query in rule_id:
                score += 20
                # Exact match gets highest priority
                if rule_id == query:
                    score += 100

            # Check rule text match
            if query_lower in rule_text.lower():
                score += 10

            if score > 0:
                results.append({"rule_id": rule_id, "text": rule_text, "score": score})

        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
