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


class MTGDataLoader:
    """
    Load and manage Magic: The Gathering card and rules data from JSON files.
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
        Load cards from a data structure organized by sets.

        Args:
            sets_data: Dictionary of set code -> set data

        Returns:
            Number of cards loaded
        """
        self.cards = {}
        total_cards = 0
        card_count = 0

        for set_code, set_data in sets_data.items():
            if isinstance(set_data, dict) and "cards" in set_data:
                cards_in_set = set_data["cards"]
                total_cards += len(cards_in_set)

                for card_data in cards_in_set:
                    # Skip non-English cards
                    if card_data.get("language", "English") != "English":
                        continue

                    # Get card name
                    card_name = card_data.get("name")
                    if not card_name:
                        continue

                    # Process card
                    if card_name not in self.cards:
                        self.cards[card_name] = self._process_card_data(card_data)
                        card_count += 1
                    elif self._should_update_card(self.cards[card_name], card_data):
                        # Replace with better version
                        self.cards[card_name] = self._process_card_data(card_data)

        logger.info(
            f"Found {total_cards} total cards across all sets, loaded {card_count} unique cards"
        )
        return card_count

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

                # Check for meta/data structure
                if not isinstance(cards_data, dict):
                    logger.error(
                        f"Expected dictionary at top level, got {type(cards_data)}"
                    )
                    return 0

                if "data" not in cards_data:
                    logger.error("No 'data' field found in cards file")
                    return 0

                # Get the data section which should contain sets
                sets_data = cards_data["data"]

                if not isinstance(sets_data, dict):
                    logger.error(
                        f"Expected dictionary in 'data' section, got {type(sets_data)}"
                    )
                    return 0

                # Count for logging
                total_cards_processed = 0
                unique_cards_found = 0

                # Iterate through each set
                for set_code, set_data in sets_data.items():
                    # Skip meta field if it exists at this level
                    if set_code == "meta":
                        continue

                    # Check if this set has cards
                    if not isinstance(set_data, dict) or "cards" not in set_data:
                        logger.debug(f"Set {set_code} has no cards field")
                        continue

                    # Get the cards array for this set
                    cards_in_set = set_data["cards"]

                    if not isinstance(cards_in_set, list):
                        logger.warning(
                            f"Expected list in 'cards' field of set {set_code}, got {type(cards_in_set)}"
                        )
                        continue

                    total_cards_processed += len(cards_in_set)

                    # Process each card in the set
                    for card_data in cards_in_set:
                        if not isinstance(card_data, dict):
                            continue

                        # Skip non-English cards
                        if card_data.get("language", "English") != "English":
                            continue

                        # Get card name
                        card_name = card_data.get("name")
                        if not card_name:
                            continue

                        # Process card
                        if card_name not in self.cards:
                            self.cards[card_name] = self._process_card_data(card_data)
                            unique_cards_found += 1
                        elif self._should_update_card(self.cards[card_name], card_data):
                            # Replace with better version
                            self.cards[card_name] = self._process_card_data(card_data)

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

    def _process_card_data(self, card_data: Dict[str, Any]) -> CardType:
        """
        Convert raw card data to standardized format.

        Args:
            card_data: Raw card data from JSON

        Returns:
            Processed card data
        """
        # Initialize with basic properties, handling different possible field names
        card: CardType = {"name": card_data.get("name", "")}

        # Add mana cost information
        if "mana_cost" in card_data:
            card["mana_cost"] = card_data["mana_cost"]
        elif "manaCost" in card_data:
            card["mana_cost"] = card_data["manaCost"]

        # Add converted mana cost / mana value
        if "cmc" in card_data:
            card["cmc"] = card_data["cmc"]
        elif "convertedManaCost" in card_data:
            card["cmc"] = card_data["convertedManaCost"]
        elif "manaValue" in card_data:
            card["cmc"] = card_data["manaValue"]

        # Add type information
        if "type_line" in card_data:
            card["type_line"] = card_data["type_line"]
        elif "type" in card_data:
            card["type_line"] = card_data["type"]

        # Add card text
        if "oracle_text" in card_data:
            card["oracle_text"] = card_data["oracle_text"]
        elif "text" in card_data:
            card["oracle_text"] = card_data["text"]

        # Add color information
        if "colors" in card_data:
            card["colors"] = card_data["colors"]

        if "color_identity" in card_data:
            card["color_identity"] = card_data["color_identity"]
        elif "colorIdentity" in card_data:
            card["color_identity"] = card_data["colorIdentity"]

        # Add power/toughness for creatures
        if "power" in card_data:
            card["power"] = card_data["power"]

        if "toughness" in card_data:
            card["toughness"] = card_data["toughness"]

        # Add loyalty for planeswalkers
        if "loyalty" in card_data:
            card["loyalty"] = card_data["loyalty"]

        # Add keywords
        if "keywords" in card_data:
            card["keywords"] = card_data["keywords"]

        # Add legalities
        if "legalities" in card_data:
            card["legalities"] = card_data["legalities"]

        # Add type information (for detailed categorization)
        if "types" in card_data:
            card["types"] = card_data["types"]

        if "subtypes" in card_data:
            card["subtypes"] = card_data["subtypes"]

        if "supertypes" in card_data:
            card["supertypes"] = card_data["supertypes"]

        # Add rulings if available
        if "rulings" in card_data:
            card["rulings"] = card_data["rulings"]

        return card

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
            category_number = category.get("category_number", "")
            category_title = category.get("title", "")

            # Process sections in each category
            for section in category.get("sections", []):
                section_number = section.get("section_number", "")
                section_title = section.get("title", "")

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
        """
        Add card documents to the document list.

        Args:
            documents: List to add documents to
        """
        for card_name, card in self.cards.items():
            doc_text = self._format_card_document_text(card_name, card)
            documents.append(
                {
                    "id": f"card_{card_name.lower().replace(' ', '_')}",
                    "type": "card",
                    "text": doc_text,
                }
            )

    def _add_rule_documents(self, documents: List[DocumentType]) -> None:
        """
        Add individual rule documents to the document list.

        Args:
            documents: List to add documents to
        """
        for rule_id, rule_text in self.rules.items():
            # Skip if the rule_text isn't a string (could be a list or dict in some formats)
            if not isinstance(rule_text, str):
                continue

            documents.append(
                {
                    "id": f"rule_{rule_id}",
                    "type": "rule",
                    "text": f"Rule {rule_id}: {rule_text}",
                }
            )

    def _format_card_document_text(self, card_name: str, card: CardType) -> str:
        """
        Format card data as text for document.

        Args:
            card_name: Name of the card
            card: Card data

        Returns:
            Formatted document text
        """
        doc_text = f"Card Name: {card_name}\n"

        # Add basic card information
        if "mana_cost" in card and card["mana_cost"]:
            doc_text += f"Mana Cost: {card['mana_cost']}\n"

        if "type_line" in card and card["type_line"]:
            doc_text += f"Type: {card['type_line']}\n"
        elif "type" in card:  # Try alternative field name
            doc_text += f"Type: {card['type']}\n"

        if "oracle_text" in card and card["oracle_text"]:
            doc_text += f"Text: {card['oracle_text']}\n"
        elif "text" in card:  # Try alternative field name
            doc_text += f"Text: {card['text']}\n"

        if "power" in card and "toughness" in card:
            doc_text += f"Power/Toughness: {card['power']}/{card['toughness']}\n"

        if "loyalty" in card and card["loyalty"]:
            doc_text += f"Loyalty: {card['loyalty']}\n"

        if "keywords" in card and card["keywords"]:
            doc_text += f"Keywords: {', '.join(card['keywords'])}\n"

        # Add type information
        types = []
        if "supertypes" in card and card["supertypes"]:
            types.extend(card["supertypes"])

        if "types" in card and card["types"]:
            types.extend(card["types"])

        if "subtypes" in card and card["subtypes"]:
            types.extend(card["subtypes"])

        if types:
            doc_text += f"All Types: {', '.join(types)}\n"

        # Add rulings if available
        if "rulings" in card and card["rulings"]:
            doc_text += "\nRulings:\n"
            for ruling in card["rulings"]:
                doc_text += f"- [{ruling['date']}] {ruling['text']}\n"

        return doc_text

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
        results = []
        query_lower = query.lower()

        for card_name, card in self.cards.items():
            score = 0

            # Check name match
            if query_lower in card_name.lower():
                score += 10
                # Exact match gets highest priority
                if query_lower == card_name.lower():
                    score += 100

            # Check text match - try different field names
            card_text = ""
            if "oracle_text" in card and card["oracle_text"]:
                card_text = card["oracle_text"]
            elif "text" in card and card["text"]:
                card_text = card["text"]

            if card_text and query_lower in card_text.lower():
                score += 5

            # Check type match - try different field names
            card_type = ""
            if "type_line" in card and card["type_line"]:
                card_type = card["type_line"]
            elif "type" in card and card["type"]:
                card_type = card["type"]

            if card_type and query_lower in card_type.lower():
                score += 3

            if score > 0:
                # Create a copy with name included and add score
                card_copy = dict(card)
                card_copy["name"] = card_name
                card_copy["score"] = score
                results.append(card_copy)

        # Sort by score and return top matches
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

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
