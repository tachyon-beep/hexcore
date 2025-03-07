# src/data/mtg_data_loader.py
import json
import os
from typing import Dict, List, Optional, Any
import logging
import requests

logger = logging.getLogger(__name__)


class MTGDataLoader:
    """
    Load and manage MTG card and rules data.
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize the MTG data loader."""
        self.data_dir = data_dir
        self.sets_file = os.path.join(data_dir, "AllSets.json")  # For loading from sets
        self.cards_file = os.path.join(data_dir, "cards.json")
        self.rules_file = os.path.join(data_dir, "comprehensive_rules.json")
        os.makedirs(data_dir, exist_ok=True)
        self.cards = {}
        self.rules = {}
        self.card_names = set()  # Keep track of unique card names

    def load_or_download_data(self, force_download: bool = False):
        """Load or download all necessary data."""
        # Check if we should load from sets file instead of Scryfall
        if os.path.exists(self.sets_file):
            self.load_cards_from_sets(force_download)
        else:
            self.load_cards_from_scryfall(force_download)

        self.load_comprehensive_rules(force_download)

        # Create documents for retrieval
        documents = self._create_documents_for_retrieval()
        return documents

    def load_cards_from_sets(self, force_process: bool = False) -> int:
        """
        Load card data from sets JSON file and process it.

        Args:
            force_process: Whether to force reprocessing even if cache exists

        Returns:
            Number of unique cards loaded
        """
        # Check if processed cards cache exists
        if os.path.exists(self.cards_file) and not force_process:
            logger.info(f"Loading processed cards from cache: {self.cards_file}")
            with open(self.cards_file, "r", encoding="utf-8") as f:
                self.cards = json.load(f)
                self.card_names = set(self.cards.keys())
            logger.info(f"Loaded {len(self.cards)} processed cards from cache")
            return len(self.cards)

        # Check if raw sets file exists
        if not os.path.exists(self.sets_file):
            logger.error(f"Sets file not found: {self.sets_file}")
            return 0

        logger.info(f"Loading cards from sets file: {self.sets_file}")

        # Load all sets from JSON file
        with open(self.sets_file, "r", encoding="utf-8") as f:
            sets_data = json.load(f)

        # Process all sets and extract cards
        self.cards = {}
        set_codes = set()
        card_count = 0

        for set_code, set_data in sets_data.items():
            set_codes.add(set_code)

            # Process each card in the set
            for card in set_data.get("cards", []):
                # Skip non-English cards and special layouts if needed
                if card.get("language", "English") != "English":
                    continue

                # Skip certain card layouts if needed
                skip_layouts = ["token", "scheme", "plane", "phenomenon", "vanguard"]
                if card.get("layout") in skip_layouts:
                    continue

                # Process this card
                card_name = card["name"]
                self.card_names.add(card_name)

                # We may have multiple printings of the same card
                # Choose the most recent one or the one with the most information
                if card_name in self.cards:
                    # Check if this version has more information and use it if so
                    existing_version = self.cards[card_name]
                    if (not existing_version.get("text") and card.get("text")) or (
                        not existing_version.get("rulings") and card.get("rulings")
                    ):
                        self.cards[card_name] = self._process_card(card, set_data)
                else:
                    # New card, add it
                    self.cards[card_name] = self._process_card(card, set_data)
                    card_count += 1

        logger.info(f"Processed {card_count} unique cards from {len(set_codes)} sets")

        # Save processed cards to cache
        with open(self.cards_file, "w", encoding="utf-8") as f:
            json.dump(self.cards, f, indent=2)
        logger.info(f"Saved processed cards to: {self.cards_file}")

        return card_count

    def _process_card(
        self, card: Dict[str, Any], set_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a single card to extract relevant information.

        Args:
            card: Raw card data from the JSON
            set_data: Set data containing this card

        Returns:
            Processed card with relevant fields
        """
        processed_card = {
            "name": card["name"],
            "mana_cost": card.get("manaCost", ""),
            "cmc": card.get("manaValue", 0),
            "type_line": card.get("type", ""),
            "oracle_text": card.get("text", ""),
            "colors": card.get("colors", []),
            "color_identity": card.get("colorIdentity", []),
            "power": card.get("power", ""),
            "toughness": card.get("toughness", ""),
            "loyalty": card.get("loyalty", ""),
            "keywords": card.get("keywords", []),
            "legalities": card.get("legalities", {}),
            "layout": card.get("layout", "normal"),
            "rarity": card.get("rarity", ""),
            "set": card.get("setCode", set_data.get("code", "")),
            "set_name": set_data.get("name", ""),
            "subtypes": card.get("subtypes", []),
            "supertypes": card.get("supertypes", []),
            "types": card.get("types", []),
        }

        # Process rulings if available (handling the specific Rulings type structure)
        if "rulings" in card and card["rulings"]:
            processed_rulings = []
            for ruling in card["rulings"]:
                # Ensure ruling has the expected structure
                if isinstance(ruling, dict) and "date" in ruling and "text" in ruling:
                    processed_rulings.append(
                        {"date": ruling["date"], "text": ruling["text"]}
                    )

            if processed_rulings:
                processed_card["rulings"] = processed_rulings

        # Handle special layouts
        if card.get("faceName"):
            processed_card["face_name"] = card["faceName"]

        if card.get("otherFaceIds"):
            processed_card["other_face_ids"] = card["otherFaceIds"]

        return processed_card

    def load_cards_from_scryfall(self, force_download: bool = False):
        """Load card data from Scryfall API or local cache."""
        if os.path.exists(self.cards_file) and not force_download:
            logger.info(f"Loading cards from cache: {self.cards_file}")
            with open(self.cards_file, "r", encoding="utf-8") as f:
                self.cards = json.load(f)
            logger.info(f"Loaded {len(self.cards)} cards from cache")
            return

        logger.info("Downloading cards from Scryfall API")
        self.cards = {}

        # Scryfall API endpoint for bulk data
        bulk_data_url = "https://api.scryfall.com/bulk-data"
        response = requests.get(bulk_data_url)
        bulk_data = response.json()

        # Find the Oracle Cards download URL
        oracle_cards_url = None
        for item in bulk_data["data"]:
            if item["type"] == "oracle_cards":
                oracle_cards_url = item["download_uri"]
                break

        if not oracle_cards_url:
            raise ValueError("Could not find Oracle Cards download URL")

        # Download Oracle Cards
        logger.info(f"Downloading Oracle Cards from {oracle_cards_url}")
        response = requests.get(oracle_cards_url)
        all_cards = response.json()

        # Process cards
        for card in all_cards:
            # Skip non-paper cards and tokens
            if "paper" not in card.get("games", []) or card.get("layout") == "token":
                continue

            # Store card by name with key fields
            card_name = card["name"]
            self.cards[card_name] = {
                "name": card_name,
                "mana_cost": card.get("mana_cost", ""),
                "cmc": card.get("cmc", 0),
                "type_line": card.get("type_line", ""),
                "oracle_text": card.get("oracle_text", ""),
                "colors": card.get("colors", []),
                "color_identity": card.get("color_identity", []),
                "power": card.get("power", ""),
                "toughness": card.get("toughness", ""),
                "loyalty": card.get("loyalty", ""),
                "legalities": card.get("legalities", {}),
                "set": card.get("set", ""),
                "image_uris": card.get("image_uris", {}),
            }

            # Process rulings if available
            if "rulings_uri" in card:
                rulings_url = card["rulings_uri"]
                rulings_response = requests.get(rulings_url)
                if rulings_response.status_code == 200:
                    rulings_data = rulings_response.json()
                    if "data" in rulings_data and rulings_data["data"]:
                        self.cards[card_name]["rulings"] = [
                            {"date": ruling["published_at"], "text": ruling["comment"]}
                            for ruling in rulings_data["data"]
                        ]

        logger.info(f"Downloaded {len(self.cards)} cards")

        # Save to cache
        with open(self.cards_file, "w", encoding="utf-8") as f:
            json.dump(self.cards, f)
        logger.info(f"Saved cards to cache: {self.cards_file}")

    def load_comprehensive_rules(self, force_download: bool = False):
        """Load comprehensive rules from web or local cache."""
        if os.path.exists(self.rules_file) and not force_download:
            logger.info(f"Loading rules from cache: {self.rules_file}")
            with open(self.rules_file, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            logger.info(f"Loaded {len(self.rules)} rule entries from cache")
            return

        # In a real implementation, we would download and parse the rules
        # from the official source. For now, we'll create a placeholder.
        logger.warning("Comprehensive rules download not yet implemented")
        self.rules = {
            "101.1": "Magic is a game where players cast spells and summon creatures to defeat their opponents.",
            "102.1": "A player wins the game if all opponents have left the game or a game effect states that player wins.",
            "103.1": "At the start of a game, each player shuffles their deck to form their library.",
            # Add more rules as needed
        }

        # Save to cache
        with open(self.rules_file, "w", encoding="utf-8") as f:
            json.dump(self.rules, f)
        logger.info(f"Saved rules to cache: {self.rules_file}")

    def _create_documents_for_retrieval(self) -> List[Dict[str, str]]:
        """Create documents for retrieval system."""
        documents = []

        # Add card documents
        for card_name, card in self.cards.items():
            doc_text = f"Card Name: {card_name}\n"

            if card.get("mana_cost"):
                doc_text += f"Mana Cost: {card['mana_cost']}\n"

            if card.get("type_line"):
                doc_text += f"Type: {card['type_line']}\n"

            if card.get("oracle_text"):
                doc_text += f"Text: {card['oracle_text']}\n"

            if card.get("power") and card.get("toughness"):
                doc_text += f"Power/Toughness: {card['power']}/{card['toughness']}\n"

            if card.get("loyalty"):
                doc_text += f"Loyalty: {card['loyalty']}\n"

            if card.get("keywords"):
                doc_text += f"Keywords: {', '.join(card['keywords'])}\n"

            # Add rulings with proper formatting
            if card.get("rulings"):
                doc_text += "\nRulings:\n"
                for ruling in card["rulings"]:
                    # Format date as YYYY-MM-DD for consistency
                    date = ruling.get("date", "")
                    text = ruling.get("text", "")
                    doc_text += f"- [{date}] {text}\n"

            # Create the main card document
            documents.append(
                {
                    "id": f"card_{card_name.lower().replace(' ', '_')}",
                    "type": "card",
                    "text": doc_text,
                }
            )

            # For cards with rulings, create additional documents for each ruling
            if card.get("rulings") and len(card["rulings"]) > 0:
                for i, ruling in enumerate(card["rulings"]):
                    ruling_text = f"Card: {card_name}\n"
                    ruling_text += f"Ruling: {ruling['text']}\n"
                    ruling_text += f"Date: {ruling['date']}\n"

                    # Add some context about the card
                    ruling_text += f"Card Type: {card.get('type_line', '')}\n"
                    if card.get("oracle_text"):
                        ruling_text += f"Card Text: {card.get('oracle_text', '')}\n"

                    documents.append(
                        {
                            "id": f"ruling_{card_name.lower().replace(' ', '_')}_{i}",
                            "type": "ruling",
                            "text": ruling_text,
                        }
                    )

        # Add rule documents
        for rule_id, rule_text in self.rules.items():
            documents.append(
                {
                    "id": f"rule_{rule_id}",
                    "type": "rule",
                    "text": f"Rule {rule_id}: {rule_text}",
                }
            )

        logger.info(f"Created {len(documents)} documents for retrieval")
        return documents

    def get_card(self, name: str) -> Optional[Dict[str, Any]]:
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
        return self.rules.get(rule_id)

    def search_cards(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for cards by name, text, or type.

        Args:
            query: Search query
            limit: Maximum number of results to return

        Returns:
            List of matching cards
        """
        results = []
        query_lower = query.lower()

        # Search by name, text, and type
        for card_name, card in self.cards.items():
            score = 0

            # Check name match
            if query_lower in card_name.lower():
                score += 10

                # Exact match gets highest priority
                if query_lower == card_name.lower():
                    score += 100

            # Check text match
            if (
                card.get("oracle_text")
                and query_lower in card.get("oracle_text", "").lower()
            ):
                score += 5

            # Check type match
            if (
                card.get("type_line")
                and query_lower in card.get("type_line", "").lower()
            ):
                score += 3

            # Keywords match
            for keyword in card.get("keywords", []):
                if query_lower in keyword.lower():
                    score += 2

            if score > 0:
                results.append((score, card_name, card))

        # Sort by score and return top matches
        results.sort(reverse=True, key=lambda x: x[0])
        return [{"name": name, **card} for _, name, card in results[:limit]]
