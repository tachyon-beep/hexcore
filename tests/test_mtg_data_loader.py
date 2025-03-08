# tests/data/test_mtg_data_loader.py

import unittest
import os
import logging
from src.data.mtg_data_loader import MTGDataLoader

# Configure logging for the test
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_mtg_data_loader")


class TestMTGDataLoader(unittest.TestCase):
    """Test cases for the MTGDataLoader class."""

    def setUp(self):
        """Set up the test environment."""
        self.data_dir = "data"
        self.loader = MTGDataLoader(data_dir=self.data_dir)

        # Check if test data files exist
        self.cards_file_exists = os.path.exists(
            os.path.join(self.data_dir, "cards.json")
        )
        self.rules_file_exists = os.path.exists(
            os.path.join(self.data_dir, "rules.json")
        )
        self.glossary_file_exists = os.path.exists(
            os.path.join(self.data_dir, "glossary.json")
        )

        logger.info("Data files status:")
        logger.info(
            f"  Cards file: {'EXISTS' if self.cards_file_exists else 'MISSING'}"
        )
        logger.info(
            f"  Rules file: {'EXISTS' if self.rules_file_exists else 'MISSING'}"
        )
        logger.info(
            f"  Glossary file: {'EXISTS' if self.glossary_file_exists else 'MISSING'}"
        )

    def test_load_cards(self):
        """Test loading card data."""
        if not self.cards_file_exists:
            logger.warning("Skipping card loading test: cards.json not found")
            self.skipTest("cards.json not found")

        card_count = self.loader.load_cards()
        logger.info(f"Loaded {card_count} cards")

        # Verify that cards were loaded
        self.assertGreater(card_count, 0, "Should load at least one card")
        self.assertGreater(
            len(self.loader.cards), 0, "Cards dictionary should not be empty"
        )

    def test_load_rules(self):
        """Test loading rules data."""
        if not self.rules_file_exists:
            logger.warning("Skipping rules loading test: rules.json not found")
            self.skipTest("rules.json not found")

        rule_count = self.loader.load_rules()
        logger.info(f"Loaded {rule_count} rules")

        # Verify that rules were loaded
        self.assertGreater(rule_count, 0, "Should load at least one rule")
        self.assertGreater(
            len(self.loader.rules), 0, "Rules dictionary should not be empty"
        )

    def test_create_documents(self):
        """Test creating documents for retrieval."""
        # First load both cards and rules
        if self.cards_file_exists:
            self.loader.load_cards()
        if self.rules_file_exists:
            self.loader.load_rules()

        # Create documents
        documents = self.loader._create_documents_for_retrieval()
        logger.info(f"Created {len(documents)} documents")

        # Verify documents were created
        self.assertGreater(len(documents), 0, "Should create at least one document")

        # Check document structure
        if documents:
            doc = documents[0]
            self.assertIn("id", doc, "Document should have an id")
            self.assertIn("type", doc, "Document should have a type")
            self.assertIn("text", doc, "Document should have text content")

    def test_card_lookup(self):
        """Test looking up cards by name."""
        if not self.cards_file_exists:
            logger.warning("Skipping card lookup test: cards.json not found")
            self.skipTest("cards.json not found")

        # Load cards first
        self.loader.load_cards()

        # Test cases - add some well-known cards that should be in most MTG datasets
        test_cases = [
            ("Lightning Bolt", True),  # Should exist
            ("Black Lotus", True),  # Should exist
            ("Island", True),  # Should exist
            ("NonexistentCardName123456789", False),  # Should not exist
        ]

        for card_name, should_exist in test_cases:
            card = self.loader.get_card(card_name)
            if should_exist:
                self.assertIsNotNone(card, f"Card '{card_name}' should exist")
                if card:
                    logger.info(f"Found card: {card_name}")
            else:
                self.assertIsNone(card, f"Card '{card_name}' should not exist")

    def test_rule_lookup(self):
        """Test looking up rules by ID."""
        if not self.rules_file_exists:
            logger.warning("Skipping rule lookup test: rules.json not found")
            self.skipTest("rules.json not found")

        # Load rules first
        self.loader.load_rules()

        # Test cases - common rule IDs that should exist in most comprehensive rules
        test_cases = [
            ("101.1", True),  # Should exist (basic rule)
            ("100.2a", True),  # Should exist (specific rule)
            ("999.999", False),  # Should not exist
        ]

        for rule_id, should_exist in test_cases:
            rule = self.loader.get_rule(rule_id)
            if should_exist:
                self.assertIsNotNone(rule, f"Rule '{rule_id}' should exist")
                if rule:
                    logger.info(f"Found rule {rule_id}: {rule[:50]}...")
            else:
                self.assertIsNone(rule, f"Rule '{rule_id}' should not exist")

    def test_card_search(self):
        """Test searching for cards."""
        if not self.cards_file_exists:
            logger.warning("Skipping card search test: cards.json not found")
            self.skipTest("cards.json not found")

        # Load cards first
        self.loader.load_cards()

        # Search terms that should yield results
        search_terms = ["lightning", "angel", "counter"]

        for term in search_terms:
            results = self.loader.search_cards(term, limit=5)
            logger.info(f"Search for '{term}' returned {len(results)} results")

            # There should be some results for these common terms
            self.assertGreater(
                len(results), 0, f"Search for '{term}' should return results"
            )

            # Check that results are properly formatted
            if results:
                self.assertIn(
                    "name", results[0], "Search result should include card name"
                )
                self.assertIn("score", results[0], "Search result should include score")

    def test_rule_search(self):
        """Test searching for rules."""
        if not self.rules_file_exists:
            logger.warning("Skipping rule search test: rules.json not found")
            self.skipTest("rules.json not found")

        # Load rules first
        self.loader.load_rules()

        # Search terms that should yield results
        search_terms = ["mulligan", "commander", "turn"]

        for term in search_terms:
            results = self.loader.search_rules(term, limit=5)
            logger.info(f"Search for '{term}' returned {len(results)} results")

            # There should be some results for these common terms
            self.assertGreater(
                len(results), 0, f"Search for '{term}' should return results"
            )

            # Check that results are properly formatted
            if results:
                self.assertIn(
                    "rule_id", results[0], "Search result should include rule ID"
                )
                self.assertIn(
                    "text", results[0], "Search result should include rule text"
                )
                self.assertIn("score", results[0], "Search result should include score")


if __name__ == "__main__":
    unittest.main()
