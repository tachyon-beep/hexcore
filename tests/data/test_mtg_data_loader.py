# tests/data/test_mtg_data_loader.py

import os
import pytest
import logging
import sys
from pathlib import Path

# Add src directory to Python path if needed
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.mtg_data_loader import MTGDataLoader

# Configure logging for the test
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_mtg_data_loader")


@pytest.fixture
def data_loader():
    """Set up the MTGDataLoader for testing."""
    data_dir = "data"
    return MTGDataLoader(data_dir=data_dir)


@pytest.fixture
def data_paths(data_loader):
    """Check if test data files exist and log their status."""
    data_dir = data_loader.data_dir
    cards_file_exists = os.path.exists(os.path.join(data_dir, "cards.json"))
    rules_file_exists = os.path.exists(os.path.join(data_dir, "rules.json"))
    glossary_file_exists = os.path.exists(os.path.join(data_dir, "glossary.json"))

    logger.info("Data files status:")
    logger.info(f"  Cards file: {'EXISTS' if cards_file_exists else 'MISSING'}")
    logger.info(f"  Rules file: {'EXISTS' if rules_file_exists else 'MISSING'}")
    logger.info(f"  Glossary file: {'EXISTS' if glossary_file_exists else 'MISSING'}")

    return {
        "cards_file_exists": cards_file_exists,
        "rules_file_exists": rules_file_exists,
        "glossary_file_exists": glossary_file_exists,
    }


@pytest.mark.data
def test_load_cards(data_loader, data_paths):
    """Test loading card data."""
    if not data_paths["cards_file_exists"]:
        logger.warning("Skipping card loading test: cards.json not found")
        pytest.skip("cards.json not found")

    card_count = data_loader.load_cards()
    logger.info(f"Loaded {card_count} cards")

    # Verify that cards were loaded
    assert card_count > 0, "Should load at least one card"
    assert len(data_loader.cards) > 0, "Cards dictionary should not be empty"


@pytest.mark.data
def test_load_rules(data_loader, data_paths):
    """Test loading rules data."""
    if not data_paths["rules_file_exists"]:
        logger.warning("Skipping rules loading test: rules.json not found")
        pytest.skip("rules.json not found")

    rule_count = data_loader.load_rules()
    logger.info(f"Loaded {rule_count} rules")

    # Verify that rules were loaded
    assert rule_count > 0, "Should load at least one rule"
    assert len(data_loader.rules) > 0, "Rules dictionary should not be empty"


@pytest.mark.data
def test_create_documents(data_loader, data_paths):
    """Test creating documents for retrieval."""
    # First load both cards and rules
    if data_paths["cards_file_exists"]:
        data_loader.load_cards()
    if data_paths["rules_file_exists"]:
        data_loader.load_rules()

    # Create documents
    documents = data_loader._create_documents_for_retrieval()
    logger.info(f"Created {len(documents)} documents")

    # Verify documents were created
    assert len(documents) > 0, "Should create at least one document"

    # Check document structure
    if documents:
        doc = documents[0]
        assert "id" in doc, "Document should have an id"
        assert "type" in doc, "Document should have a type"
        assert "text" in doc, "Document should have text content"


@pytest.mark.data
def test_card_lookup(data_loader, data_paths):
    """Test looking up cards by name."""
    if not data_paths["cards_file_exists"]:
        logger.warning("Skipping card lookup test: cards.json not found")
        pytest.skip("cards.json not found")

    # Load cards first
    data_loader.load_cards()

    # Test cases - add some well-known cards that should be in most MTG datasets
    test_cases = [
        ("Lightning Bolt", True),  # Should exist
        ("Black Lotus", True),  # Should exist
        ("Island", True),  # Should exist
        ("NonexistentCardName123456789", False),  # Should not exist
    ]

    for card_name, should_exist in test_cases:
        card = data_loader.get_card(card_name)
        if should_exist:
            assert card is not None, f"Card '{card_name}' should exist"
            if card:
                logger.info(f"Found card: {card_name}")
        else:
            assert card is None, f"Card '{card_name}' should not exist"


@pytest.mark.data
def test_rule_lookup(data_loader, data_paths):
    """Test looking up rules by ID."""
    if not data_paths["rules_file_exists"]:
        logger.warning("Skipping rule lookup test: rules.json not found")
        pytest.skip("rules.json not found")

    # Load rules first
    data_loader.load_rules()

    # Test cases - common rule IDs that should exist in most comprehensive rules
    test_cases = [
        ("101.1", True),  # Should exist (basic rule)
        ("100.2a", True),  # Should exist (specific rule)
        ("999.999", False),  # Should not exist
    ]

    for rule_id, should_exist in test_cases:
        rule = data_loader.get_rule(rule_id)
        if should_exist:
            assert rule is not None, f"Rule '{rule_id}' should exist"
            if rule:
                logger.info(f"Found rule {rule_id}: {rule[:50]}...")
        else:
            assert rule is None, f"Rule '{rule_id}' should not exist"


@pytest.mark.data
def test_card_search(data_loader, data_paths):
    """Test searching for cards."""
    if not data_paths["cards_file_exists"]:
        logger.warning("Skipping card search test: cards.json not found")
        pytest.skip("cards.json not found")

    # Load cards first
    data_loader.load_cards()

    # Search terms that should yield results
    search_terms = ["lightning", "angel", "counter"]

    for term in search_terms:
        results = data_loader.search_cards(term, limit=5)
        logger.info(f"Search for '{term}' returned {len(results)} results")

        # There should be some results for these common terms
        assert len(results) > 0, f"Search for '{term}' should return results"

        # Check that results are properly formatted
        if results:
            assert "name" in results[0], "Search result should include card name"
            assert "score" in results[0], "Search result should include score"


@pytest.mark.data
def test_rule_search(data_loader, data_paths):
    """Test searching for rules."""
    if not data_paths["rules_file_exists"]:
        logger.warning("Skipping rule search test: rules.json not found")
        pytest.skip("rules.json not found")

    # Load rules first
    data_loader.load_rules()

    # Search terms that should yield results
    search_terms = ["mulligan", "commander", "turn"]

    for term in search_terms:
        results = data_loader.search_rules(term, limit=5)
        logger.info(f"Search for '{term}' returned {len(results)} results")

        # There should be some results for these common terms
        assert len(results) > 0, f"Search for '{term}' should return results"

        # Check that results are properly formatted
        if results:
            assert "rule_id" in results[0], "Search result should include rule ID"
            assert "text" in results[0], "Search result should include rule text"
            assert "score" in results[0], "Search result should include score"
