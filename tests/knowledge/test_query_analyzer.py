# tests/knowledge/test_query_analyzer.py

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.knowledge.query_analyzer import QueryAnalyzer


class TestQueryAnalyzer:
    """Test cases for the QueryAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create a QueryAnalyzer instance for testing."""
        return QueryAnalyzer()

    @pytest.fixture
    def analyzer_with_data(self):
        """Create a QueryAnalyzer with mock data loader."""
        # Create mock data loader with card data
        data_loader = MagicMock()
        data_loader.cards = {
            "Lightning Bolt": {"id": "1", "text": "Deal 3 damage to any target."},
            "Black Lotus": {"id": "2", "text": "Add three mana of any one color."},
            "Counterspell": {"id": "3", "text": "Counter target spell."},
            "Wrath of God": {"id": "4", "text": "Destroy all creatures."},
        }
        return QueryAnalyzer(data_loader)

    def test_initialization(self):
        """Test initialization of QueryAnalyzer."""
        analyzer = QueryAnalyzer()

        # Verify initialization
        assert analyzer.data_loader is None
        assert len(analyzer.patterns) > 0
        assert len(analyzer.entity_types) > 0

        # Test with data loader
        data_loader = MagicMock()
        analyzer_with_data = QueryAnalyzer(data_loader)
        assert analyzer_with_data.data_loader is data_loader

    def test_extract_entities_basic(self, analyzer):
        """Test basic entity extraction without data loader."""
        # Test with a simple query
        query = "How does Flying work in Magic: The Gathering?"
        entities = analyzer._extract_entities(query)

        # Should extract Flying as a mechanic
        assert len(entities) >= 1
        assert any(e["type"] == "mechanic" and e["name"] == "Flying" for e in entities)

        # Test with multiple entities
        query = "What happens when a creature with Deathtouch blocks a creature with Trample?"
        entities = analyzer._extract_entities(query)

        # Should extract Deathtouch and Trample as mechanics
        assert len(entities) >= 2
        assert any(
            e["type"] == "mechanic" and e["name"] == "Deathtouch" for e in entities
        )
        assert any(e["type"] == "mechanic" and e["name"] == "Trample" for e in entities)

        # Test with a card name pattern
        query = 'What does the card "Ancestral Recall" do?'
        entities = analyzer._extract_entities(query)

        # Should extract Ancestral Recall as a card
        assert any(
            e["type"] == "card" and e["name"] == "Ancestral Recall" for e in entities
        )

    def test_extract_entities_with_data_loader(self, analyzer_with_data):
        """Test entity extraction with data loader."""
        # Test recognition of a card in the mock database
        query = "How does Lightning Bolt interact with Counterspell?"
        entities = analyzer_with_data._extract_entities(query)

        # Should extract both cards
        assert len(entities) >= 2
        assert any(
            e["type"] == "card" and e["name"] == "Lightning Bolt" for e in entities
        )
        assert any(
            e["type"] == "card" and e["name"] == "Counterspell" for e in entities
        )

        # Test with a card name not in quotes
        query = "Is Black Lotus banned in Commander?"
        entities = analyzer_with_data._extract_entities(query)

        # Should extract Black Lotus
        assert any(e["type"] == "card" and e["name"] == "Black Lotus" for e in entities)

    def test_query_type_classification(self, analyzer):
        """Test classification of query types."""
        # Test card lookup queries
        card_queries = [
            "What does the card Lightning Bolt do?",
            "Tell me about the card Black Lotus",
            "Find card named Counterspell",
        ]
        for query in card_queries:
            result = analyzer.analyze_query(query)
            assert result["query_type"] == "card_lookup"

        # Test rule lookup queries
        rule_queries = [
            "What is rule 702.2?",
            "Explain section 601.2b",
            "Show me rule 507.3",
        ]
        for query in rule_queries:
            result = analyzer.analyze_query(query)
            assert result["query_type"] == "rule_lookup"

        # Test mechanic lookup queries
        mechanic_queries = [
            "How does Flying work?",
            "Explain the Cascade ability",
            "What is Hexproof?",
        ]
        for query in mechanic_queries:
            result = analyzer.analyze_query(query)
            assert result["query_type"] == "mechanic_lookup"

        # Test complex interaction queries
        interaction_queries = [
            "When Deathtouch and Trample interact, what happens?",
            "How does the stack work with triggered abilities?",
            "If I cast Lightning Bolt and they respond with Counterspell, what resolves first?",
        ]
        for query in interaction_queries:
            result = analyzer.analyze_query(query)
            assert result["query_type"] == "complex_interaction"

        # Test strategic queries
        strategic_queries = [
            "What's the best strategy for mulligans?",
            "When should I attack versus block?",
            "What's the best approach to building a combo deck?",
        ]
        for query in strategic_queries:
            result = analyzer.analyze_query(query)
            assert result["query_type"] == "strategic"

        # Test general queries
        general_queries = [
            "Who created Magic: The Gathering?",
            "When was the game released?",
            "How many cards are in the game?",
        ]
        for query in general_queries:
            result = analyzer.analyze_query(query)
            assert result["query_type"] == "general"

    def test_retrieval_strategy_determination(self, analyzer):
        """Test determination of retrieval strategy."""
        # Test rule lookup (should prioritize graph)
        rule_query = "What is rule 702.2?"
        result = analyzer.analyze_query(rule_query)
        assert result["requires_structured_knowledge"] is True
        assert result["prioritize_graph"] is True

        # Test complex interaction (should prioritize graph)
        interaction_query = "How do Deathtouch and Trample interact?"
        result = analyzer.analyze_query(interaction_query)
        assert result["requires_structured_knowledge"] is True
        assert result["prioritize_graph"] is True

        # Test card lookup (should use structured but not prioritize graph)
        card_query = "What does Lightning Bolt do?"
        result = analyzer.analyze_query(card_query)
        assert result["requires_structured_knowledge"] is True
        assert result["prioritize_graph"] is False

        # Test general query (should use vector search)
        general_query = "Who created Magic: The Gathering?"
        result = analyzer.analyze_query(general_query)
        assert result["requires_structured_knowledge"] is False
        assert result["prioritize_graph"] is False

    def test_relationship_extraction(self, analyzer):
        """Test extraction of relationship types."""
        # Test relationship extraction for card lookup
        card_query = "What cards have Flying?"
        result = analyzer.analyze_query(card_query)
        assert "card_uses_mechanic" in result["relationship_types"]

        # Test relationship extraction for rule lookup
        rule_query = "What are the rules governing Trample?"
        result = analyzer.analyze_query(rule_query)
        assert "mechanic_governed_by_rule" in result["relationship_types"]

        # Test relationship extraction for complex interactions
        interaction_query = "What cards are mentioned in rule 702.19?"
        result = analyzer.analyze_query(interaction_query)
        assert "card_referenced_in_rule" in result["relationship_types"]

    def test_query_complexity(self, analyzer):
        """Test query complexity calculation."""
        # Test simple query
        simple_query = "What is Flying?"
        complexity = analyzer.get_query_complexity(simple_query)
        assert complexity["category"] == "simple"
        assert complexity["entity_count"] >= 1  # Should find Flying

        # Test moderate query
        moderate_query = "How does Deathtouch work with Trample?"
        complexity = analyzer.get_query_complexity(moderate_query)
        assert complexity["category"] == "moderate"
        assert complexity["entity_count"] >= 2  # Should find both mechanics

        # Test complex query
        complex_query = "If I have a creature with Deathtouch and Trample attacking, and my opponent blocks with a 1/1, how much damage goes through to the player according to rule 702.2?"
        complexity = analyzer.get_query_complexity(complex_query)
        assert complexity["category"] == "complex"
        assert complexity["entity_count"] >= 2  # Should find mechanics
        assert complexity["rule_references"] >= 1  # Should find rule reference
        assert (
            complexity["relationship_indicators"] >= 1
        )  # Should find relationship indicators

    def test_analyze_query_integration(self, analyzer_with_data):
        """Test end-to-end query analysis."""
        query = "How do Lightning Bolt and Counterspell interact on the stack?"

        # Analyze query
        result = analyzer_with_data.analyze_query(query)

        # Verify results
        assert result["query_type"] == "complex_interaction"
        assert len(result["entities"]) >= 2  # Should find both cards
        assert result["requires_structured_knowledge"] is True
        assert result["prioritize_graph"] is True

        # Should include some relationship types
        assert len(result["relationship_types"]) > 0

        # Verify entities were extracted correctly
        entity_names = [e["name"] for e in result["entities"]]
        assert "Lightning Bolt" in entity_names
        assert "Counterspell" in entity_names
