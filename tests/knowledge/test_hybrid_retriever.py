# tests/knowledge/test_hybrid_retriever.py

import sys
import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.knowledge.hybrid_retriever import HybridRetriever


class TestHybridRetriever:
    """Test cases for the HybridRetriever class."""

    @pytest.fixture
    def mock_vector_retriever(self):
        """Create a mock vector retriever."""
        vector_retriever = MagicMock()

        # Mock retrieve method
        vector_retriever.retrieve.return_value = [
            {"id": "v1", "text": "Vector result 1", "type": "card", "score": 0.9},
            {"id": "v2", "text": "Vector result 2", "type": "rule", "score": 0.8},
        ]

        # Mock index initialization
        vector_retriever.index_documents.return_value = None

        return vector_retriever

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph."""
        knowledge_graph = MagicMock()

        # Mock query method
        knowledge_graph.query.return_value = [
            {"id": "g1", "text": "Graph result 1", "type": "card", "score": 0.9},
            {"id": "g2", "text": "Graph result 2", "type": "glossary", "score": 0.8},
        ]

        # Mock build graph method
        knowledge_graph.build_graph_from_data.return_value = None

        return knowledge_graph

    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager."""
        cache = MagicMock()

        # Mock get and set methods
        cache.get.return_value = None  # Default to cache miss
        cache.set.return_value = None

        # Mock invalidate method
        cache.invalidate.return_value = 5  # Number of entries invalidated

        # Mock metrics method
        cache.get_metrics.return_value = {
            "hits": 10,
            "misses": 5,
            "hit_rate_percent": 66.7,
        }

        return cache

    @pytest.fixture
    def mock_latency_tracker(self):
        """Create a mock latency tracker."""
        tracker = MagicMock()

        # Mock record method
        tracker.record.return_value = None

        # Mock statistics method
        tracker.get_statistics.return_value = {
            "overall": {
                "budget_exceeded_rate": 0.1,
                "alert_status": False,
            },
            "components": {
                "total": {"mean_ms": 150, "p95_ms": 200},
                "vector": {"mean_ms": 80, "p95_ms": 120},
                "graph": {"mean_ms": 60, "p95_ms": 90},
            },
        }

        return tracker

    @pytest.fixture
    def mock_query_analyzer(self):
        """Create a mock query analyzer."""
        analyzer = MagicMock()

        # Mock analyze_query method
        analyzer.analyze_query.return_value = {
            "query_type": "card_lookup",
            "entities": [{"type": "card", "name": "Lightning Bolt"}],
            "requires_structured_knowledge": True,
            "relationship_types": ["card_uses_mechanic"],
            "prioritize_graph": False,
        }

        return analyzer

    @pytest.fixture
    def mock_context_assembler(self):
        """Create a mock context assembler."""
        assembler = MagicMock()

        # Mock assemble_context method
        assembler.assemble_context.return_value = {
            "text": "Assembled context text",
            "documents": ["v1", "g1"],
            "token_count": 150,
            "metrics": {
                "docs_included": 2,
                "latency_ms": 30,
            },
        }

        return assembler

    @pytest.fixture
    def hybrid_retriever(
        self,
        mock_vector_retriever,
        mock_knowledge_graph,
        mock_cache_manager,
        mock_latency_tracker,
        mock_query_analyzer,
        mock_context_assembler,
    ):
        """Create a HybridRetriever instance with mocked dependencies."""
        return HybridRetriever(
            vector_retriever=mock_vector_retriever,
            knowledge_graph=mock_knowledge_graph,
            cache_manager=mock_cache_manager,
            latency_tracker=mock_latency_tracker,
            query_analyzer=mock_query_analyzer,
            context_assembler=mock_context_assembler,
            default_latency_budget_ms=200.0,
        )

    def test_initialization(self):
        """Test initialization of HybridRetriever."""
        # Test with default initialization (no components provided)
        retriever = HybridRetriever()

        # Verify component creation
        assert retriever.vector_retriever is not None
        assert retriever.knowledge_graph is not None
        assert retriever.cache is not None
        assert retriever.latency_tracker is not None
        assert retriever.query_analyzer is not None
        assert retriever.context_assembler is not None

        # Verify initialization state
        assert retriever.vector_retriever_initialized is False
        assert retriever.knowledge_graph_initialized is False
        assert retriever.default_latency_budget_ms == 200.0

    def test_initialize_vector_retriever(self, hybrid_retriever, mock_vector_retriever):
        """Test initializing the vector retriever."""
        # Test data
        documents = [
            {"id": "doc1", "type": "card", "text": "Sample card text"},
            {"id": "doc2", "type": "rule", "text": "Sample rule text"},
        ]

        # Initialize vector retriever
        result = hybrid_retriever.initialize_vector_retriever(documents)

        # Verify initialization
        assert result is True
        assert hybrid_retriever.vector_retriever_initialized is True
        mock_vector_retriever.index_documents.assert_called_once_with(documents)

    def test_initialize_vector_retriever_failure(
        self, hybrid_retriever, mock_vector_retriever
    ):
        """Test failure in initializing the vector retriever."""
        # Make index_documents raise an exception
        mock_vector_retriever.index_documents.side_effect = Exception("Indexing error")

        # Initialize vector retriever
        result = hybrid_retriever.initialize_vector_retriever([])

        # Verify initialization failure
        assert result is False
        assert hybrid_retriever.vector_retriever_initialized is False

    def test_build_knowledge_graph(self, hybrid_retriever, mock_knowledge_graph):
        """Test building the knowledge graph."""
        # Test data
        cards_data = [
            {"name": "Lightning Bolt", "text": "Deals 3 damage to any target."}
        ]
        rules_data = [{"id": "702.2", "text": "Deathtouch rule"}]
        glossary_data = {"Flying": "Flying ability description"}

        # Build knowledge graph
        result = hybrid_retriever.build_knowledge_graph(
            cards_data, rules_data, glossary_data
        )

        # Verify initialization
        assert result is True
        assert hybrid_retriever.knowledge_graph_initialized is True
        mock_knowledge_graph.build_graph_from_data.assert_called_once_with(
            cards_data, rules_data, glossary_data
        )

    def test_build_knowledge_graph_failure(
        self, hybrid_retriever, mock_knowledge_graph
    ):
        """Test failure in building the knowledge graph."""
        # Make build_graph_from_data raise an exception
        mock_knowledge_graph.build_graph_from_data.side_effect = Exception(
            "Graph error"
        )

        # Build knowledge graph
        result = hybrid_retriever.build_knowledge_graph([], [])

        # Verify initialization failure
        assert result is False
        assert hybrid_retriever.knowledge_graph_initialized is False

    def test_retrieve_with_cache_hit(
        self, hybrid_retriever, mock_cache_manager, mock_latency_tracker
    ):
        """Test retrieval with cache hit."""
        # Set up cache hit
        cached_results = [
            {"id": "cached1", "text": "Cached result 1", "type": "card"},
            {"id": "cached2", "text": "Cached result 2", "type": "rule"},
        ]
        mock_cache_manager.get.return_value = cached_results

        # Perform retrieval
        results = hybrid_retriever.retrieve("lightning bolt")

        # Verify cache was checked
        mock_cache_manager.get.assert_called_once()

        # Verify cache hit was recorded
        mock_latency_tracker.record.assert_any_call("cache_lookup", ANY)
        mock_latency_tracker.record.assert_any_call("total", ANY)
        mock_latency_tracker.record.assert_any_call("cached", ANY)

        # Verify we got back the cached results with metadata
        assert len(results) == 2
        assert results[0]["id"] == "cached1"
        assert results[0]["_metadata"]["cache_hit"] is True
        assert results[1]["id"] == "cached2"

    def test_retrieve_with_cache_miss(
        self,
        hybrid_retriever,
        mock_cache_manager,
        mock_latency_tracker,
        mock_query_analyzer,
    ):
        """Test retrieval with cache miss."""
        # Set up cache miss
        mock_cache_manager.get.return_value = None

        # Perform retrieval
        results = hybrid_retriever.retrieve("lightning bolt")

        # Verify cache miss workflow
        mock_cache_manager.get.assert_called_once()
        mock_query_analyzer.analyze_query.assert_called_once_with("lightning bolt")

        # Verify results were cached
        mock_cache_manager.set.assert_called_once()

        # Verify latency tracking
        mock_latency_tracker.record.assert_any_call("cache_lookup", ANY)
        mock_latency_tracker.record.assert_any_call("query_analysis", ANY)
        mock_latency_tracker.record.assert_any_call("vector", ANY)
        mock_latency_tracker.record.assert_any_call("graph", ANY)
        mock_latency_tracker.record.assert_any_call("merge", ANY)
        mock_latency_tracker.record.assert_any_call("total", ANY)

    def test_retrieve_vector_only(
        self,
        hybrid_retriever,
        mock_vector_retriever,
        mock_knowledge_graph,
        mock_query_analyzer,
    ):
        """Test retrieval with vector only (no graph traversal)."""
        # Configure query analysis to skip graph traversal
        mock_query_analyzer.analyze_query.return_value = {
            "query_type": "general",
            "entities": [],
            "requires_structured_knowledge": False,
            "relationship_types": [],
            "prioritize_graph": False,
        }

        # Perform retrieval
        results = hybrid_retriever.retrieve("general query")

        # Verify vector retrieval was called
        mock_vector_retriever.retrieve.assert_called_once()

        # Verify graph traversal was not called
        mock_knowledge_graph.query.assert_not_called()

        # Verify results
        assert len(results) == 2
        assert results[0]["id"] == "v1"
        assert results[1]["id"] == "v2"

    def test_retrieve_with_graph(
        self,
        hybrid_retriever,
        mock_vector_retriever,
        mock_knowledge_graph,
        mock_query_analyzer,
    ):
        """Test retrieval with graph traversal."""
        # Configure query analysis to require structured knowledge
        mock_query_analyzer.analyze_query.return_value = {
            "query_type": "card_lookup",
            "entities": [{"type": "card", "name": "Lightning Bolt"}],
            "requires_structured_knowledge": True,
            "relationship_types": ["card_uses_mechanic"],
            "prioritize_graph": True,
        }

        # Perform retrieval
        results = hybrid_retriever.retrieve("what does lightning bolt do?")

        # Verify both vector and graph were called
        mock_vector_retriever.retrieve.assert_called_once()
        mock_knowledge_graph.query.assert_called()

        # Verify results have metadata
        assert len(results) > 0
        assert "_metadata" in results[0]
        assert "strategy" in results[0]["_metadata"]

    def test_retrieve_and_assemble(
        self,
        hybrid_retriever,
        mock_vector_retriever,
        mock_knowledge_graph,
        mock_query_analyzer,
        mock_context_assembler,
    ):
        """Test combined retrieval and context assembly."""
        # Perform retrieval and assembly
        result = hybrid_retriever.retrieve_and_assemble(
            "lightning bolt", max_tokens=2000, latency_budget_ms=300.0
        )

        # Verify query analysis was called at least once
        assert mock_query_analyzer.analyze_query.call_count >= 1

        # Verify retrieval was called
        mock_vector_retriever.retrieve.assert_called_once()
        mock_knowledge_graph.query.assert_called()

        # Verify context assembly was called
        mock_context_assembler.assemble_context.assert_called_once()

        # Verify result structure
        assert "text" in result
        assert "documents" in result
        assert "token_count" in result
        assert "metrics" in result

        # Verify metrics
        assert "retrieval_time_ms" in result["metrics"]
        assert "total_time_ms" in result["metrics"]

    def test_merge_results(self, hybrid_retriever):
        """Test merging results from vector and graph retrieval."""
        # Test data
        vector_results = [
            {"id": "v1", "text": "Vector result 1", "type": "card", "score": 0.9},
            {"id": "v2", "text": "Vector result 2", "type": "rule", "score": 0.8},
            {"id": "v3", "text": "Vector result 3", "type": "card", "score": 0.7},
        ]

        graph_results = [
            {"id": "g1", "text": "Graph result 1", "type": "card", "score": 0.95},
            {"id": "g2", "text": "Graph result 2", "type": "glossary", "score": 0.85},
            {
                "id": "v2",
                "text": "Vector result 2",
                "type": "rule",
                "score": 0.8,
            },  # Duplicate
        ]

        query = "lightning bolt"
        query_analysis = {
            "query_type": "card_lookup",
            "entities": [{"type": "card", "name": "Lightning Bolt"}],
            "requires_structured_knowledge": True,
            "relationship_types": ["card_uses_mechanic"],
            "prioritize_graph": False,
        }

        # Test normal merging
        merged = hybrid_retriever._merge_results(
            vector_results, graph_results, query, query_analysis
        )

        # Verify deduplication (v2 appears in both results)
        assert len(merged) == 5  # Not 6 because of deduplication

        # Verify order (should have highest scores first)
        assert merged[0]["id"] in ["v1", "g1"]  # First results from both sources

        # Test prioritizing graph
        query_analysis["prioritize_graph"] = True
        merged_prioritized = hybrid_retriever._merge_results(
            vector_results, graph_results, query, query_analysis
        )

        # Verify graph results come first
        assert merged_prioritized[0]["id"] == "g1"

    def test_ensure_results_format(self, hybrid_retriever):
        """Test converting results to the standard format."""
        # Test with empty results
        assert hybrid_retriever._ensure_results_format([]) == []

        # Test with list of dicts (already correct format)
        dict_results = [{"id": "1", "text": "Test"}]
        assert hybrid_retriever._ensure_results_format(dict_results) == dict_results

        # Test with list of non-dicts
        str_results = ["result1", "result2"]
        formatted = hybrid_retriever._ensure_results_format(str_results)
        assert len(formatted) == 2
        assert formatted[0]["content"] == "result1"

        # Test with single dict
        single_dict = {"id": "1", "text": "Test"}
        assert hybrid_retriever._ensure_results_format(single_dict) == [single_dict]

        # Test with non-list, non-dict
        assert hybrid_retriever._ensure_results_format("string") == [
            {"content": "string"}
        ]

    def test_emergency_retrieval(self, hybrid_retriever, mock_vector_retriever):
        """Test emergency retrieval when budget is exceeded."""
        # Perform emergency retrieval
        results = hybrid_retriever._emergency_retrieval("lightning bolt")

        # Verify vector retrieval was called
        mock_vector_retriever.retrieve.assert_called_once()

        # Verify results
        assert len(results) == 2
        assert results[0]["id"] == "v1"

    def test_get_status(self, hybrid_retriever):
        """Test getting system status."""
        # Get status
        status = hybrid_retriever.get_status()

        # Verify status structure
        assert "vector_retriever" in status
        assert "knowledge_graph" in status
        assert "cache_metrics" in status
        assert "latency_metrics" in status
        assert "default_latency_budget_ms" in status

    def test_invalidate_cache(self, hybrid_retriever, mock_cache_manager):
        """Test cache invalidation."""
        # Invalidate cache
        count = hybrid_retriever.invalidate_cache(entity_type="card", entity_id="123")

        # Verify cache invalidation was called
        mock_cache_manager.invalidate.assert_called_once_with(
            entity_type="card", entity_id="123"
        )

        # Verify count
        assert count == 5  # From our mock return value

    def test_latency_budget_exceeded(
        self,
        hybrid_retriever,
        mock_vector_retriever,
        mock_query_analyzer,
        mock_latency_tracker,
    ):
        """Test handling of exceeded latency budget."""

        # Make query_analysis take too long
        def slow_analyze(*args, **kwargs):
            time.sleep(0.1)  # 100ms, exceeding our budget
            return {
                "query_type": "card_lookup",
                "entities": [{"type": "card", "name": "Lightning Bolt"}],
                "requires_structured_knowledge": True,
                "relationship_types": [],
                "prioritize_graph": False,
            }

        mock_query_analyzer.analyze_query.side_effect = slow_analyze

        # Perform retrieval with tiny budget
        results = hybrid_retriever.retrieve("lightning bolt", latency_budget_ms=10)

        # Verify emergency retrieval was triggered
        mock_vector_retriever.retrieve.assert_called_once()

        # Verify latency tracking
        mock_latency_tracker.record.assert_any_call("cache_lookup", ANY)
        mock_latency_tracker.record.assert_any_call("query_analysis", ANY)
        mock_latency_tracker.record.assert_any_call("total", ANY)
