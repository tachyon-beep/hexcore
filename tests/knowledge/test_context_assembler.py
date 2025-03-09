# tests/knowledge/test_context_assembler.py

import sys
import pytest
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.knowledge.context_assembler import ContextAssembler


class TestContextAssembler:
    """Test cases for the ContextAssembler class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = MagicMock()

        # Mock encode and decode methods
        tokenizer.encode.side_effect = lambda text: [0] * (
            len(text) // 4
        )  # Roughly 4 chars per token
        tokenizer.decode.side_effect = lambda tokens: "x" * (
            len(tokens) * 4
        )  # Reverse of encode

        return tokenizer

    @pytest.fixture
    def mock_latency_tracker(self):
        """Create a mock latency tracker for testing."""
        tracker = MagicMock()
        tracker.record.return_value = None
        return tracker

    @pytest.fixture
    def assembler(self, mock_tokenizer, mock_latency_tracker):
        """Create a ContextAssembler instance for testing."""
        return ContextAssembler(
            tokenizer=mock_tokenizer,
            latency_tracker=mock_latency_tracker,
            max_context_tokens=1000,
            default_latency_budget_ms=50.0,
        )

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "id": "doc1",
                "type": "card",
                "text": "Lightning Bolt is an instant that deals 3 damage to any target for just one red mana.",
                "score": 0.9,
            },
            {
                "id": "doc2",
                "type": "rule",
                "text": "702.2 Deathtouch is a static ability that causes damage dealt by an object to be especially effective.",
                "score": 0.7,
            },
            {
                "id": "doc3",
                "type": "glossary",
                "text": "Trample is a keyword ability that allows attacking creatures to deal excess damage to the defending player or planeswalker.",
                "score": 0.6,
            },
            {
                "id": "doc4",
                "type": "card",
                "text": "Counterspell is an instant that counters target spell. It requires two blue mana.",
                "score": 0.5,
            },
            {
                "id": "doc5",
                "type": "rule",
                "text": "A very long rule text "
                + "that exceeds the token limit." * 50,  # Make this artificially long
                "score": 0.4,
            },
        ]

    def test_initialization(self, mock_tokenizer, mock_latency_tracker):
        """Test initialization of ContextAssembler."""
        max_tokens = 2000
        latency_budget = 100.0

        assembler = ContextAssembler(
            tokenizer=mock_tokenizer,
            latency_tracker=mock_latency_tracker,
            max_context_tokens=max_tokens,
            default_latency_budget_ms=latency_budget,
        )

        # Verify initialization
        assert assembler.tokenizer == mock_tokenizer
        assert assembler.latency_tracker == mock_latency_tracker
        assert assembler.max_context_tokens == max_tokens
        assert assembler.default_latency_budget_ms == pytest.approx(latency_budget)

    def test_empty_documents(self, assembler):
        """Test handling of empty document list."""
        query = "What does Lightning Bolt do?"
        result = assembler.assemble_context(query, [])

        # Verify empty context result
        assert "No relevant information found" in result["text"]
        assert result["documents"] == []
        assert result["token_count"] == 0
        assert result["metrics"]["doc_count"] == 0
        assert result["metrics"]["docs_included"] == 0

    def test_basic_context_assembly(self, assembler, sample_documents):
        """Test basic context assembly with documents under token limit."""
        query = "What does Lightning Bolt do?"
        result = assembler.assemble_context(
            query, sample_documents[:3]
        )  # First 3 docs should fit

        # Verify context assembly result
        assert "[CARD]" in result["text"]  # Should include document types
        assert "Lightning Bolt" in result["text"]  # Should include content
        assert "doc1" in result["documents"]
        assert len(result["documents"]) == 3  # All 3 docs should be included
        assert result["token_count"] > 0
        assert result["metrics"]["docs_included"] == 3
        assert result["metrics"]["compression_applied"] is False
        assert result["metrics"]["truncation_applied"] is False

    def test_context_assembly_with_token_limit(self, assembler, sample_documents):
        """Test context assembly with token limit constraint."""
        # Set a very small token limit that only allows 1-2 documents
        query = "What does Lightning Bolt do?"
        result = assembler.assemble_context(query, sample_documents, max_tokens=50)

        # Verify token limit is respected
        assert result["token_count"] <= 50
        assert len(result["documents"]) < len(sample_documents)
        assert result["metrics"]["docs_included"] < len(sample_documents)

    def test_document_scoring(self, assembler, sample_documents):
        """Test document scoring functionality."""
        query = "How do Lightning Bolt and Counterspell interact?"
        query_analysis = {
            "query_type": "complex_interaction",
            "entities": [
                {"type": "card", "name": "Lightning Bolt"},
                {"type": "card", "name": "Counterspell"},
            ],
        }

        # Patch the _score_document_relevance method to spy on its calls
        with patch.object(
            assembler,
            "_score_document_relevance",
            wraps=assembler._score_document_relevance,
        ) as mock_score:
            result = assembler.assemble_context(
                query, sample_documents, query_analysis=query_analysis
            )

            # Verify scoring was called with correct arguments
            mock_score.assert_called_once()
            args, kwargs = mock_score.call_args
            assert args[0] == query
            assert args[1] == sample_documents
            assert kwargs["query_analysis"] == query_analysis

            # Verify cards mentioned in the query are included in the result
            assert "Lightning Bolt" in result["text"]
            assert "Counterspell" in result["text"]

            # Both Lightning Bolt and Counterspell documents should be included
            assert "doc1" in result["documents"]
            assert "doc4" in result["documents"]

    def test_document_compression(self, assembler):
        """Test document compression functionality."""
        query = "What is Deathtouch?"

        # Create a single long document that needs compression
        long_doc = {
            "id": "long1",
            "type": "rule",
            "text": "Deathtouch is a keyword ability. "
            + "It has many rules and implications. " * 50
            + "A creature with deathtouch destroys any creature it damages.",
        }

        # Patch the internal compress method to verify it's called
        with patch.object(
            assembler, "_compress_document", wraps=assembler._compress_document
        ) as mock_compress:
            # Force compression by setting a low token limit
            result = assembler.assemble_context(query, [long_doc], max_tokens=200)

            # Verify compression was called
            mock_compress.assert_called_once()

            # Verify the result indicates compression
            assert result["metrics"]["compression_applied"] is True
            assert "SUMMARY" in result["text"]  # Compression marker

    def test_document_truncation(self, assembler):
        """Test document truncation functionality."""
        query = "What is Deathtouch?"

        # Create a single very long document that will need truncation
        very_long_doc = {
            "id": "verylong1",
            "type": "rule",
            "text": "Deathtouch rules: " + "Very long explanation. " * 100,
        }

        # Patch both compress and truncate to see which is called
        with patch.object(
            assembler, "_compress_document", return_value="Still too long" * 50
        ) as mock_compress:
            with patch.object(
                assembler, "_truncate_document", wraps=assembler._truncate_document
            ) as mock_truncate:
                # Force truncation by setting a low token limit and ensuring compression still exceeds it
                result = assembler.assemble_context(
                    query, [very_long_doc], max_tokens=100
                )

                # Verify both compression and truncation were attempted
                mock_compress.assert_called_once()
                mock_truncate.assert_called_once()

                # Verify the result indicates truncation
                assert result["metrics"]["truncation_applied"] is True
                assert "TRUNCATED" in result["text"]  # Truncation marker

    def test_latency_budget_respect(self, assembler, sample_documents):
        """Test respect for latency budget."""
        query = "What does Lightning Bolt do?"

        # Instead of delaying the scoring process which may be skipped,
        # let's patch a method that's called early in the process to trigger
        # the budget exceeded flag but also ensure a document is returned
        original_time_now = time.time

        call_count = 0

        def delayed_time_mock():
            nonlocal call_count
            # First call - return normal time
            # Second call - add delay to simulate elapsed time
            if call_count == 0:
                call_count += 1
                return original_time_now()
            else:
                return original_time_now() + 0.010  # 10ms delay

        with patch("time.time", side_effect=delayed_time_mock):
            # Use a tiny budget which will be exceeded after the delay
            result = assembler.assemble_context(
                query, sample_documents, latency_budget_ms=1.0
            )

            # Verify budget exceeded is reported
            assert result["metrics"]["budget_exceeded"] is True

            # Should still include at least one document
            assert len(result["documents"]) >= 1
            assert result["token_count"] > 0

    def test_token_estimation(self, assembler, mock_tokenizer):
        """Test token estimation functionality."""
        text = "This is a test document."

        # Verify tokenizer is used when available
        _ = assembler._estimate_tokens(text)
        mock_tokenizer.encode.assert_called_with(text)

        # Test with empty text
        assert assembler._estimate_tokens("") == 0

        # Create assembler without tokenizer
        assembler_no_tokenizer = ContextAssembler()

        # Verify character-based estimation is used as fallback
        fallback_count = assembler_no_tokenizer._estimate_tokens(text)
        assert fallback_count == len(text) // 4 + 1

    def test_keyword_extraction(self, assembler):
        """Test keyword extraction for document matching."""
        query = "How does Flying work with Deathtouch in multiple combat scenarios?"

        # Test with query analysis
        query_analysis = {
            "entities": [
                {"type": "mechanic", "name": "Flying"},
                {"type": "mechanic", "name": "Deathtouch"},
            ]
        }

        keywords = assembler._extract_keywords(query, query_analysis)

        # Should extract entities from query analysis
        assert "Flying" in keywords
        assert "Deathtouch" in keywords

        # Test without query analysis
        basic_keywords = assembler._extract_keywords(query, None)

        # Should filter out stop words and short words
        assert "how" not in basic_keywords
        assert "does" not in basic_keywords
        assert "with" not in basic_keywords

        # Should keep meaningful words
        assert "Flying" in basic_keywords
        assert "Deathtouch" in basic_keywords
        assert "combat" in basic_keywords
        assert "multiple" in basic_keywords

    def test_type_match_scoring(self, assembler):
        """Test document type matching with query type."""
        # Test card lookup query with different document types
        assert assembler._calculate_type_match_score(
            "card_lookup", "card"
        ) == pytest.approx(1.0)
        assert assembler._calculate_type_match_score(
            "card_lookup", "rule"
        ) == pytest.approx(0.5)
        assert assembler._calculate_type_match_score(
            "card_lookup", "glossary"
        ) == pytest.approx(0.7)

        # Test rule lookup query with different document types
        assert assembler._calculate_type_match_score(
            "rule_lookup", "rule"
        ) == pytest.approx(1.0)
        assert assembler._calculate_type_match_score(
            "rule_lookup", "card"
        ) == pytest.approx(0.3)

        # Test unknown document type
        assert assembler._calculate_type_match_score(
            "card_lookup", "unknown_type"
        ) == pytest.approx(0.5)

        # Test unknown query type (should use general defaults)
        assert assembler._calculate_type_match_score(
            "unknown_query_type", "card"
        ) == pytest.approx(0.7)
