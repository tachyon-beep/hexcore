# tests/knowledge/test_retriever.py

import sys
import pytest
import numpy as np
import os
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.knowledge.retriever import MTGRetriever, get_faiss_version


class TestMTGRetriever:
    """Test cases for the MTGRetriever class."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer for testing."""
        # Create mock encoder that returns fixed embeddings
        with patch("src.knowledge.retriever.SentenceTransformer") as mock_st:
            # Setup the mock embedder
            mock_embedder = MagicMock()

            # Make encode return predictable embeddings
            def mock_encode(**kwargs):
                sentences = kwargs.get("sentences", [])
                # Return a 384-dimensional embedding (common for sentence-transformers)
                if isinstance(sentences, list):
                    return np.array([[0.1] * 384] * len(sentences), dtype=np.float32)
                else:
                    return np.array([[0.1] * 384], dtype=np.float32)

            mock_embedder.encode.side_effect = mock_encode
            mock_st.return_value = mock_embedder
            yield mock_st

    @pytest.fixture
    def mock_faiss(self):
        """Create mock FAISS functionality for testing."""
        with patch("src.knowledge.retriever.faiss") as mock_faiss:
            # Setup mock index
            mock_index = MagicMock()
            mock_index.search.return_value = (
                np.array([[0.9, 0.8, 0.7]]),  # Distances
                np.array([[0, 1, 2]]),  # Indices
            )

            # Setup mock index creation
            mock_faiss.IndexFlatIP.return_value = mock_index

            # Setup mock read/write functions
            mock_faiss.write_index = MagicMock()
            mock_faiss.read_index = MagicMock(return_value=mock_index)

            yield mock_faiss

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "id": "card1",
                "type": "card",
                "text": "Lightning Bolt deals 3 damage to any target.",
            },
            {
                "id": "rule1",
                "type": "rule",
                "text": "702.2 Deathtouch is a static ability.",
            },
            {
                "id": "glossary1",
                "type": "glossary",
                "text": "Trample: A creature with trample can deal excess damage to the player.",
            },
        ]

    @pytest.fixture
    def retriever(self, mock_sentence_transformer, mock_faiss):
        """Create a MTGRetriever instance with mocked dependencies."""
        return MTGRetriever()

    def test_initialization(self, mock_sentence_transformer):
        """Test initialization of MTGRetriever."""
        # Test with default model
        retriever = MTGRetriever()
        mock_sentence_transformer.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Verify initial state
        assert retriever.document_index is None
        assert retriever.documents == []
        assert retriever.document_types == []
        assert retriever.document_ids == []

        # Test with custom model
        custom_model = "distilbert-base-nli-mean-tokens"
        with patch("src.knowledge.retriever.SentenceTransformer") as mock_st:
            retriever = MTGRetriever(embedding_model=custom_model)
            mock_st.assert_called_once_with(custom_model)

    def test_faiss_version_compatibility(self, retriever, sample_documents, mock_faiss):
        """Test FAISS version compatibility handling in actual use."""
        # This test verifies the real behavior we care about: how the code handles different FAISS versions

        # Test successful indexing with modern FAISS
        with patch("src.knowledge.retriever.get_faiss_version", return_value=(1, 7, 3)):
            # Index should work normally
            retriever.index_documents(sample_documents)
            mock_faiss.IndexFlatIP.assert_called_once()
            assert retriever.document_index.add.call_count == 1

        # Reset the mock
        mock_faiss.reset_mock()
        mock_faiss.IndexFlatIP.return_value.add.reset_mock()

        # Test with outdated FAISS version that raises an exception
        with patch("src.knowledge.retriever.get_faiss_version", return_value=(1, 5, 0)):
            # Make add raise an exception to simulate incompatibility
            mock_index = MagicMock()
            mock_index.add.side_effect = RuntimeError("FAISS error")
            mock_faiss.IndexFlatIP.return_value = mock_index

            # Should raise with helpful error message
            with pytest.raises(
                RuntimeError, match="Failed to add embeddings.*Consider upgrading FAISS"
            ):
                retriever.index_documents(sample_documents)

    def test_indexing_documents(self, retriever, sample_documents, mock_faiss):
        """Test indexing documents."""
        # Index documents
        retriever.index_documents(sample_documents)

        # Verify document extraction
        assert len(retriever.documents) == 3
        assert retriever.documents[0] == "Lightning Bolt deals 3 damage to any target."
        assert retriever.document_types == ["card", "rule", "glossary"]
        assert retriever.document_ids == ["card1", "rule1", "glossary1"]

        # Verify FAISS index was created
        mock_faiss.IndexFlatIP.assert_called_once_with(
            384
        )  # Our mock returns 384-dim vectors

        # Verify embeddings were added to index
        assert retriever.document_index.add.call_count == 1

    def test_indexing_empty_documents(self, retriever):
        """Test indexing with empty document list."""
        retriever.index_documents([])

        # Verify no processing occurred
        assert retriever.documents == []
        assert retriever.document_types == []
        assert retriever.document_ids == []
        assert retriever.document_index is None

    def test_retrieve(self, retriever, sample_documents):
        """Test document retrieval."""
        # First index documents
        retriever.index_documents(sample_documents)

        # Setup the search return value
        retriever.document_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # Distances
            np.array([[0, 1, 2]]),  # Indices
        )

        # Retrieve documents
        results = retriever.retrieve("lightning bolt", top_k=3)

        # Verify search was called correctly
        retriever.document_index.search.assert_called_once()

        # Verify results
        assert len(results) == 3

        # Check first result
        assert results[0]["id"] == "card1"
        assert results[0]["type"] == "card"
        assert results[0]["text"] == "Lightning Bolt deals 3 damage to any target."
        assert results[0]["score"] == 0.9

    def test_retrieve_with_type_filter(self, retriever, sample_documents):
        """Test retrieval with document type filter."""
        # First index documents
        retriever.index_documents(sample_documents)

        # Setup the search return value
        retriever.document_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # Distances
            np.array([[0, 1, 2]]),  # Indices
        )

        # Retrieve only cards
        results = retriever.retrieve("lightning bolt", top_k=3, doc_type="card")

        # Verify filtered results
        assert len(results) == 1
        assert results[0]["type"] == "card"
        assert results[0]["id"] == "card1"

    def test_retrieve_error_handling(self, retriever):
        """Test error handling during retrieval."""
        # Attempt to retrieve without indexing
        with pytest.raises(ValueError, match="Document index not initialized"):
            retriever.retrieve("test query")

    def test_retrieve_by_categories(self, retriever, sample_documents):
        """Test category-based retrieval."""
        # First index documents
        retriever.index_documents(sample_documents)

        # Setup the search return value
        retriever.document_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]]),  # Distances
            np.array([[0, 1, 2]]),  # Indices
        )

        # Retrieve by categories
        results = retriever.retrieve_by_categories("lightning bolt", top_k_per_type=1)

        # Verify search was called
        retriever.document_index.search.assert_called_once()

        # Verify results structure
        assert set(results.keys()) == {"card", "rule", "glossary"}

        # Check first category
        assert len(results["card"]) == 1
        assert results["card"][0]["id"] == "card1"
        assert results["card"][0]["score"] == 0.9

        # Check second category
        assert len(results["rule"]) == 1
        assert results["rule"][0]["id"] == "rule1"
        assert results["rule"][0]["score"] == 0.8

    def test_save_and_load_index(self, retriever, sample_documents, mock_faiss):
        """Test saving and loading the index."""
        # First index documents
        retriever.index_documents(sample_documents)

        # Save index to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".tmp") as temp:
            file_path = temp.name

            # Save the index
            retriever.save_index(file_path)

            # Verify faiss.write_index was called
            mock_faiss.write_index.assert_called_once_with(ANY, f"{file_path}.index")

            # Verify metadata was saved
            assert os.path.exists(f"{file_path}.metadata.json")

            # Create new retriever for loading
            new_retriever = MTGRetriever()

            # Load the index
            new_retriever.load_index(file_path)

            # Verify faiss.read_index was called
            mock_faiss.read_index.assert_called_once_with(f"{file_path}.index")

            # Test that metadata was loaded
            assert new_retriever.documents == retriever.documents
            assert new_retriever.document_types == retriever.document_types
            assert new_retriever.document_ids == retriever.document_ids

            # Clean up metadata file
            if os.path.exists(f"{file_path}.metadata.json"):
                os.remove(f"{file_path}.metadata.json")

    def test_get_index(self, retriever, sample_documents):
        """Test getting indexed document IDs."""
        # Without indexing
        assert retriever.get_index() == []

        # After indexing
        retriever.index_documents(sample_documents)
        assert retriever.get_index() == ["card1", "rule1", "glossary1"]

    def test_indexing_failure(self, retriever, sample_documents, mock_faiss):
        """Test handling indexing failures."""
        # We need to set up retriever with document_index first,
        # since the index is created during the indexing process

        # Create a mock index that raises an exception on add
        mock_index = MagicMock()
        mock_index.add.side_effect = RuntimeError("FAISS error")
        mock_faiss.IndexFlatIP.return_value = mock_index

        # Patch the version to show as outdated
        with patch("src.knowledge.retriever.get_faiss_version", return_value=(1, 5, 0)):
            # Attempt to index, should re-raise with additional context
            with pytest.raises(
                RuntimeError, match="Failed to add embeddings to FAISS index"
            ):
                retriever.index_documents(sample_documents)
