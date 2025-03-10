# src/knowledge/retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import re
import importlib.metadata
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


def get_faiss_version() -> Tuple[int, int, int]:
    """
    Get the installed FAISS version as a tuple of (major, minor, patch).

    Returns:
        Tuple containing the version components.
    """
    try:
        # Get version string using importlib.metadata instead of deprecated pkg_resources
        version_str = None

        try:
            version_str = importlib.metadata.version("faiss-cpu")
        except (
            ImportError
        ):  # ImportError is a parent class that includes PackageNotFoundError
            try:
                # If faiss-cpu is not found, try faiss-gpu
                version_str = importlib.metadata.version("faiss-gpu")
            except (
                ImportError
            ) as e:  # ImportError is a parent class that includes PackageNotFoundError
                logger.warning(f"Could not determine FAISS version: {str(e)}")
                return (1, 7, 0)  # Default fallback version

        if not version_str:
            return (1, 7, 0)  # Default fallback version if we couldn't get a version

        # Parse version string - handle different formats
        # Could be something like '1.7.2' or '1.7.2.post1'
        match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
        if match:
            major, minor, patch = map(int, match.groups())
            return (major, minor, patch)
        return (1, 7, 0)  # Default fallback version
    except Exception as e:
        logger.warning(f"Could not determine FAISS version: {str(e)}")
        return (1, 7, 0)  # Default fallback version


class MTGRetriever:
    """
    Retrieval-Augmented Generation system for MTG-related text.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the MTG retriever.

        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.document_index = None
        self.documents = []
        self.document_types = []
        self.document_ids = []

        # Check FAISS version for compatibility
        self.faiss_version = get_faiss_version()
        logger.info(
            f"Initialized MTG Retriever with {embedding_model}, FAISS version {'.'.join(map(str, self.faiss_version))}"
        )

    def get_index(self) -> List[str]:
        """
        Get the indexed documents.

        Returns:
            List of document IDs in the index
        """
        if not self.document_ids:
            logger.warning("No documents have been indexed")
            return []

        return self.document_ids

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index documents for retrieval.

        Args:
            documents: List of dictionaries containing document text and metadata
                Each document should have 'text', 'type', and 'id' fields
        """
        if not documents:
            logger.warning("No documents provided for indexing")
            return

        # Extract text, type, and ID from documents
        texts = [doc["text"] for doc in documents]
        self.document_types = [doc["type"] for doc in documents]
        self.document_ids = [doc["id"] for doc in documents]
        self.documents = texts

        # Create embeddings
        logger.info(f"Creating embeddings for {len(texts)} documents")
        embeddings = self.embedding_model.encode(
            sentences=texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        # Convert to numpy array with the right dtype
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        # We handle it ourselves to avoid FAISS version inconsistencies
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # To avoid division by zero
        embeddings_np = embeddings_np / norms

        # Create FAISS index
        vector_dimension = embeddings_np.shape[1]
        self.document_index = faiss.IndexFlatIP(vector_dimension)

        # Add embeddings to index with FAISS version compatibility handling
        try:
            # The modern API (1.7.0+) expects a simple array
            self.document_index.add(
                x=embeddings_np
            )  # type: ignore  # Suppress parameter name checking
            logger.info("Added embeddings using standard FAISS method")
        except Exception as e:
            # Log the error with useful version information
            logger.error(
                f"Error adding embeddings to FAISS index (version {'.'.join(map(str, self.faiss_version))}): {str(e)}"
            )
            logger.warning(
                "Your FAISS version may be incompatible with this code. "
                "Consider upgrading to FAISS 1.7.0 or later."
            )
            # Re-raise with more context
            raise RuntimeError(
                f"Failed to add embeddings to FAISS index. FAISS version: {'.'.join(map(str, self.faiss_version))}. "
                f"Error: {str(e)}. Consider upgrading FAISS to 1.7.0 or later."
            )

        logger.info(f"Indexed {len(texts)} documents with dimension {vector_dimension}")

    def retrieve(
        self, query: str, top_k: int = 5, doc_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            doc_type: Optional filter for document type

        Returns:
            List of retrieved documents with relevance scores
        """
        if self.document_index is None:
            raise ValueError(
                "Document index not initialized. Call index_documents first."
            )

        # Create query embedding - explicitly specify all parameters to avoid errors
        query_embedding = self.embedding_model.encode(
            sentences=[query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        # Convert to numpy array with the right dtype
        query_np = np.array(query_embedding, dtype=np.float32)

        # Normalize for cosine similarity (do it ourselves instead of using faiss.normalize_L2)
        norms = np.linalg.norm(query_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # To avoid division by zero
        query_np = query_np / norms

        # In FAISS 1.9.0, search directly returns distances and indices
        distances, labels = self.document_index.search(query_np, top_k)  # type: ignore

        # Process results
        results = []

        for i in range(labels.shape[1]):
            idx = labels[0, i]

            # Skip invalid indices
            if idx < 0 or idx >= len(self.documents):
                continue

            # Create document result
            doc = {
                "text": self.documents[idx],
                "type": self.document_types[idx],
                "id": self.document_ids[idx],
                "score": float(distances[0, i]),
            }

            # Apply document type filter if specified
            if doc_type is None or doc["type"] == doc_type:
                results.append(doc)

                # Break if we have enough results
                if len(results) >= top_k:
                    break

        return results

    def retrieve_by_categories(
        self, query: str, top_k_per_type: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents by category, retrieving top_k for each document type.

        Args:
            query: Query text
            top_k_per_type: Number of documents to retrieve per type

        Returns:
            Dictionary mapping document types to retrieved documents
        """
        if self.document_index is None:
            raise ValueError(
                "Document index not initialized. Call index_documents first."
            )

        if not self.documents:
            logger.warning("No documents in the index")
            return {}

        # Get all unique document types
        unique_types = set(self.document_types)

        # Create query embedding - explicitly specify all parameters to avoid errors
        query_embedding = self.embedding_model.encode(
            sentences=[query],
            batch_size=1,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )

        # Convert to numpy array with the right dtype
        query_np = np.array(query_embedding, dtype=np.float32)

        # Normalize for cosine similarity (do it ourselves instead of using faiss.normalize_L2)
        norms = np.linalg.norm(query_np, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # To avoid division by zero
        query_np = query_np / norms

        # Determine how many results to retrieve
        search_k = top_k_per_type * len(unique_types)
        search_k = min(search_k, len(self.documents))

        # In FAISS 1.9.0, search directly returns distances and indices
        distances, labels = self.document_index.search(query_np, search_k)  # type: ignore

        # Group results by document type
        results_by_type = {doc_type: [] for doc_type in unique_types}

        # Process search results
        for i in range(labels.shape[1]):
            idx = labels[0, i]

            # Skip invalid indices
            if idx < 0 or idx >= len(self.documents):
                continue

            doc_type = self.document_types[idx]

            # Only add if we need more of this type
            if len(results_by_type[doc_type]) < top_k_per_type:
                doc = {
                    "text": self.documents[idx],
                    "type": doc_type,
                    "id": self.document_ids[idx],
                    "score": float(distances[0, i]),
                }
                results_by_type[doc_type].append(doc)

        return results_by_type

    def save_index(self, path):
        """Save the FAISS index and document data to disk."""
        if self.document_index is None:
            raise ValueError("No index to save")
        faiss.write_index(self.document_index, f"{path}.index")
        with open(f"{path}.metadata.json", "w") as f:
            json.dump(
                {
                    "documents": self.documents,
                    "document_types": self.document_types,
                    "document_ids": self.document_ids,
                },
                f,
            )

    def load_index(self, path):
        """Load the FAISS index and document data from disk."""
        self.document_index = faiss.read_index(f"{path}.index")
        with open(f"{path}.metadata.json", "r") as f:
            data = json.load(f)
            self.documents = data["documents"]
            self.document_types = data["document_types"]
            self.document_ids = data["document_ids"]
