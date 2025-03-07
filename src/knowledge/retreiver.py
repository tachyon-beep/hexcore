# src/knowledge/retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


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

        logger.info(f"Initialized MTG Retriever with {embedding_model}")

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
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Convert to numpy array with the right dtype
        embeddings_np = np.array(embeddings, dtype=np.float32)

        # Create a copy before normalizing (since normalize_L2 modifies in-place)
        embeddings_norm = embeddings_np.copy()

        # Normalize for cosine similarity - provide the vector array to normalize_L2
        faiss.normalize_L2(embeddings_norm)

        # Create FAISS index
        vector_dimension = embeddings_norm.shape[1]
        self.document_index = faiss.IndexFlatIP(vector_dimension)

        # Add the embeddings to the index with both required parameters
        # n = number of vectors, x = the vector data
        n_vectors = embeddings_norm.shape[0]
        self.document_index.add(n_vectors, embeddings_norm)

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

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])

        # Convert to numpy array with the right dtype
        query_np = np.array(query_embedding, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_np)

        # Create arrays to store results
        n_queries = query_np.shape[0]  # Number of query vectors (usually 1)
        distances = np.empty((n_queries, top_k), dtype=np.float32)
        labels = np.empty((n_queries, top_k), dtype=np.int64)

        # Parameters for search
        search_params = faiss.SearchParameters()

        # Perform the search with all five required parameters
        self.document_index.search(
            query_np,  # x: query vectors
            top_k,  # k: number of results
            distances,  # distances: output array for distances
            labels,  # labels: output array for indices
            search_params,  # search_params: additional parameters
        )

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

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])

        # Convert to numpy array with the right dtype
        query_np = np.array(query_embedding, dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_np)

        # Determine how many results to retrieve
        search_k = top_k_per_type * len(unique_types)
        search_k = min(search_k, len(self.documents))

        # Create arrays to store results
        n_queries = query_np.shape[0]  # Number of query vectors (usually 1)
        distances = np.empty((n_queries, search_k), dtype=np.float32)
        labels = np.empty((n_queries, search_k), dtype=np.int64)

        # Parameters for search
        search_params = faiss.SearchParameters()

        # Perform the search with all five required parameters
        self.document_index.search(
            query_np,  # x: query vectors
            search_k,  # k: number of results
            distances,  # distances: output array for distances
            labels,  # labels: output array for indices
            search_params,  # search_params: additional parameters
        )

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
