# src/knowledge/context_assembler.py
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Intelligent context assembly for knowledge retrieval results.

    This component selects, ranks, formats and assembles retrieved knowledge
    to create an optimal context for language model generation. It handles:
    - Priority-based selection of retrieved documents
    - Truncation and compression for long documents
    - Token budget management
    - Latency-aware assembly
    """

    def __init__(
        self,
        tokenizer=None,
        latency_tracker=None,
        max_context_tokens: int = 3072,
        default_latency_budget_ms: float = 50.0,
    ):
        """
        Initialize the context assembler.

        Args:
            tokenizer: Optional tokenizer for token counting
            latency_tracker: Optional latency tracker for performance monitoring
            max_context_tokens: Maximum tokens in assembled context
            default_latency_budget_ms: Default budget for context assembly in ms
        """
        self.tokenizer = tokenizer
        self.latency_tracker = latency_tracker
        self.max_context_tokens = max_context_tokens
        self.default_latency_budget_ms = default_latency_budget_ms

        logger.info(
            f"Initialized context assembler (max_tokens={max_context_tokens}, "
            f"latency_budget={default_latency_budget_ms}ms)"
        )

    def assemble_context(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        latency_budget_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Assemble context from retrieved documents based on query needs.

        Args:
            query: User query text
            retrieved_docs: List of retrieved documents
            query_analysis: Optional query analysis results
            max_tokens: Maximum tokens to include in context
            latency_budget_ms: Maximum time to spend on context assembly in ms

        Returns:
            Dictionary containing:
            - text: Assembled context text
            - documents: List of document metadata included in context
            - token_count: Approximate token count of assembled context
            - metrics: Performance and selection metrics
        """
        start_time = time.time()

        # Set default values if not provided
        token_limit = max_tokens or self.max_context_tokens
        latency_budget = latency_budget_ms or self.default_latency_budget_ms

        # Initialize metrics
        metrics = {
            "doc_count": len(retrieved_docs),
            "docs_included": 0,
            "doc_types": {},
            "token_count": 0,
            "compression_applied": False,
            "truncation_applied": False,
            "latency_ms": 0,
            "budget_exceeded": False,
        }

        # If no documents, return empty context
        if not retrieved_docs:
            result = {
                "text": "No relevant information found for this query.",
                "documents": [],
                "token_count": 0,
                "metrics": metrics,
            }

            # Record metrics if latency tracker available
            if self.latency_tracker:
                total_time_ms = (time.time() - start_time) * 1000
                self.latency_tracker.record("context_assembly", total_time_ms)

            return result

        # Time checkpoint after initialization
        init_time = time.time()
        init_time_ms = (init_time - start_time) * 1000

        # Check latency budget
        remaining_budget_ms = latency_budget - init_time_ms
        if remaining_budget_ms <= 0:
            metrics["budget_exceeded"] = True
            logger.warning(
                "Latency budget exceeded during context assembly initialization"
            )
            # Return with top document only
            doc = retrieved_docs[0]
            doc_text = doc.get("text", "")
            doc_id = doc.get("id", "unknown")

            # Update metrics to track included document
            metrics["docs_included"] = 1

            result = {
                "text": f"[{doc.get('type', 'DOCUMENT').upper()}]\n{doc_text}",
                "documents": [doc_id],  # Make sure we include the document ID
                "token_count": self._estimate_tokens(doc_text),
                "metrics": metrics,
            }

            # Record metrics if latency tracker available
            if self.latency_tracker:
                total_time_ms = (time.time() - start_time) * 1000
                self.latency_tracker.record("context_assembly", total_time_ms)

            return result

        # Score documents based on relevance to query
        scoring_start = time.time()
        try:
            doc_scores = self._score_document_relevance(
                query,
                retrieved_docs,
                query_analysis=query_analysis,
                timeout_ms=min(
                    remaining_budget_ms * 0.3, 20
                ),  # Allocate part of budget
            )
        except TimeoutError:
            # Fallback to simpler scoring
            logger.warning("Document scoring timed out, using original order")
            doc_scores = {i: 1.0 - i * 0.01 for i in range(len(retrieved_docs))}
        scoring_time_ms = (time.time() - scoring_start) * 1000

        # Update remaining budget
        remaining_budget_ms -= scoring_time_ms

        # Sort documents by relevance score
        scored_docs = [
            (doc, doc_scores.get(i, 0.0)) for i, doc in enumerate(retrieved_docs)
        ]
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        # Assemble context by adding documents in priority order until token limit
        included_docs = []
        included_doc_ids = []
        assembled_text_parts = []
        token_count = 0
        skipped_high_value = False

        for doc, score in sorted_docs:
            # Check if we have time to process this document
            if time.time() - start_time > latency_budget / 1000:
                metrics["budget_exceeded"] = True
                logger.debug("Latency budget exceeded during document processing")
                break

            # Extract document text and type
            doc_text = doc.get("text", "")
            doc_type = doc.get("type", "DOCUMENT").upper()
            doc_id = doc.get("id", "unknown")

            # Track document types for metrics
            if doc_type not in metrics["doc_types"]:
                metrics["doc_types"][doc_type] = 0
            metrics["doc_types"][doc_type] += 1

            # Skip if this document is already included
            if doc_id in included_doc_ids:
                continue

            # Estimate tokens in this document
            doc_tokens = self._estimate_tokens(doc_text)

            # Check if adding would exceed token limit
            if token_count + doc_tokens <= token_limit:
                # Can add without modifications
                assembled_text_parts.append(f"[{doc_type}]\n{doc_text}")
                included_docs.append(doc)
                included_doc_ids.append(doc_id)
                token_count += doc_tokens

            elif score > 0.7 and doc_tokens > 100:  # High relevance but too long
                # Try compression first
                compressed_text = self._compress_document(
                    doc_text, query, query_analysis=query_analysis
                )
                compressed_tokens = self._estimate_tokens(compressed_text)

                if token_count + compressed_tokens <= token_limit:
                    # Add compressed version
                    assembled_text_parts.append(
                        f"[{doc_type} - SUMMARY]\n{compressed_text}"
                    )
                    included_docs.append(doc)
                    included_doc_ids.append(doc_id)
                    token_count += compressed_tokens
                    metrics["compression_applied"] = True
                else:
                    # Try truncation as last resort for high-value content
                    truncated_text = self._truncate_document(
                        doc_text, token_limit - token_count
                    )
                    truncated_tokens = self._estimate_tokens(truncated_text)

                    if (
                        truncated_tokens > 50
                    ):  # Only include if meaningful content remains
                        assembled_text_parts.append(
                            f"[{doc_type} - TRUNCATED]\n{truncated_text}"
                        )
                        included_docs.append(doc)
                        included_doc_ids.append(doc_id)
                        token_count += truncated_tokens
                        metrics["truncation_applied"] = True
                    else:
                        skipped_high_value = True
            else:
                # Document too long and not high enough value to process further
                continue

            # Stop if we've included enough documents
            if len(included_docs) >= 10:
                break

        # Build final context
        assembled_text = "\n\n".join(assembled_text_parts)

        # Update metrics
        metrics["docs_included"] = len(included_docs)
        metrics["token_count"] = token_count
        metrics["skipped_high_value"] = skipped_high_value
        metrics["latency_ms"] = (time.time() - start_time) * 1000

        # Record latency if tracker available
        if self.latency_tracker:
            self.latency_tracker.record("context_assembly", metrics["latency_ms"])

        return {
            "text": assembled_text,
            "documents": included_doc_ids,
            "token_count": token_count,
            "metrics": metrics,
        }

    def _score_document_relevance(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        query_analysis: Optional[Dict[str, Any]] = None,
        timeout_ms: float = 20.0,
    ) -> Dict[int, float]:
        """
        Score document relevance to the query.

        Args:
            query: Query text
            documents: List of documents to score
            query_analysis: Optional query analysis results
            timeout_ms: Maximum time to spend scoring in ms

        Returns:
            Dictionary mapping document indices to relevance scores (0.0-1.0)
        """
        start_time = time.time()
        deadline = start_time + (timeout_ms / 1000)

        # Initialize result with default scores
        default_scores = {}
        for i, doc in enumerate(documents):
            # By default, use the original order but slightly decreasing scores
            default_scores[i] = max(0.0, 1.0 - (i * 0.01))

        # If no specific information to enhance scoring, return default scores
        if not query_analysis:
            return default_scores

        scores = {}

        # Extract entities mentioned in the query
        query_entities = query_analysis.get("entities", [])
        query_type = query_analysis.get("query_type", "general")

        # Get entity names for matching
        entity_names = [entity["name"].lower() for entity in query_entities]
        entity_types = [entity["type"] for entity in query_entities]

        # Process each document
        for i, doc in enumerate(documents):
            # Check for timeout
            if time.time() > deadline:
                logger.warning("Document scoring timed out, returning partial results")
                # Return scores calculated so far, filling missing with defaults
                for j in range(len(documents)):
                    if j not in scores:
                        scores[j] = default_scores[j]
                return scores

            # Start with the default score
            base_score = default_scores[i]

            # Get document text and type
            doc_text = doc.get("text", "").lower()
            doc_type = doc.get("type", "").lower()

            # Initial score is based on document type match with query type
            type_match_score = self._calculate_type_match_score(query_type, doc_type)

            # Score based on entity mentions
            entity_match_score = 0.0
            for entity_name in entity_names:
                if entity_name in doc_text:
                    entity_match_score += 0.2  # Each entity mention adds to the score

            # Score based on specific query terms
            # Split query into terms and check for matches
            query_terms = query.lower().split()
            term_match_count = 0

            for term in query_terms:
                if len(term) >= 4 and term in doc_text:  # Only check meaningful terms
                    term_match_count += 1

            term_match_score = min(0.5, term_match_count * 0.05)  # Cap at 0.5

            # Calculate final score - weighted combination of factors
            final_score = (
                0.4 * base_score
                + 0.3 * type_match_score
                + 0.2 * entity_match_score
                + 0.1 * term_match_score
            )

            # Normalize and store
            scores[i] = min(1.0, max(0.0, final_score))

        return scores

    def _calculate_type_match_score(self, query_type: str, doc_type: str) -> float:
        """
        Calculate how well a document type matches the query type.

        Args:
            query_type: Type of query
            doc_type: Type of document

        Returns:
            Match score from 0.0 to 1.0
        """
        # Define type match preferences
        type_matches = {
            "card_lookup": {"card": 1.0, "rule": 0.5, "glossary": 0.7},
            "rule_lookup": {"rule": 1.0, "glossary": 0.6, "card": 0.3},
            "mechanic_lookup": {"glossary": 1.0, "rule": 0.8, "card": 0.4},
            "complex_interaction": {"rule": 1.0, "glossary": 0.7, "card": 0.5},
            "strategic": {"strategy": 1.0, "card": 0.7, "deck": 0.9, "article": 0.8},
            "general": {"glossary": 0.7, "rule": 0.7, "card": 0.7, "article": 0.7},
        }

        # Get match dictionary for this query type
        match_dict = type_matches.get(query_type, type_matches["general"])

        # Return score for this document type, defaulting to 0.5
        return match_dict.get(doc_type, 0.5)

    def _compress_document(
        self, text: str, query: str, query_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Compress document by extracting the most query-relevant parts.

        Args:
            text: Document text to compress
            query: Query text
            query_analysis: Optional query analysis results

        Returns:
            Compressed document text
        """
        # Split document into sentences or sections
        parts = re.split(r"(?<=[.!?:])\s+", text)

        # If document is short enough, return as-is
        if len(parts) <= 5:
            return text

        # Extract keywords from query
        keywords = self._extract_keywords(query, query_analysis)

        # Score each part based on keyword mentions
        scored_parts = []

        for part in parts:
            score = 0
            for keyword in keywords:
                if keyword.lower() in part.lower():
                    score += 1

            scored_parts.append((part, score))

        # Sort by score and keep top parts
        sorted_parts = sorted(scored_parts, key=lambda x: x[1], reverse=True)
        top_parts = [part for part, score in sorted_parts[:10] if score > 0]

        # If we have meaningful parts that match keywords
        if top_parts:
            return " ".join(top_parts)

        # Fallback - just keep first and last sections
        if len(parts) > 10:
            return " ".join(parts[:3] + ["..."] + parts[-3:])

        # If nothing else works, use original text
        return text

    def _truncate_document(self, text: str, max_tokens: int) -> str:
        """
        Truncate document to fit within token limit.

        Args:
            text: Document text to truncate
            max_tokens: Maximum tokens to allow

        Returns:
            Truncated document text
        """
        if not text:
            return ""

        # If we have a tokenizer, use it for precise truncation
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text

            # Truncate to max_tokens tokens
            truncated_tokens = tokens[:max_tokens]

            # Decode and add ellipsis to indicate truncation
            truncated_text = self.tokenizer.decode(truncated_tokens)

            # Ensure the text ends at a sentence boundary if possible
            last_period = truncated_text.rfind(".")
            if (
                last_period > len(truncated_text) * 0.7
            ):  # Only if period is near the end
                truncated_text = truncated_text[: last_period + 1]

            return truncated_text + " [...]"

        # Without tokenizer, use character-based estimation
        # Assume average of 4 characters per token as a rough estimate
        if len(text) <= max_tokens * 4:
            return text

        # Simple character-based truncation
        truncated_text = text[: max_tokens * 4]

        # Ensure truncation at a sentence boundary if possible
        last_period = truncated_text.rfind(".")
        if last_period > len(truncated_text) * 0.7:  # Only if period is near the end
            truncated_text = truncated_text[: last_period + 1]

        return truncated_text + " [...]"

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # If tokenizer is available, use it
        if self.tokenizer:
            return len(self.tokenizer.encode(text))

        # Fallback: estimate based on characters
        # Use a rough approximation of 4 characters per token for English text
        return len(text) // 4 + 1

    def _extract_keywords(
        self, query: str, query_analysis: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Extract keywords from query for document matching.

        Args:
            query: Query text
            query_analysis: Optional query analysis results

        Returns:
            List of keywords extracted from query
        """
        # If query analysis is available, use entities
        if query_analysis and "entities" in query_analysis:
            entities = query_analysis["entities"]
            keywords = [entity["name"] for entity in entities]

            # Add any additional keywords from query analysis
            if "keywords" in query_analysis:
                keywords.extend(query_analysis["keywords"])

            # Return unique keywords
            return list(set(keywords))

        # Simple fallback: split query into words and filter common words
        words = query.split()

        # Filter out common stop words and short words
        stop_words = {
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "and",
            "or",
            "if",
            "of",
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "this",
            "that",
            "does",  # Added to fix test_keyword_extraction
        }

        keywords = [
            word for word in words if word.lower() not in stop_words and len(word) >= 4
        ]

        return keywords
