# src/knowledge/hybrid_retriever.py
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import threading

from .retriever import MTGRetriever
from .knowledge_graph import MTGKnowledgeGraph
from .cache_manager import KnowledgeGraphCache
from .latency_tracker import RetrievalLatencyTracker
from .query_analyzer import QueryAnalyzer
from .context_assembler import ContextAssembler

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid knowledge retrieval system combining vector search and knowledge graph.

    This component orchestrates the knowledge retrieval process, dynamically choosing
    between vector-based semantic search and structured knowledge graph traversal
    based on query characteristics. It integrates:

    - Vector-based semantic search for natural language understanding
    - Knowledge graph traversal for structured relationship queries
    - Query analysis to determine optimal retrieval strategy
    - Caching for performance optimization
    - Latency tracking and monitoring
    - Context assembly for optimal prompt construction
    """

    def __init__(
        self,
        vector_retriever: Optional[MTGRetriever] = None,
        knowledge_graph: Optional[MTGKnowledgeGraph] = None,
        cache_manager: Optional[KnowledgeGraphCache] = None,
        latency_tracker: Optional[RetrievalLatencyTracker] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
        context_assembler: Optional[ContextAssembler] = None,
        data_loader=None,
        tokenizer=None,
        default_latency_budget_ms: float = 200.0,
    ):
        """
        Initialize the hybrid retriever.

        Args:
            vector_retriever: Vector-based retriever (MTGRetriever)
            knowledge_graph: Knowledge graph for structured queries
            cache_manager: Cache manager for result caching
            latency_tracker: Latency tracker for performance monitoring
            query_analyzer: Query analyzer for strategy selection
            context_assembler: Context assembler for formatting results
            data_loader: Data loader for MTG data
            tokenizer: Tokenizer for token counting
            default_latency_budget_ms: Default latency budget for retrieval
        """
        # Initialize components, creating them if not provided
        self.vector_retriever = vector_retriever or MTGRetriever()
        self.knowledge_graph = knowledge_graph or MTGKnowledgeGraph()
        self.cache = cache_manager or KnowledgeGraphCache()
        self.latency_tracker = latency_tracker or RetrievalLatencyTracker()
        self.query_analyzer = query_analyzer or QueryAnalyzer(data_loader=data_loader)
        self.context_assembler = context_assembler or ContextAssembler(
            tokenizer=tokenizer, latency_tracker=self.latency_tracker
        )

        self.data_loader = data_loader
        self.tokenizer = tokenizer

        # Configuration
        self.default_latency_budget_ms = default_latency_budget_ms

        # Component initialization status
        self.vector_retriever_initialized = vector_retriever is not None
        self.knowledge_graph_initialized = knowledge_graph is not None

        # Thread safety for initialization
        self._init_lock = threading.RLock()

        logger.info(
            f"Initialized hybrid retriever (latency_budget={default_latency_budget_ms}ms, "
            f"vector_initialized={self.vector_retriever_initialized}, "
            f"graph_initialized={self.knowledge_graph_initialized})"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        latency_budget_ms: Optional[float] = None,
        prioritize_graph: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve information using the optimal method based on query analysis.

        Args:
            query: User query string
            top_k: Number of results to retrieve
            latency_budget_ms: Maximum allowed retrieval time in milliseconds
            prioritize_graph: Whether to prioritize graph search over vector search

        Returns:
            List of retrieved information items
        """
        start_time = time.time()

        # Set default latency budget if not provided
        latency_budget = latency_budget_ms or self.default_latency_budget_ms

        # Check cache first
        cache_key = self._generate_cache_key(query, top_k, prioritize_graph)
        cache_lookup_start = time.time()
        cached_result = self.cache.get(cache_key)
        cache_lookup_time = (time.time() - cache_lookup_start) * 1000

        # Record cache lookup latency
        self.latency_tracker.record("cache_lookup", cache_lookup_time)

        # Process cached result
        if cached_result:
            # Fast path: return cached result
            retrieval_time = (time.time() - start_time) * 1000
            self.latency_tracker.record("total", retrieval_time)
            self.latency_tracker.record("cached", retrieval_time)

            # Ensure we return the right type
            if isinstance(cached_result, list):
                # If it's already a list, just return it with metadata annotation as needed
                result_list = cached_result

                # Add cache hit info if results are dicts
                if result_list and isinstance(result_list[0], dict):
                    for result in result_list:
                        if "_metadata" not in result:
                            result["_metadata"] = {"cache_hit": True}

                return result_list
            else:
                # If it's not a list, wrap it in a list
                return [{"content": cached_result, "_metadata": {"cache_hit": True}}]

        # Update remaining budget after cache lookup
        remaining_budget_ms = latency_budget - cache_lookup_time

        # Cache miss - analyze query to determine optimal retrieval strategy
        query_analysis_start = time.time()
        query_analysis = self.query_analyzer.analyze_query(query)
        query_analysis_time = (time.time() - query_analysis_start) * 1000

        # Record query analysis latency
        self.latency_tracker.record("query_analysis", query_analysis_time)

        # Check if we've already spent too much time
        remaining_budget_ms -= query_analysis_time
        if remaining_budget_ms <= 0:
            # Emergency fallback to simple vector retrieval
            logger.warning(
                f"Latency budget exceeded during query analysis. "
                f"Using emergency vector retrieval fallback."
            )

            # Record total latency before returning
            total_time = (time.time() - start_time) * 1000
            self.latency_tracker.record("total", total_time)

            return self._emergency_retrieval(query, top_k)

        # Prepare retrieval strategy based on query analysis and priority
        use_graph = (
            query_analysis["requires_structured_knowledge"] or prioritize_graph
        ) and self.knowledge_graph_initialized

        # Define latency allocations based on strategy
        if use_graph:
            vector_allocation = min(
                remaining_budget_ms * 0.4, 80
            )  # 40% for vector, max 80ms
            graph_allocation = min(
                remaining_budget_ms * 0.5, 100
            )  # 50% for graph, max 100ms
            merge_allocation = min(
                remaining_budget_ms * 0.1, 20
            )  # 10% for merging, max 20ms
        else:
            # Vector-only allocation
            vector_allocation = min(
                remaining_budget_ms * 0.9, 180
            )  # 90% for vector, max 180ms
            graph_allocation = 0
            merge_allocation = min(
                remaining_budget_ms * 0.1, 20
            )  # 10% for post-processing

        # Prepare results containers
        vector_results = []
        graph_results = []

        # Vector retrieval (always perform as a fallback)
        if self.vector_retriever_initialized:
            vector_start = time.time()
            try:
                # Perform vector retrieval with timeout
                vector_results = self._retrieve_vector(
                    query, top_k=top_k, timeout_ms=vector_allocation
                )
            except TimeoutError:
                logger.warning(f"Vector retrieval timed out for query: {query}")
            vector_time = (time.time() - vector_start) * 1000
            self.latency_tracker.record("vector", vector_time)

        # Graph traversal if appropriate and time allows
        if use_graph and self.knowledge_graph_initialized and graph_allocation > 0:
            graph_start = time.time()
            try:
                # Perform graph traversal with timeout
                graph_results = self._retrieve_graph(
                    query, query_analysis, timeout_ms=graph_allocation
                )
            except TimeoutError:
                logger.warning(f"Graph traversal timed out for query: {query}")
            except Exception as e:
                logger.error(f"Error during graph retrieval: {str(e)}")
            graph_time = (time.time() - graph_start) * 1000
            self.latency_tracker.record("graph", graph_time)

        # Check remaining time for merging
        elapsed_ms = (time.time() - start_time) * 1000
        remaining_budget_ms = latency_budget - elapsed_ms

        # Merge and deduplicate results
        merge_start = time.time()
        merged_results: List[Dict[str, Any]] = []  # Initialize with correct type

        try:
            if remaining_budget_ms <= 0:
                # Emergency return if over budget
                raw_results = vector_results or graph_results
                # Convert results to list of dicts if needed
                merged_results = self._ensure_results_format(raw_results)
                logger.warning(
                    f"Latency budget exceeded before merging. "
                    f"Returning {'vector' if vector_results else 'graph'} results directly."
                )
            else:
                # Merge vector and graph results with deduplication and ranking
                merged_results = self._merge_results(
                    vector_results,
                    graph_results,
                    query,
                    query_analysis,
                    timeout_ms=merge_allocation,
                )
        except Exception as e:
            logger.error(f"Error merging results: {str(e)}")
            # Emergency fallback - ensure it's a list of dicts
            raw_results = vector_results or graph_results
            merged_results = self._ensure_results_format(raw_results)

        merge_time = (time.time() - merge_start) * 1000
        self.latency_tracker.record("merge", merge_time)

        # Record metrics for this retrieval
        total_time = (time.time() - start_time) * 1000
        self.latency_tracker.record("total", total_time)

        # Create retrieval metadata
        retrieval_metadata = {
            "latency_ms": total_time,
            "vector_count": len(vector_results),
            "graph_count": len(graph_results),
            "merged_count": len(merged_results),
            "query_type": query_analysis["query_type"],
            "entities_found": len(query_analysis["entities"]),
            "strategy": (
                "hybrid"
                if (vector_results and graph_results)
                else "vector" if vector_results else "graph"
            ),
        }

        # Add metadata to results - create new list to avoid modifying originals
        results_with_metadata = []
        for result in merged_results:
            result_copy = result.copy()
            result_copy["_metadata"] = retrieval_metadata
            results_with_metadata.append(result_copy)

        # Cache the result with metadata for entity-based invalidation
        cache_metadata = {
            "entity_types": list(
                set(entity["type"] for entity in query_analysis["entities"])
            ),
            "entities": query_analysis["entities"],
            "query_type": query_analysis["query_type"],
            "timestamp": time.time(),
        }
        self.cache.set(cache_key, results_with_metadata, cache_metadata)

        # Alert if consistently over budget
        if total_time > latency_budget:
            logger.warning(
                f"Retrieval latency ({total_time:.1f}ms) exceeded budget ({latency_budget}ms) "
                f"for query: {query[:50]}..."
            )

        return results_with_metadata

    def _ensure_results_format(self, results: Any) -> List[Dict[str, Any]]:
        """
        Ensure results are in the correct format (list of dicts).

        Args:
            results: Results to normalize

        Returns:
            List of dictionaries
        """
        if not results:
            return []

        # If it's already a list
        if isinstance(results, list):
            # Convert non-dict items to dicts
            return [
                item if isinstance(item, dict) else {"content": item}
                for item in results
            ]

        # If it's a single dict
        if isinstance(results, dict):
            return [results]

        # If it's anything else
        return [{"content": results}]

    def retrieve_and_assemble(
        self,
        query: str,
        max_tokens: int = 3072,
        latency_budget_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve knowledge and assemble it into a context for the model.

        This is a convenience method that combines retrieval and context assembly.

        Args:
            query: User query string
            max_tokens: Maximum tokens for the assembled context
            latency_budget_ms: Maximum retrieval latency budget

        Returns:
            Dictionary with assembled context and metadata
        """
        # Allocate latency budget: 80% for retrieval, 20% for assembly
        total_budget = latency_budget_ms or self.default_latency_budget_ms
        retrieval_budget = total_budget * 0.8
        assembly_budget = total_budget * 0.2

        # First retrieve information
        start_time = time.time()

        # Analyze query for retrieval strategy - only analyze once to avoid duplicate calls
        query_analysis = self.query_analyzer.analyze_query(query)
        prioritize_graph = query_analysis.get("prioritize_graph", False)

        # Store the query analysis to prevent the retrieve method from calling it again
        # by caching the result for this specific query
        cache_key = self._generate_cache_key(query, 10, prioritize_graph)
        self.cache.set(
            cache_key + "_analysis", query_analysis, {"timestamp": time.time()}
        )

        # Retrieve results
        retrieved_info = self.retrieve(
            query,
            top_k=10,  # Get more than needed to allow filtering
            latency_budget_ms=retrieval_budget,
            prioritize_graph=prioritize_graph,
        )

        retrieval_time = (time.time() - start_time) * 1000

        # Then assemble context
        context_result = self.context_assembler.assemble_context(
            query,
            retrieved_info,
            query_analysis=query_analysis,
            max_tokens=max_tokens,
            latency_budget_ms=assembly_budget,
        )

        # Add retrieval metrics to context result
        if isinstance(context_result, dict):
            metrics = context_result.get("metrics", {})
            if not isinstance(metrics, dict):
                metrics = {}
            metrics["retrieval_time_ms"] = retrieval_time
            metrics["total_time_ms"] = (time.time() - start_time) * 1000
            context_result["metrics"] = metrics

        return context_result

    def _retrieve_vector(
        self, query: str, top_k: int = 5, timeout_ms: float = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform vector-based retrieval with timeout.

        Args:
            query: User query
            top_k: Number of results to retrieve
            timeout_ms: Maximum time in milliseconds

        Returns:
            List of retrieved documents

        Raises:
            TimeoutError: If retrieval exceeds timeout
        """
        start_time = time.time()
        deadline = start_time + (timeout_ms / 1000)

        # Check if retriever is initialized
        if not self.vector_retriever_initialized:
            logger.warning(
                "Vector retriever not initialized, skipping vector retrieval"
            )
            return []

        # Perform vector retrieval
        try:
            results = self.vector_retriever.retrieve(query, top_k=top_k)

            # Check timeout
            if time.time() > deadline:
                raise TimeoutError(
                    f"Vector retrieval exceeded timeout of {timeout_ms}ms"
                )

            # Ensure correct return type
            return self._ensure_results_format(results)
        except Exception as e:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Vector retrieval exceeded timeout of {timeout_ms}ms"
                )
            raise e

    def _retrieve_graph(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        timeout_ms: float = 100,
    ) -> List[Dict[str, Any]]:
        """
        Perform knowledge graph traversal with timeout.

        Args:
            query: User query
            query_analysis: Query analysis result
            timeout_ms: Maximum time in milliseconds

        Returns:
            List of retrieved information

        Raises:
            TimeoutError: If traversal exceeds timeout
        """
        start_time = time.time()
        deadline = start_time + (timeout_ms / 1000)

        # Check if knowledge graph is initialized
        if not self.knowledge_graph_initialized:
            logger.warning("Knowledge graph not initialized, skipping graph traversal")
            return []

        # Extract entities to query
        entities = query_analysis["entities"]
        if not entities:
            return []  # No entities to query

        results = []

        # Process only top 3 entities to avoid timeout
        for entity in entities[:3]:
            # Check timeout before each entity
            if time.time() > deadline:
                raise TimeoutError(
                    f"Graph traversal exceeded timeout of {timeout_ms}ms"
                )

            entity_type = entity["type"]
            entity_name = entity["name"]

            # Query entity information
            entity_result = self.knowledge_graph.query(
                "entity", entity_type=entity_type, entity_name=entity_name
            )

            # Add entity results
            results.extend(entity_result)

            # Check timeout before querying neighbors
            if time.time() > deadline:
                break

            # If we have a matching entity, query its neighbors
            if entity_result:
                entity_id = entity_result[0].get("id")
                if entity_id:
                    # Get relationship types from query analysis if available
                    relation_types = query_analysis.get("relationship_types", [])

                    # For each relevant relationship type
                    for rel_type in relation_types[:2]:  # Limit to 2 types
                        # Check timeout
                        if time.time() > deadline:
                            break

                        # Query neighbors through this relationship
                        neighbors = self.knowledge_graph.query(
                            "neighbors",
                            entity_type=entity_type,
                            entity_id=entity_id,
                            relation_type=rel_type,
                        )

                        # Add neighborhood results
                        results.extend(neighbors)

        # Check one final time
        if time.time() > deadline:
            raise TimeoutError(f"Graph traversal exceeded timeout of {timeout_ms}ms")

        # Ensure correct format
        return self._ensure_results_format(results)

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        query: str,
        query_analysis: Dict[str, Any],
        timeout_ms: float = 20,
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from vector and graph retrieval.

        Args:
            vector_results: Results from vector retrieval
            graph_results: Results from graph traversal
            query: Original query
            query_analysis: Query analysis result
            timeout_ms: Maximum time in milliseconds

        Returns:
            Merged and deduplicated results
        """
        start_time = time.time()
        deadline = start_time + (timeout_ms / 1000)

        # Handle cases with only one type of results
        if not vector_results:
            return self._ensure_results_format(graph_results)
        if not graph_results:
            return self._ensure_results_format(vector_results)

        # Create a map of document IDs we've seen
        seen_ids = set()
        merged = []

        # Helper to add a result if we haven't seen it before
        def add_unique_result(result, priority: int = 0):
            doc_id = result.get("id", None)

            # If no ID, use content hash
            if doc_id is None:
                content = result.get("text", "") or result.get("content", "")
                doc_id = hash(content)

            if doc_id not in seen_ids:
                seen_ids.add(doc_id)

                # Add priority score (use copy to avoid modifying originals)
                result_copy = result.copy()
                result_copy["_priority"] = priority
                merged.append(result_copy)
                return True
            return False

        # Add graph results first (prioritized for structured queries)
        if query_analysis["prioritize_graph"]:
            for result in graph_results:
                if time.time() > deadline:
                    break
                add_unique_result(result, priority=20)  # Integer priority

            # Then add vector results
            for result in vector_results:
                if time.time() > deadline:
                    break
                add_unique_result(result, priority=10)  # Integer priority
        else:
            # Normal priority: mix results based on score

            # Add the highest scoring vector result first for general context
            if vector_results:
                add_unique_result(vector_results[0], priority=30)  # Integer priority

            # Add the highest scoring graph result for specific information
            if graph_results:
                add_unique_result(graph_results[0], priority=30)  # Integer priority

            # Now interleave remaining results
            max_count = max(len(vector_results), len(graph_results))
            for i in range(1, max_count):
                # Check timeout
                if time.time() > deadline:
                    break

                # Calculate integer priorities
                vector_priority = 20 - min(i, 10)  # Integer priority from 19-10
                graph_priority = 20 - min(i, 10)  # Integer priority from 19-10

                # Add vector result if available
                if i < len(vector_results):
                    add_unique_result(vector_results[i], priority=vector_priority)

                # Add graph result if available
                if i < len(graph_results):
                    add_unique_result(graph_results[i], priority=graph_priority)

        # Sort by priority (if available)
        merged.sort(key=lambda x: x.get("_priority", 0), reverse=True)

        # Remove temporary priority field
        for result in merged:
            if "_priority" in result:
                del result["_priority"]

        return merged

    def _generate_cache_key(
        self, query: str, top_k: int, prioritize_graph: bool
    ) -> str:
        """
        Generate a cache key for a retrieval request.

        Args:
            query: User query
            top_k: Number of results requested
            prioritize_graph: Whether graph search is prioritized

        Returns:
            Cache key string
        """
        # Normalize query (lowercase, remove excess whitespace)
        norm_query = " ".join(query.lower().split())

        # Create key components
        components = [f"q={norm_query}", f"k={top_k}", f"g={int(prioritize_graph)}"]

        return "||".join(components)

    def _emergency_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform emergency retrieval when budgets are exceeded.

        Args:
            query: User query
            top_k: Number of results to retrieve

        Returns:
            List of results (may be empty if both retrievers fail)
        """
        # First try vector retrieval (usually faster)
        if self.vector_retriever_initialized:
            try:
                results = self.vector_retriever.retrieve(query, top_k=top_k)
                return self._ensure_results_format(results)
            except Exception as e:
                logger.error(f"Emergency vector retrieval failed: {str(e)}")

        # If vector fails or isn't available, try direct graph query
        if self.knowledge_graph_initialized:
            try:
                # Simple entity extraction
                words = query.lower().split()
                results = []

                # Try each word as a potential entity name (naive approach)
                for word in words:
                    if len(word) >= 4:  # Only try reasonable length words
                        for entity_type in ["card", "mechanic", "keyword"]:
                            entity_results = self.knowledge_graph.query(
                                "entity", entity_type=entity_type, entity_name=word
                            )
                            if entity_results:
                                results.extend(entity_results)
                                if len(results) >= top_k:
                                    return self._ensure_results_format(results[:top_k])

                return self._ensure_results_format(results[:top_k] if results else [])
            except Exception as e:
                logger.error(f"Emergency graph retrieval failed: {str(e)}")

        # If all else fails
        return []

    def build_knowledge_graph(
        self,
        cards_data: List[Dict],
        rules_data: List[Dict],
        glossary_data: Optional[Dict] = None,
    ) -> bool:
        """
        Build the knowledge graph from data sources.

        Args:
            cards_data: List of card data dictionaries
            rules_data: List of rule data dictionaries
            glossary_data: Optional dictionary of glossary terms

        Returns:
            True if successful, False otherwise
        """
        with self._init_lock:
            try:
                start_time = time.time()

                # Initialize graph if needed
                if not self.knowledge_graph_initialized:
                    self.knowledge_graph = MTGKnowledgeGraph()

                # Build the graph
                self.knowledge_graph.build_graph_from_data(
                    cards_data, rules_data, glossary_data
                )

                # Update initialization flag
                self.knowledge_graph_initialized = True

                build_time = (time.time() - start_time) * 1000
                logger.info(f"Knowledge graph built in {build_time:.2f}ms")

                return True
            except Exception as e:
                # Make sure to set initialized to False on error
                self.knowledge_graph_initialized = False
                logger.error(f"Error building knowledge graph: {str(e)}")
                return False

    def initialize_vector_retriever(self, documents: List[Dict[str, str]]) -> bool:
        """
        Initialize the vector retriever with documents.

        Args:
            documents: List of document dictionaries with text, type, and id

        Returns:
            True if successful, False otherwise
        """
        with self._init_lock:
            try:
                start_time = time.time()

                # Initialize retriever if needed
                if not self.vector_retriever_initialized:
                    self.vector_retriever = MTGRetriever()

                # Index documents
                self.vector_retriever.index_documents(documents)

                # Update initialization flag
                self.vector_retriever_initialized = True

                index_time = (time.time() - start_time) * 1000
                logger.info(f"Vector retriever initialized in {index_time:.2f}ms")

                return True
            except Exception as e:
                # Make sure to set initialized to False on error
                self.vector_retriever_initialized = False
                logger.error(f"Error initializing vector retriever: {str(e)}")
                return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the hybrid retriever.

        Returns:
            Dictionary with component status information
        """
        vector_status = (
            "initialized" if self.vector_retriever_initialized else "uninitialized"
        )
        graph_status = (
            "initialized" if self.knowledge_graph_initialized else "uninitialized"
        )

        # Get cache and latency metrics if available
        cache_metrics = {}
        if hasattr(self.cache, "get_metrics"):
            cache_metrics = self.cache.get_metrics()

        latency_metrics = {}
        if hasattr(self.latency_tracker, "get_statistics"):
            latency_metrics = self.latency_tracker.get_statistics()

        return {
            "vector_retriever": vector_status,
            "knowledge_graph": graph_status,
            "cache_metrics": cache_metrics,
            "latency_metrics": latency_metrics,
            "default_latency_budget_ms": self.default_latency_budget_ms,
        }

    def invalidate_cache(
        self, entity_type: Optional[str] = None, entity_id: Optional[str] = None
    ) -> int:
        """
        Invalidate cache entries based on entity type or ID.

        Args:
            entity_type: Optional entity type to invalidate
            entity_id: Optional entity ID to invalidate

        Returns:
            Number of cache entries invalidated
        """
        return self.cache.invalidate(entity_type=entity_type, entity_id=entity_id)
