# src/knowledge/__init__.py
"""
Knowledge subsystem for MTG AI, providing access to Magic: The Gathering information.

This package contains components for retrieving, analyzing, and presenting
knowledge about Magic: The Gathering cards, rules, mechanics, and interactions.
It combines vector-based semantic search with structured knowledge graph traversal,
optimized for both accuracy and performance.

Key components:
- MTGRetriever: Vector-based semantic search for MTG knowledge
- MTGKnowledgeGraph: Structured knowledge graph for entity and relationship queries
- HybridRetriever: Combined retrieval system that intelligently selects the best approach
- QueryAnalyzer: Query analysis for determining optimal retrieval strategy
- ContextAssembler: Assembly of retrieved knowledge into optimized context for LLMs
- KnowledgeGraphCache: Caching system for retrieval results
- RetrievalLatencyTracker: Performance monitoring and budgeting
"""

from .retriever import MTGRetriever
from .knowledge_graph import MTGKnowledgeGraph
from .cache_manager import KnowledgeGraphCache
from .latency_tracker import RetrievalLatencyTracker
from .query_analyzer import QueryAnalyzer
from .context_assembler import ContextAssembler
from .hybrid_retriever import HybridRetriever

__all__ = [
    "MTGRetriever",
    "MTGKnowledgeGraph",
    "KnowledgeGraphCache",
    "RetrievalLatencyTracker",
    "QueryAnalyzer",
    "ContextAssembler",
    "HybridRetriever",
]
