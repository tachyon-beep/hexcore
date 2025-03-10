#!/usr/bin/env python
# examples/knowledge_system_demo.py
"""
Demonstration of the MTG Knowledge System capabilities.

This script shows how to initialize and use the enhanced knowledge system
to retrieve MTG knowledge using both vector-based and graph-based approaches.
It demonstrates loading data, building indices, and querying information.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add src to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge import (
    MTGRetriever,
    MTGKnowledgeGraph,
    KnowledgeGraphCache,
    RetrievalLatencyTracker,
    QueryAnalyzer,
    ContextAssembler,
    HybridRetriever,
)
from src.data.mtg_data_loader import MTGDataLoader


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_data(data_path="data"):
    """Load MTG data for the knowledge system."""
    print(f"Loading MTG data from {data_path}...")

    # Load data using MTGDataLoader
    data_loader = MTGDataLoader(data_dir=data_path)

    # Load specific data files
    cards = json.loads(Path(data_path, "cards.json").read_text())
    rules = json.loads(Path(data_path, "rules.json").read_text())
    glossary = json.loads(Path(data_path, "glossary.json").read_text())

    print(
        f"Loaded {len(cards)} cards, {len(rules)} rules, {len(glossary)} glossary terms"
    )

    return data_loader, cards, rules, glossary


def prepare_retriever_documents(cards, rules, glossary):
    """Prepare documents for vector-based retrieval."""
    documents = []

    # Add card documents
    for card in cards:
        if "oracle_text" in card and card["oracle_text"]:
            doc = {
                "id": f"card:{card['id']}",
                "type": "card",
                "text": f"Card: {card['name']}\n\n{card.get('oracle_text', '')}",
            }
            documents.append(doc)

    # Add rule documents
    for rule in rules:
        doc = {
            "id": f"rule:{rule['id']}",
            "type": "rule",
            "text": f"Rule {rule['id']}: {rule.get('text', '')}",
        }
        documents.append(doc)

    # Add glossary documents
    for term, definition in glossary.items():
        doc = {
            "id": f"glossary:{term}",
            "type": "glossary",
            "text": f"Glossary Term: {term}\n\n{definition}",
        }
        documents.append(doc)

    print(f"Prepared {len(documents)} documents for retrieval")
    return documents


def initialize_knowledge_system(data_loader, cards, rules, glossary, documents):
    """Initialize the knowledge system components."""
    print("Initializing knowledge system components...")

    # Create latency tracker
    latency_tracker = RetrievalLatencyTracker(window_size=100)

    # Create cache
    cache = KnowledgeGraphCache(max_cache_size=1000, ttl_seconds=3600)

    # Create query analyzer
    query_analyzer = QueryAnalyzer(data_loader=data_loader)

    # Initialize hybrid retriever (this will create the base components as needed)
    hybrid_retriever = HybridRetriever(
        vector_retriever=None,  # Will be initialized later
        knowledge_graph=None,  # Will be initialized later
        cache_manager=cache,
        latency_tracker=latency_tracker,
        query_analyzer=query_analyzer,
        data_loader=data_loader,
    )

    # Build knowledge graph
    print("Building knowledge graph...")
    start_time = time.time()
    hybrid_retriever.build_knowledge_graph(cards, rules, glossary)
    graph_time = time.time() - start_time
    print(f"Knowledge graph built in {graph_time:.2f} seconds")

    # Initialize vector retriever
    print("Initializing vector retriever (this may take a minute)...")
    start_time = time.time()
    hybrid_retriever.initialize_vector_retriever(documents)
    vector_time = time.time() - start_time
    print(f"Vector retriever initialized in {vector_time:.2f} seconds")

    return hybrid_retriever


def demonstrate_retrieval(hybrid_retriever):
    """Demonstrate different retrieval capabilities."""
    print("\n--- RETRIEVAL DEMONSTRATIONS ---\n")

    # Simple vector retrieval
    query = "How does flying work in Magic?"
    print(f"Vector Retrieval Query: '{query}'")
    vector_results = hybrid_retriever._retrieve_vector(query, top_k=3)
    print(f"Found {len(vector_results)} results from vector search")
    for i, result in enumerate(vector_results):
        print(f"  Result {i+1}: {result['type']} - Score: {result['score']:.4f}")
        print(f"  {result['text'][:100]}...")

    # Graph-based retrieval
    query = "What are the rules for trample?"
    query_analysis = hybrid_retriever.query_analyzer.analyze_query(query)
    print(f"\nGraph Retrieval Query: '{query}'")
    print(
        f"Query Analysis: {query_analysis['query_type']}, Entities: {[e['name'] for e in query_analysis['entities']]}"
    )
    graph_results = hybrid_retriever._retrieve_graph(query, query_analysis)
    print(f"Found {len(graph_results)} results from graph search")

    # Hybrid retrieval
    complex_queries = [
        "How do lifelink and deathtouch interact?",
        "What happens when Wrath of God is cast while a creature has regeneration?",
        "How does first strike work in combat against a creature with deathtouch?",
    ]

    for query in complex_queries:
        print(f"\nHybrid Retrieval Query: '{query}'")
        start_time = time.time()
        results = hybrid_retriever.retrieve(query, top_k=5)
        retrieval_time = time.time() - start_time

        print(f"Found {len(results)} results in {retrieval_time:.4f} seconds")
        for i, result in enumerate(results[:2]):  # Show only first 2 for brevity
            print(f"  Result {i+1}: {result.get('type', 'unknown')}")
            text = result.get("text", result.get("content", "No text"))
            print(f"  {text[:100]}...")

        # Use context assembly
        print("\nAssembling context for the model...")
        context = hybrid_retriever.retrieve_and_assemble(query, max_tokens=1024)
        print(f"Assembled context with {context.get('token_count', 0)} tokens")
        if "metrics" in context:
            metrics = context["metrics"]
            print(f"  Retrieval time: {metrics.get('retrieval_time_ms', 0):.2f}ms")
            print(f"  Total time: {metrics.get('total_time_ms', 0):.2f}ms")
            print(f"  Documents included: {metrics.get('docs_included', 0)}")
        print("\n" + "-" * 50)


def main():
    """Main function to run the demonstration."""
    setup_logging()
    print("MTG Knowledge System Demonstration")
    print("=" * 50)

    # Load data
    data_loader, cards, rules, glossary = load_data()

    # Prepare documents for retrieval
    documents = prepare_retriever_documents(cards, rules, glossary)

    # Initialize the knowledge system
    hybrid_retriever = initialize_knowledge_system(
        data_loader, cards, rules, glossary, documents
    )

    # Demonstrate retrieval capabilities
    demonstrate_retrieval(hybrid_retriever)

    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()
