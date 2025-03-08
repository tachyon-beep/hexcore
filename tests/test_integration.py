# tests/test_integration.py
import pytest
import json
import sys
import time
from pathlib import Path
from typing import List, Dict
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_loader import load_quantized_model
from src.knowledge.retreiver import MTGRetriever
from src.models.transaction_classifier import TransactionClassifier
from src.data.mtg_data_loader import MTGDataLoader
from src.inference.pipeline import MTGInferencePipeline
from src.utils.gpu_memory_tracker import GPUMemoryTracker
from src.models.expert_adapters import ExpertAdapterManager  # Add this import
from src.models.cross_expert import CrossExpertAttention  # Add this import


@pytest.fixture(scope="module")
def prepared_pipeline():
    """Setup the complete pipeline for testing. This fixture is scoped at module level
    to avoid repeated loading of the heavy model for each test function."""

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_quantized_model()
    assert isinstance(
        model, PreTrainedModel
    ), "Loaded model is not a PreTrainedModel instance"
    assert isinstance(
        tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    ), "Tokenizer is not a valid PreTrainedTokenizer instance"

    # Load and validate MTG data
    print("Loading MTG data...")
    data_loader = MTGDataLoader()
    documents = data_loader.load_data()
    assert len(documents) > 0, "No documents loaded from data source"

    # Prepare documents for retriever with required fields
    documents_dicts: List[Dict[str, str]] = []
    for idx, doc in enumerate(documents, start=1):
        # Validate document structure
        assert "text" in doc, f"Document {idx} missing 'text' field"
        assert "metadata" in doc, f"Document {idx} missing 'metadata' field"

        # Construct document with required fields for retriever
        processed_doc = {
            "text": doc["text"],
            "metadata": json.dumps(doc["metadata"]),
            "type": doc.get("type", "rules"),  # Default document type
            "id": str(
                doc.get("id", hash(doc["text"]))
            ),  # Generate ID from text hash if missing
        }
        documents_dicts.append(processed_doc)

    # Initialize and populate knowledge retriever
    print("Initializing knowledge retriever...")
    retriever = MTGRetriever()
    retriever.index_documents(documents_dicts)
    assert len(retriever.get_index()) == len(
        documents_dicts
    ), "Retriever index count doesn't match document count"

    # Initialize transaction classifier
    print("Initializing transaction classifier...")
    classifier = TransactionClassifier()
    assert hasattr(
        classifier, "classify"
    ), "TransactionClassifier missing required 'classify' method"

    # Initialize expert adapter manager
    print("Initializing expert adapter manager...")
    expert_manager = ExpertAdapterManager(model, adapters_dir="adapters")

    # Initialize cross-expert attention
    print("Initializing cross-expert attention...")
    cross_expert = CrossExpertAttention(hidden_size=model.config.hidden_size)

    # Construct inference pipeline
    print("Building inference pipeline...")
    pipeline = MTGInferencePipeline(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
        retriever=retriever,
        data_loader=data_loader,
        expert_manager=expert_manager,  # Add expert manager
        cross_expert_attention=cross_expert,  # Add cross-expert attention
    )
    assert hasattr(
        pipeline, "generate_response"
    ), "Pipeline missing required 'generate_response' method"

    # Return the prepared pipeline
    return pipeline


@pytest.mark.integration
def test_basic_pipeline_operation(prepared_pipeline):
    """Test basic operation of the complete pipeline."""
    result = prepared_pipeline.generate_response(
        "What does Lightning Bolt do?",
        use_multiple_experts=False,  # Specify single expert mode for testing
    )

    # Validate response structure
    assert isinstance(result, dict), "Response should be a dictionary"

    # Validate required fields - update field names to match new structure
    assert "expert_types" in result, "Missing expert_types in response"
    assert "confidences" in result, "Missing confidences scores"
    assert "response" in result, "Missing response text"
    assert "metrics" in result, "Missing performance metrics"

    # Validate data types
    assert isinstance(result["expert_types"], list), "Expert types should be a list"
    assert len(result["expert_types"]) > 0, "Expert types list should not be empty"
    assert isinstance(result["confidences"], dict), "Confidences should be a dictionary"
    assert isinstance(result["response"], str), "Response should be a string"
    assert isinstance(result["metrics"], dict), "Metrics should be a dictionary"

    # Validate response content
    assert len(result["response"]) >= 50, "Response seems too short"
    assert (
        "damage" in result["response"].lower()
    ), "Response should mention 'damage' for Lightning Bolt"

    # Validate metrics
    assert "total_time" in result["metrics"], "Missing total_time metric"
    assert result["metrics"]["total_time"] > 0, "Invalid processing time"


@pytest.mark.integration
@pytest.mark.parametrize(
    "query,expected_expert,keywords",
    [
        (
            "What happens when Lightning Bolt targets a creature with protection from red?",
            "REASON",
            ["protection", "damage", "prevent"],
        ),
        (
            "Explain how the stack works in Magic?",
            "EXPLAIN",
            ["last", "first", "resolve"],
        ),
        (
            "I'm new to Magic. Can you teach me about mana?",
            "TEACH",
            ["land", "color", "produce"],
        ),
        (
            "In my current board state, should I attack with all creatures or hold back blockers?",
            "PREDICT",
            ["opponent", "risk", "damage"],
        ),
        (
            "I lost my last game when my opponent played a board wipe. What could I have done differently?",
            "RETROSPECT",
            ["hold", "counter", "anticipate"],
        ),
    ],
)
def test_expert_classification(prepared_pipeline, query, expected_expert, keywords):
    """Test that different query types are routed to the appropriate experts."""
    print(f"Testing expert classification for query: {query[:50]}...")
    result = prepared_pipeline.generate_response(
        query, use_multiple_experts=False  # Use single expert for clear testing
    )

    # Check that the correct expert was chosen (now it's a list)
    assert (
        expected_expert in result["expert_types"]
    ), f"Expected {expected_expert} in expert_types list"

    # Check if it's the primary expert (first in the list)
    assert (
        result["expert_types"][0] == expected_expert
    ), f"Expected {expected_expert} as primary expert"

    # Check for expected keywords in the response
    response_lower = result["response"].lower()
    for keyword in keywords:
        assert (
            keyword.lower() in response_lower
        ), f"Keyword '{keyword}' not found in response"


@pytest.mark.integration
def test_card_specific_knowledge(prepared_pipeline):
    """Test that the system correctly retrieves and uses card-specific knowledge."""
    result = prepared_pipeline.generate_response(
        "What are the best targets for Path to Exile?", use_multiple_experts=False
    )

    # Should mention high-value targets
    assert any(
        term in result["response"].lower()
        for term in ["creature", "threat", "opponent", "exile"]
    ), "Response should mention appropriate targets for Path to Exile"


@pytest.mark.integration
def test_rules_knowledge(prepared_pipeline):
    """Test that the system correctly handles rules-based queries."""
    result = prepared_pipeline.generate_response(
        "How does trample work when a creature is blocked?", use_multiple_experts=False
    )

    # Should reference combat damage and excess damage
    assert (
        "excess" in result["response"].lower()
    ), "Response should mention 'excess' damage"
    assert (
        "defend" in result["response"].lower() or "player" in result["response"].lower()
    ), "Response should mention damage to the defending player"


@pytest.mark.integration
def test_error_handling_edge_cases(prepared_pipeline):
    """Test error handling and edge cases."""
    # Empty query test
    empty_result = prepared_pipeline.generate_response("", use_multiple_experts=False)
    assert empty_result is not None, "Should handle empty query gracefully"
    assert "response" in empty_result, "Should provide a response even for empty query"

    # Very short query test
    short_result = prepared_pipeline.generate_response(
        "MTG?", use_multiple_experts=False
    )
    assert short_result is not None, "Should handle very short query"
    assert len(short_result["response"]) > 0, "Should provide a non-empty response"


@pytest.mark.integration
def test_memory_usage(prepared_pipeline):
    """Test memory usage during inference."""
    # Skip if GPU memory tracking is not available
    try:
        memory_tracker = GPUMemoryTracker()
    except Exception as e:
        pytest.skip(f"GPU memory tracking not available: {e}")

    # Start memory tracking
    memory_tracker.start_monitoring()

    # Process a complex query that would require significant resources
    complex_query = (
        "I'm playing a Modern tournament with Jund. My hand is two Thoughtseize, "
        "Tarmogoyf, Lightning Bolt, Bloodbraid Elf, and two lands (one Blackcleave "
        "Cliffs, one Overgrown Tomb). My opponent is on Amulet Titan and just played "
        "a turn one Amulet of Vigor. What's my optimal play sequence for the next two turns?"
    )

    result = prepared_pipeline.generate_response(
        complex_query, use_multiple_experts=False
    )

    # Stop tracking and get memory stats
    memory_tracker.stop_monitoring()
    max_gpu_usage, max_cpu_usage = memory_tracker.get_max_memory_usage()

    # Print memory usage for debugging (won't fail test)
    print(f"Max GPU memory usage: {max_gpu_usage}")
    print(f"Max CPU memory usage: {max_cpu_usage}MB")

    # Ensure we got a valid response
    assert result is not None
    assert "response" in result
    assert len(result["response"]) > 100


@pytest.mark.integration
def test_response_time_performance(prepared_pipeline):
    """Test response time performance."""
    start_time = time.time()

    # Process a simple query
    result = prepared_pipeline.generate_response(
        "What is Magic: The Gathering?", use_multiple_experts=False
    )

    # Ensure the response doesn't take too long
    total_time = time.time() - start_time
    print(f"Total response time: {total_time:.2f}s")

    # Check that response time metrics match actual time (within reason)
    assert (
        abs(result["metrics"]["total_time"] - total_time) < 1.0
    ), "Reported time metrics should be close to actual time"

    # For CI/CD, we might want to set a maximum acceptable time
    # This is commented out as actual performance depends on hardware
    # assert total_time < 10.0, "Response should be generated within 10 seconds"


@pytest.mark.integration
def test_multiple_sequential_queries(prepared_pipeline):
    """Test handling multiple sequential queries to ensure stability."""
    queries = [
        "What does Counterspell do?",
        "How much life do players start with in Commander?",
        "What's the best way to deal with planeswalkers?",
        "How does casting spells work in Magic?",
        "What happens during the draw step?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"Processing query {i}/{len(queries)}: {query}")
        result = prepared_pipeline.generate_response(query, use_multiple_experts=False)
        assert result is not None, f"Failed to get response for query {i}: {query}"
        assert "response" in result, f"Missing response for query {i}"
        assert len(result["response"]) > 0, f"Empty response for query {i}"

        # Allow a short delay between queries
        time.sleep(0.5)


@pytest.mark.integration
def test_multi_expert_mode(prepared_pipeline):
    """Test the multi-expert mode functionality."""
    # This is a new test to specifically verify multi-expert functionality
    query = "What happens when I cast Lightning Bolt on a creature with hexproof and my opponent responds with Veil of Summer?"

    # This complex rules interaction is a good candidate for multi-expert processing
    result = prepared_pipeline.generate_response(query, use_multiple_experts=True)

    # Verify we got multiple experts
    assert (
        len(result["expert_types"]) >= 2
    ), "Should use multiple experts for complex query"

    # Verify the primary expert is REASON (for rules interactions)
    assert (
        "REASON" in result["expert_types"]
    ), "REASON expert should be used for rules interaction"

    # Verify we got reasonable content
    assert "hexproof" in result["response"].lower(), "Response should address hexproof"
    assert (
        "veil of summer" in result["response"].lower()
    ), "Response should address Veil of Summer"
    assert "target" in result["response"].lower(), "Response should discuss targeting"

    # Response should be comprehensive due to multiple experts
    assert (
        len(result["response"]) > 200
    ), "Multi-expert response should be comprehensive"
