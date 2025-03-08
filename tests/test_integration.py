# tests/test_integration.py
import pytest
import json
import sys
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


@pytest.mark.integration
def test_integration_pipeline():
    """End-to-end integration test of the MTG assistance pipeline."""

    # Load model and tokenizer with type checking
    model, tokenizer = load_quantized_model()
    assert isinstance(
        model, PreTrainedModel
    ), "Loaded model is not a PreTrainedModel instance"
    assert isinstance(
        tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)
    ), "Tokenizer is not a valid PreTrainedTokenizer instance"

    # Load and validate MTG data
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
    retriever = MTGRetriever()
    retriever.index_documents(documents_dicts)
    assert len(retriever.get_index()) == len(
        documents_dicts
    ), "Retriever index count doesn't match document count"

    # Initialize transaction classifier
    classifier = TransactionClassifier()
    assert classifier is not None, "Failed to initialize TransactionClassifier"

    # Construct inference pipeline
    pipeline = MTGInferencePipeline(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
        retriever=retriever,
        data_loader=data_loader,
    )
    assert pipeline is not None, "Failed to initialize MTGInferencePipeline"

    # Define test queries with varied complexity
    test_queries = [
        ("What protection from red means for Lightning Bolt targets?", "rules"),
        ("How to build a commander deck for beginners?", "deckbuilding"),
        (
            "Respond to opponent's T1 Thoughtseize with this hand: Forest, Llanowar Elves, ...",
            "gameplay",
        ),
    ]

    # Validate pipeline responses
    for query, expected_type in test_queries:
        result = pipeline.generate_response(query)

        # Validate response structure
        assert isinstance(result, dict), "Response should be a dictionary"

        # Validate expert type classification
        assert "expert_type" in result, "Missing expert_type in response"
        assert isinstance(result["expert_type"], str), "Expert type should be a string"
        assert result["expert_type"] in [
            "REASON",
            "EXPLAIN",
            "TEACH",
            "PREDICT",
            "RETROSPECT",
        ], f"Unexpected expert type: {result['expert_type']}"

        # Validate confidence score
        assert "confidence" in result, "Missing confidence score"
        assert isinstance(result["confidence"], float), "Confidence should be a float"
        assert 0 <= result["confidence"] <= 1, "Confidence score out of valid range"

        # Validate response content
        assert "response" in result, "Missing response text"
        assert isinstance(result["response"], str), "Response should be a string"
        assert len(result["response"]) >= 50, "Response seems too short"

        # Validate performance metrics
        assert "metrics" in result, "Missing performance metrics"
        assert isinstance(result["metrics"], dict), "Metrics should be a dictionary"
        assert "total_time" in result["metrics"], "Missing total_time metric"
        assert isinstance(
            result["metrics"]["total_time"], float
        ), "Time metric should be a float"
        assert result["metrics"]["total_time"] > 0, "Invalid processing time"

    # Validate pipeline component integration
    assert pipeline.model is model, "Model instance mismatch in pipeline"
    assert pipeline.tokenizer is tokenizer, "Tokenizer instance mismatch in pipeline"
    assert pipeline.retriever is retriever, "Retriever instance mismatch in pipeline"
