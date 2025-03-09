"""
Integration tests for the MTG AI Assistant.

This package contains integration tests that verify the end-to-end functionality
of the MTG AI Assistant system, focusing on multi-component interactions.
"""

import pytest
import sys
import torch
from pathlib import Path
from typing import List, Dict, Union, Any, cast
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_loader import load_quantized_model
from src.knowledge.retriever import MTGRetriever
from src.models.transaction_classifier import TransactionClassifier
from src.data.mtg_data_loader import MTGDataLoader
from src.inference.pipeline import MTGInferencePipeline
from src.models.expert_adapters import ExpertAdapterManager
from src.models.cross_expert import CrossExpertAttention
from src.utils.device_mapping import DeviceMapper


# Define shared fixture that all integration tests can use
@pytest.fixture(scope="module")
def prepared_pipeline():
    """
    Setup the complete pipeline for testing.

    This fixture is scoped at module level to avoid repeated loading of the
    heavy model for each test function.
    """

    # Set up device mapping for dual GPU
    print("Setting up device mapping for dual GPU...")
    device_mapper = DeviceMapper(num_experts=8, num_layers=32)

    # For tests, use a proper device mapping to split across both GPUs
    print(
        f"Found {torch.cuda.device_count()} GPUs, configuring device mapping for testing"
    )

    # Get device map that explicitly assigns components to specific GPUs
    # with quantization-specific optimizations
    quantization_bits = 4  # Start with 4-bit quantization

    # Explicitly use float16 for test environments
    print("Loading model and tokenizer with simplified config for testing...")
    try:
        # First attempt: Try with 4-bit quantization - use our custom device mapping
        print("Attempting to load with 4-bit quantization and custom device mapping...")
        # Create an explicit device map with our custom mapper
        device_map_for_loader = device_mapper.create_mixtral_device_map(
            quantization_bits=4  # Match the desired quantization
        )
        print(f"Created custom balanced device map for 4-bit quantization")

        model, tokenizer = load_quantized_model(
            device_map=device_map_for_loader,  # Use our optimized map directly
            quantization_type="4bit",
            compute_dtype=torch.float16,
            use_safetensors=True,
        )
    except (ValueError, RuntimeError) as e:
        # Clear CUDA cache before retrying with 8-bit
        torch.cuda.empty_cache()
        print(f"4-bit quantization failed: {e}, falling back to 8-bit quantization")

        try:
            # Try with 8-bit quantization next - more memory but more compatible
            # Use custom device mapping for 8-bit as well
            device_map_for_loader = device_mapper.create_mixtral_device_map(
                quantization_bits=8  # Match the quantization type
            )
            print(f"Created custom balanced device map for 8-bit quantization")

            model, tokenizer = load_quantized_model(
                device_map=device_map_for_loader,  # Use our optimized map directly
                quantization_type="8bit",  # Fall back to 8-bit
                use_safetensors=True,
            )
        except RuntimeError as e2:
            # If that also fails, try with reduced reserve memory
            torch.cuda.empty_cache()
            print(
                f"8-bit quantization failed: {e2}, trying with reduced reserve memory"
            )

            # Create a custom device map with minimal reserved memory
            if torch.cuda.device_count() >= 2:
                device_map_for_loader = device_mapper.create_mixtral_device_map(
                    quantization_bits=8
                )
                print("Created custom device map with reduced memory reservation")
            else:
                device_map_for_loader = "auto"

            model, tokenizer = load_quantized_model(
                device_map=device_map_for_loader,
                quantization_type="8bit",
                use_safetensors=True,
            )

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
        import json

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

    # Initialize cross-expert attention and ensure it's on CUDA
    print("Initializing cross-expert attention...")
    cross_expert = CrossExpertAttention(hidden_size=model.config.hidden_size)
    # Move cross-expert attention module to the same device as the embedding layer
    # This ensures it's on CUDA, not CPU
    cross_expert = cross_expert.to("cuda:0")

    # Construct inference pipeline
    print("Building inference pipeline...")
    pipeline = MTGInferencePipeline(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
        retriever=retriever,
        data_loader=data_loader,
        expert_manager=expert_manager,
        cross_expert_attention=cross_expert,
    )
    assert hasattr(
        pipeline, "generate_response"
    ), "Pipeline missing required 'generate_response' method"

    # Return the prepared pipeline
    return pipeline
