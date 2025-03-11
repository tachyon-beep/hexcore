#!/usr/bin/env python
"""
Advanced Reasoning System Integration with LLM Inference

This script demonstrates a direct integration between the advanced reasoning system
and the actual model inference pipeline, using the real MTG AI model.

Key features:
- Loads the actual Mixtral model with proper quantization
- Integrates the reasoning selector to choose appropriate reasoning methods
- Enhances prompts with structured reasoning templates
- Demonstrates knowledge integration into reasoning
- Uses the full inference pipeline with multiple expert adapters
"""

import os
import sys
import time
import torch
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the reasoning components
from src.inference.reasoning_selector import ReasoningModeSelector
from src.inference.reasoning_factory import create_reasoning

# Import the model loading utilities
from src.models.model_loader import load_quantized_model
from src.inference.enhanced_pipeline import EnhancedMTGInferencePipeline
from src.models.transaction_classifier import TransactionClassifier
from src.data.mtg_data_loader import MTGDataLoader
from src.models.expert_adapters import ExpertAdapterManager
from src.models.cross_expert import CrossExpertAttention
from src.knowledge.hybrid_retriever import HybridRetriever
from src.utils.kv_cache_manager import KVCacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_inference_with_reasoning():
    """Run the full inference pipeline with reasoning enhancements."""
    logger.info("Initializing advanced reasoning integration with model inference...")

    # Sample MTG questions with different reasoning needs
    test_queries = [
        {
            "query": "How does the stack work when multiple triggered abilities resolve at the same time?",
            "expert_type": "REASON",
            "description": "Stack mechanics query - Chain-of-Thought reasoning",
        },
        {
            "query": "What's the probability of drawing Lightning Bolt in my opening hand if I have 4 copies in a 60 card deck?",
            "expert_type": "PREDICT",
            "description": "Probability calculation - MCTS reasoning",
        },
        {
            "query": "In a tournament, if my opponent controls Teferi, Time Raveler and I cast an instant during my main phase, can they respond with a counterspell?",
            "expert_type": "REASON",
            "description": "Complex tournament ruling - R1-style reasoning",
        },
    ]

    # Step 1: Load the model and tokenizer using proper quantization
    try:
        logger.info("Loading Mixtral model with quantization...")
        model_id = "mistralai/Mixtral-8x7B-v0.1"  # Default model ID

        # Check for a local model path (for testing without downloading)
        local_path = os.environ.get("MTG_MODEL_PATH", "./models/mtg-mixtral-8x7b")
        if os.path.exists(local_path):
            logger.info(f"Using local model path: {local_path}")
            model_id = local_path

        # Properly load model with quantization settings
        model, tokenizer = load_quantized_model(
            model_id=model_id,
            quantization_type="4bit",  # 4-bit quantization for efficiency
            compute_dtype=torch.bfloat16,
            use_memory_optimizations=True,
        )
        logger.info(f"Successfully loaded {model.__class__.__name__} model")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Running in reasoning-only mode without model...")
        demo_reasoning_only()
        return

    # Step 2: Initialize auxiliary components
    try:
        # Create necessary pipeline components
        transaction_classifier = TransactionClassifier()
        data_loader = MTGDataLoader()
        expert_manager = ExpertAdapterManager(
            base_model=model
        )  # Pass the model to the expert manager
        knowledge_retriever = HybridRetriever()

        # Create optional optimization components
        kv_cache_manager = KVCacheManager()
        cross_expert = CrossExpertAttention(hidden_size=model.config.hidden_size)

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        logger.info("Running in reasoning-only mode without components...")
        demo_reasoning_only()
        return

    # Step 3: Initialize the enhanced inference pipeline
    try:
        pipeline = EnhancedMTGInferencePipeline(
            model=model,
            tokenizer=tokenizer,
            classifier=transaction_classifier,
            retriever=knowledge_retriever,
            data_loader=data_loader,
            expert_manager=expert_manager,
            cross_expert_attention=cross_expert,
            device=device,
            kv_cache_manager=kv_cache_manager,
            enable_monitoring=True,
        )
        logger.info("Enhanced inference pipeline initialized successfully")

    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        logger.info("Running in reasoning-only mode without pipeline...")
        demo_reasoning_only()
        return

    # Step 4: Initialize the reasoning components
    selector = ReasoningModeSelector()

    # Process each test query
    for i, query_data in enumerate(test_queries):
        query = query_data["query"]
        expert_type = query_data["expert_type"]
        description = query_data["description"]

        logger.info(f"\n\n=== Query {i+1}: {query} ===")
        logger.info(f"Expert Type: {expert_type}")
        logger.info(f"Description: {description}")

        # Step 5: Select the appropriate reasoning mode
        confidence_score = 0.85  # In a real system, this comes from the classifier
        selected_mode, config = selector.select_reasoning_mode(
            query, expert_type, confidence_score
        )
        logger.info(f"Selected reasoning mode: {selected_mode}")
        logger.info(f"Reasoning configuration: {config}")

        # Step 6: Create the reasoning implementation
        reasoning = create_reasoning(selected_mode)

        # Step 7: Create knowledge context
        knowledge_context = create_sample_knowledge(query)
        logger.info(f"Created knowledge context for query")

        # Step 8: Create initial inputs for reasoning
        inputs = {
            "prompt": f"Query: {query}\n\nPlease analyze this Magic: The Gathering question."
        }

        # Step 9: Apply reasoning to enhance the inputs
        logger.info("Applying reasoning enhancement to prompt...")
        enhanced_inputs = reasoning.apply(query, inputs, knowledge_context, config)

        # Display the enhanced prompt (truncated for brevity)
        enhanced_prompt = enhanced_inputs.get("prompt", "")
        logger.info(f"Enhanced prompt (truncated):\n{enhanced_prompt[:500]}...\n")

        # Step 10: Generate response using the pipeline
        logger.info("Generating response using model inference pipeline...")

        # Use the generate_response method for direct inference
        try:
            # Time the inference
            start_time = time.time()

            response_data = pipeline.generate_response(
                query=enhanced_prompt,
                max_new_tokens=512,
                temperature=0.7,
                use_multiple_experts=True,
            )

            end_time = time.time()
            generation_time = end_time - start_time

            # Extract and display the response
            if isinstance(response_data, dict) and "response" in response_data:
                logger.info(
                    f"\nGenerated response: {response_data['response'][:500]}..."
                )
                logger.info(f"Generation took {generation_time:.2f} seconds")

            # Show additional information if available
            if isinstance(response_data, dict):
                if "expert_types" in response_data:
                    logger.info(f"Experts used: {response_data['expert_types']}")
                if "confidences" in response_data:
                    logger.info(f"Confidence scores: {response_data['confidences']}")
                if "metrics" in response_data:
                    logger.info(f"Performance metrics: {response_data['metrics']}")

        except Exception as e:
            logger.error(f"Error during inference: {e}")

        logger.info(f"Processing of query {i+1} complete")

    logger.info("\nAll queries processed successfully!")


def demo_reasoning_only():
    """
    Demonstrate the reasoning system without model integration.
    This runs when the model can't be loaded but the reasoning system can still be shown.
    """
    logger.info("\nDemonstrating reasoning system only (without model inference)...")

    # Sample MTG questions with different reasoning needs
    test_queries = [
        {
            "query": "How does the stack work when multiple triggered abilities resolve at the same time?",
            "expert_type": "REASON",
            "description": "Stack mechanics query - Chain-of-Thought reasoning",
        },
        {
            "query": "What's the probability of drawing Lightning Bolt in my opening hand if I have 4 copies in a 60 card deck?",
            "expert_type": "PREDICT",
            "description": "Probability calculation - MCTS reasoning",
        },
    ]

    # Initialize reasoning selector
    selector = ReasoningModeSelector()

    # Process each test query
    for i, query_data in enumerate(test_queries):
        query = query_data["query"]
        expert_type = query_data["expert_type"]
        description = query_data["description"]

        logger.info(f"\n\n=== Query {i+1}: {query} ===")
        logger.info(f"Expert Type: {expert_type}")
        logger.info(f"Description: {description}")

        # Select reasoning mode
        confidence_score = 0.85
        selected_mode, config = selector.select_reasoning_mode(
            query, expert_type, confidence_score
        )

        logger.info(f"Selected reasoning mode: {selected_mode}")
        logger.info(f"Reasoning configuration: {config}")

        # Create the reasoning implementation
        reasoning = create_reasoning(selected_mode)

        # Create knowledge context
        knowledge_context = create_sample_knowledge(query)

        # Create initial inputs
        inputs = {
            "prompt": f"Query: {query}\n\nPlease analyze this Magic: The Gathering question."
        }

        # Apply reasoning to enhance inputs
        enhanced_inputs = reasoning.apply(query, inputs, knowledge_context, config)

        # Display the enhanced prompt (truncated for brevity)
        enhanced_prompt = enhanced_inputs.get("prompt", "")
        logger.info(f"Enhanced prompt (truncated):\n{enhanced_prompt[:500]}...\n")

        logger.info(f"Processing of query {i+1} complete")

    logger.info("\nReasoning demonstration complete!")


def create_sample_knowledge(query: str) -> Dict[str, Any]:
    """
    Create appropriate sample knowledge context based on query content.

    Args:
        query: The user query

    Returns:
        Dictionary with relevant knowledge context
    """
    if "stack" in query.lower():
        return {
            "kg_data": {
                "type": "rule_data",
                "data": [
                    {
                        "id": "405.1",
                        "text": "When a spell is cast, it goes on top of the stack. When an ability is activated or triggers, it goes on top of the stack.",
                    },
                    {
                        "id": "405.2",
                        "text": "The stack resolves one object at a time, always taking the top object.",
                    },
                    {
                        "id": "603.2",
                        "text": "Whenever a game event or game state matches a triggered ability's trigger event, that ability automatically triggers. The ability doesn't do anything at this point.",
                    },
                    {
                        "id": "603.3",
                        "text": "Once an ability has triggered, its controller puts it on the stack as an object that's not a card the next time a player would receive priority.",
                    },
                ],
            }
        }
    elif "Teferi" in query:
        return {
            "kg_data": {
                "type": "card_data",
                "data": [
                    {
                        "name": "Teferi, Time Raveler",
                        "card_types": ["Planeswalker"],
                        "text": "Each opponent can cast spells only any time they could cast a sorcery.\n+1: Until your next turn, you may cast sorcery spells as though they had flash.\n−2: Return up to one target artifact, creature, or enchantment to its owner's hand. Draw a card.",
                        "loyalty": 4,
                    },
                    {
                        "name": "Counterspell",
                        "card_types": ["Instant"],
                        "text": "Counter target spell.",
                    },
                ],
            }
        }
    elif "probability" in query.lower() or "opening hand" in query.lower():
        return {
            "kg_data": {
                "type": "card_data",
                "data": [
                    {
                        "name": "Lightning Bolt",
                        "card_types": ["Instant"],
                        "text": "Lightning Bolt deals 3 damage to any target.",
                    }
                ],
            },
            "rag_data": [
                {
                    "text": "The probability of drawing at least one copy of a card in an opening hand can be calculated using the hypergeometric distribution: P(X ≥ 1) = 1 - P(X = 0) = 1 - C(deck_size-copies, hand_size) / C(deck_size, hand_size)"
                }
            ],
        }
    else:
        return {}


if __name__ == "__main__":
    run_inference_with_reasoning()
