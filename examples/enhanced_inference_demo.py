#!/usr/bin/env python
# examples/enhanced_inference_demo.py

"""
Demo script for the Enhanced MTG Inference Pipeline.

This script demonstrates how to set up and use the EnhancedMTGInferencePipeline
for robust, production-ready inference with features like:
- Multi-expert routing via transaction classifier
- Streaming response generation
- Error handling with circuit breakers
- Multi-turn conversations with context management
- Performance monitoring
- KV cache optimization

Run this script to see the pipeline in action with sample queries.
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.enhanced_pipeline import EnhancedMTGInferencePipeline
from src.models.cross_expert import CrossExpertAttention
from src.data.mtg_data_loader import MTGDataLoader
from src.models.transaction_classifier import TransactionClassifier
from src.knowledge.hybrid_retriever import HybridRetriever  # Corrected import
from src.models.expert_adapters import ExpertAdapterManager  # Corrected import name
from src.utils.kv_cache_manager import KVCacheManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("enhanced_inference_demo")


def create_pipeline(base_model_path: str, enable_monitoring: bool = True):
    """
    Create an instance of the enhanced inference pipeline.

    Args:
        base_model_path: Path to the base model directory
        enable_monitoring: Whether to enable performance monitoring

    Returns:
        An initialized EnhancedMTGInferencePipeline
    """
    logger.info(f"Loading base model from {base_model_path}")

    try:
        # Note: In a real application you would load the actual models
        # Here we're creating placeholder components for demo purposes

        # Set up tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # In demo mode, we'll just check if the model exists
        if not os.path.exists(base_model_path):
            logger.warning(f"Model path {base_model_path} not found.")
            logger.info("Using dummy model and tokenizer for demonstration.")

            # Create dummy components for the demo
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
        else:
            # Load the actual model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            model = AutoModelForCausalLM.from_pretrained(base_model_path)

        # Set up devices
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load auxiliary components
        mtg_data_loader = MTGDataLoader()
        transaction_classifier = TransactionClassifier()
        knowledge_retriever = HybridRetriever()  # Using corrected class name
        expert_manager = ExpertAdapterManager(model)  # Using corrected class name
        cross_expert = CrossExpertAttention(hidden_size=model.config.hidden_size)
        kv_cache_manager = KVCacheManager()

        # Create the pipeline
        pipeline = EnhancedMTGInferencePipeline(
            model=model,
            tokenizer=tokenizer,
            classifier=transaction_classifier,
            retriever=knowledge_retriever,
            data_loader=mtg_data_loader,
            expert_manager=expert_manager,
            cross_expert_attention=cross_expert,
            device=device,
            kv_cache_manager=kv_cache_manager,
            enable_monitoring=enable_monitoring,
            enable_circuit_breakers=True,
        )

        logger.info("Enhanced inference pipeline created successfully")
        return pipeline

    except Exception as e:
        logger.error(f"Error creating pipeline: {str(e)}", exc_info=True)
        raise


def demo_single_query(pipeline):
    """Run a simple single query through the pipeline."""
    logger.info("\n===== SINGLE QUERY DEMO =====")

    query = "How does the cascade rule work in MTG?"
    logger.info(f"Query: {query}")

    # Time the inference
    start_time = time.time()
    result = pipeline.generate_response(
        query=query, max_new_tokens=100, temperature=0.7, use_multiple_experts=True
    )
    end_time = time.time()

    logger.info(f"Response: {result['response']}")
    logger.info(f"Experts used: {result['expert_types']}")
    logger.info(f"Confidence scores: {result['confidences']}")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")

    # Show detailed metrics
    metrics = result.get("metrics", {})
    if metrics:
        logger.info(
            f"Classification time: {metrics.get('classification_time', 0):.2f} seconds"
        )
        logger.info(f"Retrieval time: {metrics.get('retrieval_time', 0):.2f} seconds")
        logger.info(f"Generation time: {metrics.get('generation_time', 0):.2f} seconds")


def demo_conversation(pipeline):
    """Run a multi-turn conversation through the pipeline."""
    logger.info("\n===== CONVERSATION DEMO =====")

    conversation_id = "demo-conversation-1"

    # First turn
    query1 = "What are some powerful board wipes in Standard format?"
    logger.info(f"Turn 1 - Query: {query1}")

    result1 = pipeline.generate_response(
        query=query1,
        max_new_tokens=100,
        temperature=0.7,
        conversation_id=conversation_id,
    )

    logger.info(f"Turn 1 - Response: {result1['response']}")

    # Second turn (follow-up)
    query2 = "Which one works best against creature-heavy aggro decks?"
    logger.info(f"Turn 2 - Query: {query2}")

    result2 = pipeline.generate_response(
        query=query2,
        max_new_tokens=100,
        temperature=0.7,
        conversation_id=conversation_id,
    )

    logger.info(f"Turn 2 - Response: {result2['response']}")

    # Show conversation context
    context = pipeline.conversation_manager.get_context()
    logger.info(
        f"Conversation context now contains {len(pipeline.conversation_manager.history)} turns"
    )


def demo_error_handling(pipeline):
    """Demonstrate error handling and circuit breakers."""
    logger.info("\n===== ERROR HANDLING DEMO =====")

    # Simulate a query that might trigger errors
    query = "How do the rules for layers interact with mutate?"
    logger.info(f"Potentially complex query: {query}")

    # Process normally
    result = pipeline.generate_response(query)
    logger.info(f"Response with normal processing: {result['response'][:100]}...")

    # Show performance monitoring stats if available
    if pipeline.enable_monitoring:
        stats = pipeline.performance_monitor.get_stats()
        logger.info("Performance statistics:")
        for metric, values in stats.items():
            if metric == "expert_distribution":
                logger.info(f"Expert distribution: {values}")
            elif isinstance(values, dict) and "avg" in values:
                logger.info(f"{metric} average: {values['avg']}")


def main():
    """Main entry point for the demo."""
    # Create the pipeline - you would set the path to your model here
    pipeline = create_pipeline(base_model_path="./models/mtg-mixtral-8x7b")

    # Run the demos
    demo_single_query(pipeline)
    demo_conversation(pipeline)
    demo_error_handling(pipeline)

    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
