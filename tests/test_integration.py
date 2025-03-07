# tests/test_integration.py
import os
import sys
import json
import logging
import torch
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.model_loader import load_quantized_model
from src.knowledge.retreiver import MTGRetriever
from src.models.transaction_classifier import TransactionClassifier
from src.data.mtg_data_loader import MTGDataLoader
from src.inference.pipeline import MTGInferencePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def test_integration():
    """Test the integration of all components."""
    try:
        logger.info("Starting integration test")

        # Step 1: Load model
        logger.info("Loading model...")
        model, tokenizer = load_quantized_model()

        # Step 2: Load data
        logger.info("Loading MTG data...")
        data_loader = MTGDataLoader()
        documents = data_loader.load_or_download_data()

        # Step 3: Initialize retriever
        logger.info("Initializing retriever...")
        retriever = MTGRetriever()
        retriever.index_documents(documents)

        # Step 4: Initialize classifier
        logger.info("Initializing classifier...")
        classifier = TransactionClassifier()

        # Step 5: Initialize pipeline
        logger.info("Creating inference pipeline...")
        pipeline = MTGInferencePipeline(
            model=model,
            tokenizer=tokenizer,
            classifier=classifier,
            retriever=retriever,
            data_loader=data_loader,
        )

        # Step 6: Test different query types
        test_queries = [
            "What happens when Lightning Bolt targets a creature with protection from red?",
            "Can you explain how the stack works in Magic?",
            "I'm new to Magic. Can you teach me the basics of deckbuilding?",
            "I have a hand with Island, Mountain, Lightning Bolt, and Counterspell. What's my best play against an aggro deck?",
            "I lost a game where my opponent used a board wipe. What could I have done differently?",
        ]

        logger.info("Testing queries...")
        for query in test_queries:
            logger.info(f"Query: {query}")
            result = pipeline.generate_response(query)
            logger.info(
                f"Expert: {result['expert_type']} (Confidence: {result['confidence']:.2f})"
            )
            logger.info(f"Response time: {result['metrics']['total_time']:.2f}s")
            logger.info(f"Response: {result['response'][:200]}...")
            logger.info("---")

        logger.info("Integration test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    test_integration()
