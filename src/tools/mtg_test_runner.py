#!/usr/bin/env python
# src/tools/mtg_test_runner.py
"""
MTG Test Runner - A tool for executing test cases for the MTG AI Assistant.

This utility is not a test itself, but a command-line tool for running specified test cases
from a JSON file and generating performance metrics. It's useful for benchmarking, regression
testing, and system verification without requiring the pytest framework.

Usage:
    python -m src.tools.mtg_test_runner --test-cases tests/data/sample_test_cases.json

For complete usage options:
    python -m src.tools.mtg_test_runner --help
"""

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, cast
from transformers import PreTrainedModel

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.model_loader import load_quantized_model
from src.models.expert_adapters import ExpertAdapterManager
from src.models.cross_expert import CrossExpertAttention
from src.data.mtg_data_loader import MTGDataLoader, DocumentType
from src.knowledge.retriever import MTGRetriever
from src.inference.pipeline import MTGInferencePipeline
from src.models.transaction_classifier import TransactionClassifier


def setup_logging():
    """Configure logging for the tool."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_test_cases(file_path: str) -> List[Dict[str, Any]]:
    """Load test cases from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    """Main entry point for the MTG test runner tool."""
    parser = argparse.ArgumentParser(description="MTG AI Assistant Test Runner Tool")
    parser.add_argument(
        "--model", default="mistralai/Mixtral-8x7B-v0.1", help="Model ID"
    )
    parser.add_argument(
        "--quantization", default="4bit", help="Quantization type (4bit, 8bit)"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--adapters-dir", default="adapters", help="Adapters directory")
    parser.add_argument(
        "--test-cases", default="test_cases.json", help="Test cases file"
    )
    parser.add_argument(
        "--output", default="results.json", help="Output file for results"
    )
    parser.add_argument(
        "--multi-expert", action="store_true", help="Use multiple experts"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger("mtg_test_runner")

    # Load components
    logger.info(f"Loading model: {args.model}")
    model, tokenizer = load_quantized_model(
        args.model, quantization_type=args.quantization
    )
    # Cast model to PreTrainedModel to fix type warning
    model = cast(PreTrainedModel, model)

    logger.info("Loading MTG data")
    data_loader = MTGDataLoader(args.data_dir)
    documents = data_loader.load_data()

    logger.info("Initializing retriever")
    retriever = MTGRetriever()

    # Convert documents to the required format for the retriever
    formatted_docs: List[Dict[str, str]] = []
    for doc in documents:
        formatted_doc = {
            "id": doc["id"],
            "type": doc["type"],
            "text": doc["text"],
            "metadata": json.dumps(doc.get("metadata", {})),
        }
        formatted_docs.append(formatted_doc)

    # Index the formatted documents
    retriever.index_documents(formatted_docs)

    logger.info("Initializing classifier")
    classifier = TransactionClassifier()

    logger.info("Initializing expert adapter manager")
    expert_manager = ExpertAdapterManager(model, args.adapters_dir)

    logger.info("Initializing cross-expert attention")
    # Get hidden size from model config if available
    hidden_size = getattr(model.config, "hidden_size", 4096)
    cross_expert = CrossExpertAttention(hidden_size=hidden_size)

    logger.info("Initializing inference pipeline")
    pipeline = MTGInferencePipeline(
        model,
        tokenizer,
        classifier,
        retriever,
        data_loader,
        expert_manager,
        cross_expert,
    )

    # Load test cases
    logger.info(f"Loading test cases from {args.test_cases}")
    test_cases = load_test_cases(args.test_cases)

    # Run tests
    results = []
    for i, test_case in enumerate(test_cases):
        logger.info(
            f"Running test case {i+1}/{len(test_cases)}: {test_case['query'][:50]}..."
        )
        result = pipeline.generate_response(
            test_case["query"], use_multiple_experts=args.multi_expert
        )
        results.append(
            {
                "query": test_case["query"],
                "response": result["response"],
                "expert_types": result["expert_types"],
                "confidences": result["confidences"],
                "metrics": result["metrics"],
            }
        )

    # Save results
    logger.info(f"Saving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
