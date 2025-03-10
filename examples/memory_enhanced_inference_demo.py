#!/usr/bin/env python
# examples/memory_enhanced_inference_demo.py

"""
Demo script for the Enhanced MTG Inference Pipeline with Memory Profiling.

This script demonstrates how to set up the EnhancedMTGInferencePipeline
with the advanced memory profiler for comprehensive monitoring, leak detection,
and memory optimization. Features include:
- Multi-expert routing via transaction classifier
- Component-level memory usage attribution
- Memory leak detection and alerting
- Performance monitoring with memory metrics
- Memory-aware execution with automatically triggered optimizations

Run this script to see the memory-aware pipeline in action with sample queries.
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
from src.knowledge.hybrid_retriever import HybridRetriever
from src.models.expert_adapters import ExpertAdapterManager
from src.utils.kv_cache_manager import KVCacheManager
from src.utils.memory_profiler import MemoryProfiler, profile_memory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("memory_enhanced_inference_demo")

# Set up directory for memory profiles
MEMORY_LOGS_DIR = Path("logs/memory")
MEMORY_LOGS_DIR.mkdir(exist_ok=True, parents=True)


def memory_alert_handler(alert):
    """Handle memory alerts by taking adaptive actions."""
    device = f"GPU {alert.gpu_id}" if alert.gpu_id is not None else "CPU"
    logger.warning(f"Memory alert: {alert.type} on {device} ({alert.value:.1f}%)")

    if alert.recommendations:
        logger.warning("Recommendations:")
        for i, recommendation in enumerate(alert.recommendations, 1):
            logger.warning(f"  {i}. {recommendation}")

    # Implement adaptive actions based on alert type
    if alert.type == "HIGH_UTILIZATION" and alert.value > 90:
        logger.warning("Taking action: Clearing caches and releasing unused tensors")
        # In a real application, this would trigger aggressive memory reclamation


def create_memory_aware_pipeline(base_model_path: str):
    """
    Create an enhanced inference pipeline with memory profiling.

    Args:
        base_model_path: Path to the base model directory

    Returns:
        An initialized EnhancedMTGInferencePipeline with memory monitoring
    """
    # Create memory profiler
    memory_profiler = MemoryProfiler(
        alert_threshold=0.8,  # 80% memory utilization triggers alerts
        leak_detection_window=20,  # Use 20 samples for trend analysis
        sampling_interval=0.5,  # Sample every 0.5 seconds
        auto_start=False,
    )

    # Register alert handler
    memory_profiler.register_alert_handler(memory_alert_handler)

    logger.info("Memory profiler created with alert threshold: 80%")
    logger.info(f"Loading base model from {base_model_path}")

    try:
        # Start memory monitoring
        memory_profiler.start_monitoring()
        logger.info("Memory monitoring started")

        # Note: In a real application you would use the context manager
        # Here we'll demonstrate direct profiling for loading models
        logger.info("Loading model with memory tracking")

        # Import and set up tokenizer/model
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
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path, device_map="auto", torch_dtype=torch.float16
            )

        logger.info("Model loading complete")

        # Set up devices
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load and register components with direct memory tracking
        logger.info("Initializing and registering components")

        # Load auxiliary components
        mtg_data_loader = MTGDataLoader()
        memory_profiler.register_component("mtg_data_loader", mtg_data_loader)

        transaction_classifier = TransactionClassifier()
        memory_profiler.register_component(
            "transaction_classifier", transaction_classifier
        )

        knowledge_retriever = HybridRetriever()
        memory_profiler.register_component("knowledge_retriever", knowledge_retriever)

        expert_manager = ExpertAdapterManager(model)
        memory_profiler.register_component("expert_manager", expert_manager)

        cross_expert = CrossExpertAttention(hidden_size=model.config.hidden_size)
        memory_profiler.register_component("cross_expert", cross_expert)

        kv_cache_manager = KVCacheManager()
        memory_profiler.register_component("kv_cache_manager", kv_cache_manager)

        logger.info("Component initialization complete")

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
            enable_monitoring=True,
            enable_circuit_breakers=True,
        )

        # Register the pipeline itself
        memory_profiler.register_component("pipeline", pipeline)

        # Display memory usage after initialization
        memory_summary = memory_profiler.get_memory_usage_summary()
        logger.info("Memory usage after pipeline initialization:")
        logger.info(
            f"  GPU utilization: {memory_summary.get('gpu_utilization_percent', 0):.1f}%"
        )
        logger.info(
            f"  CPU utilization: {memory_summary.get('cpu_utilization_percent', 0):.1f}%"
        )

        # Show component-level memory usage
        component_stats = memory_profiler.get_component_memory_stats()
        logger.info("Component memory usage:")
        for name, stats in component_stats.items():
            logger.info(f"  {name}: {stats.get('current_mb', 0):.1f}MB")

        # Save memory report
        memory_profiler.save_memory_report(
            str(MEMORY_LOGS_DIR / "pipeline_initialization_report.txt")
        )

        # Store the memory profiler (as a custom attribute for demo purposes)
        # In a production system, we would use a more structured approach
        setattr(pipeline, "memory_profiler", memory_profiler)

        logger.info("Memory-aware enhanced inference pipeline created successfully")
        return pipeline

    except Exception as e:
        # Stop memory monitoring on error
        memory_profiler.stop_monitoring()
        logger.error(f"Error creating pipeline: {str(e)}", exc_info=True)
        raise


def demo_memory_aware_inference(pipeline):
    """Demonstrate memory-aware inference with monitoring and optimization."""
    logger.info("\n===== MEMORY-AWARE INFERENCE DEMO =====")

    # Get memory profiler
    memory_profiler = pipeline.memory_profiler

    query = "Explain how protection from a color works when casting spells."
    logger.info(f"Query: {query}")

    # Run inference with memory profiling
    # Note: In a production environment, we'd use the context manager
    # Here we'll use direct profiling calls for demonstration purposes

    # Start profiling
    logger.info("Starting memory profiling for query processing")

    # Record memory before processing
    before_memory = memory_profiler.get_memory_usage_summary()
    logger.info(
        f"Memory before query: {before_memory.get('gpu_utilization_percent', 0):.1f}% GPU"
    )

    # Process the query
    start_time = time.time()
    result = pipeline.generate_response(
        query=query, max_new_tokens=100, temperature=0.7, use_multiple_experts=True
    )
    end_time = time.time()

    # Record memory after processing
    after_memory = memory_profiler.get_memory_usage_summary()
    logger.info(
        f"Memory after query: {after_memory.get('gpu_utilization_percent', 0):.1f}% GPU"
    )

    # Save report
    report_path = str(MEMORY_LOGS_DIR / "query_processing_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("Memory Profile for Query Processing\n\n")
        f.write(f"Before: {before_memory.get('gpu_utilization_percent', 0):.1f}% GPU\n")
        f.write(f"After: {after_memory.get('gpu_utilization_percent', 0):.1f}% GPU\n")
        f.write(f"Processing time: {end_time - start_time:.2f} seconds\n")

    logger.info(f"Memory profile saved to {report_path}")

    # Display results
    logger.info(f"Response: {result['response']}")
    logger.info(f"Experts used: {result['expert_types']}")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")

    # Check for memory leaks after processing
    leak_result = memory_profiler.detect_memory_leaks()
    if leak_result and leak_result.get("detected", False):
        component = leak_result.get("component", "unknown")
        growth_rate = leak_result.get("growth_rate_mb_per_snapshot", 0)
        logger.warning(f"Memory leak detected in component: {component}")
        logger.warning(f"Growth rate: {growth_rate:.1f}MB per snapshot")

        # In a real application, this would trigger memory reclamation
        logger.info("Demonstrating memory reclamation...")
        torch.cuda.empty_cache()

    # Display memory usage after inference
    memory_summary = memory_profiler.get_memory_usage_summary()
    logger.info("Memory usage after inference:")
    logger.info(
        f"  GPU utilization: {memory_summary.get('gpu_utilization_percent', 0):.1f}%"
    )


def demo_memory_monitoring_under_load(pipeline):
    """Demonstrate memory monitoring under simulated load conditions."""
    logger.info("\n===== MEMORY MONITORING UNDER LOAD =====")

    # Get memory profiler
    memory_profiler = pipeline.memory_profiler

    # Run multiple queries in succession
    queries = [
        "How does the cascade rule work in MTG?",
        "What happens if my commander is exiled?",
        "Explain the difference between hexproof and ward.",
        "How do mutate mechanics work with token creatures?",
        "What is the proper sequence for resolving triggers in a complex stack?",
    ]

    # Track memory over multiple queries
    logger.info(f"Processing {len(queries)} queries in succession...")

    memory_snapshots = []
    for i, query in enumerate(queries, 1):
        logger.info(f"Query {i}: {query}")

        # Process query (direct profiling for demo)
        logger.info(f"Processing query {i} with memory profiling")

        # Process the query
        result = pipeline.generate_response(query=query, max_new_tokens=100)
        logger.info(f"Response {i}: {result['response'][:50]}...")

        # Save simple memory report
        report_path = str(MEMORY_LOGS_DIR / f"query_{i}_report.txt")
        current_memory = memory_profiler.get_memory_usage_summary()

        with open(report_path, "w") as f:
            f.write(f"Memory Profile for Query {i}\n\n")
            f.write(
                f"GPU utilization: {current_memory.get('gpu_utilization_percent', 0):.1f}%\n"
            )

        # Record memory snapshot
        memory_snapshots.append(memory_profiler.get_memory_usage_summary())

        # Force some memory reclamation every other query
        if i % 2 == 0:
            logger.info("Performing memory reclamation...")
            torch.cuda.empty_cache()

    # Show memory trend
    logger.info("Memory utilization trend across queries:")
    for i, snapshot in enumerate(memory_snapshots, 1):
        logger.info(
            f"  Query {i}: {snapshot.get('gpu_utilization_percent', 0):.1f}% GPU"
        )

    # Check for memory leaks after all queries
    leak_result = memory_profiler.detect_memory_leaks()
    if leak_result and leak_result.get("detected", False):
        component = leak_result.get("component", "unknown")
        logger.warning(
            f"Memory leak detected across multiple queries in component: {component}"
        )
    else:
        logger.info("No memory leaks detected across multiple queries")


def demo_component_memory_tracking(pipeline):
    """Demonstrate component-level memory tracking."""
    logger.info("\n===== COMPONENT-LEVEL MEMORY TRACKING =====")

    # Get memory profiler
    memory_profiler = pipeline.memory_profiler

    # Show current component memory
    component_stats = memory_profiler.get_component_memory_stats()
    logger.info("Initial component memory usage:")
    for name, stats in component_stats.items():
        logger.info(
            f"  {name}: {stats.get('current_mb', 0):.1f}MB (peak: {stats.get('peak_mb', 0):.1f}MB)"
        )

    # Run a complex query that exercises multiple components
    logger.info("Processing complex query to exercise multiple components...")
    query = "If I control Aesi, Tyrant of Gyre Strait and play an extra land with Azusa, Lost but Seeking, how many cards do I draw?"

    # Direct profiling for demonstration
    logger.info("Processing complex query with memory profiling")
    pipeline.generate_response(
        query=query, max_new_tokens=150, use_multiple_experts=True
    )

    # Save memory usage snapshot
    report_path = str(MEMORY_LOGS_DIR / "complex_query_report.txt")
    with open(report_path, "w") as f:
        f.write("Memory Profile for Complex Query Processing\n\n")
        f.write("Component memory usage:\n")
        for name, stats in memory_profiler.get_component_memory_stats().items():
            f.write(
                f"  {name}: {stats.get('current_mb', 0):.1f}MB (peak: {stats.get('peak_mb', 0):.1f}MB)\n"
            )

    logger.info(f"Memory profile saved to {report_path}")

    # Show updated component memory
    component_stats = memory_profiler.get_component_memory_stats()
    logger.info("Updated component memory usage:")
    for name, stats in component_stats.items():
        logger.info(
            f"  {name}: {stats.get('current_mb', 0):.1f}MB (peak: {stats.get('peak_mb', 0):.1f}MB)"
        )

    # Show which components used the most memory
    sorted_components = sorted(
        [(name, stats.get("peak_mb", 0)) for name, stats in component_stats.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    logger.info("Components by peak memory usage:")
    for name, peak_mb in sorted_components[:3]:  # Top 3 components
        logger.info(f"  {name}: {peak_mb:.1f}MB")


def shutdown_with_report(pipeline):
    """Gracefully shut down the pipeline and generate final memory report."""
    logger.info("\n===== SHUTDOWN AND FINAL REPORT =====")

    # Get memory profiler
    memory_profiler = pipeline.memory_profiler

    # Generate comprehensive memory report
    logger.info("Generating final memory report...")
    memory_profiler.save_memory_report(
        str(MEMORY_LOGS_DIR / "final_memory_report.txt"),
        include_alerts=True,
        include_recommendations=True,
    )

    # Stop memory monitoring
    memory_profiler.stop_monitoring()
    logger.info("Memory monitoring stopped")

    # Display summary stats
    summary = memory_profiler.get_memory_usage_summary()
    if summary:
        logger.info("Final memory usage summary:")
        logger.info(
            f"  GPU utilization: {summary.get('gpu_utilization_percent', 0):.1f}%"
        )
        logger.info(f"  Total alerts: {len(memory_profiler.alerts)}")

    logger.info("Final memory report saved to logs/memory/final_memory_report.txt")


def main():
    """Main entry point for the memory-aware enhanced inference demo."""
    try:
        # Create the memory-aware pipeline
        pipeline = create_memory_aware_pipeline(
            base_model_path="./models/mtg-mixtral-8x7b"
        )

        # Run the memory-aware demos
        demo_memory_aware_inference(pipeline)
        demo_memory_monitoring_under_load(pipeline)
        demo_component_memory_tracking(pipeline)

        # Clean up and show final report
        shutdown_with_report(pipeline)

        logger.info("\nMemory-enhanced demo completed successfully!")

    except Exception as e:
        logger.error(f"Error in demo: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
