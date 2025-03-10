"""
Demonstrates integration of the advanced memory profiler with the enhanced MTG inference pipeline.

This example shows:
1. How to integrate memory monitoring with the production inference pipeline
2. Using memory alerts to adaptively manage model parameters
3. Detecting and responding to potential memory leaks
4. Component-level memory tracking for experts and knowledge components
"""

import os
import sys
import time
import logging
import torch
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.memory_profiler import MemoryProfiler, profile_memory

# Pseudocode for importing the pipeline components
# Uncomment these for a real implementation
# from src.inference.enhanced_pipeline import EnhancedMTGInferencePipeline
# from src.models.expert_adapters import ExpertAdapterManager
# from src.knowledge.hybrid_retriever import HybridRetriever
# from src.knowledge.context_assembler import ContextAssembler
# from src.knowledge.knowledge_graph import MTGKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Memory alert handler for dynamic adaptation
def memory_alert_handler(alert):
    """Handle memory alerts by adapting model parameters."""
    if alert.type == "HIGH_UTILIZATION" and alert.gpu_id is not None:
        logger.warning(
            f"Memory alert: {alert.type} on GPU {alert.gpu_id} - {alert.value:.1f}%"
        )
        logger.warning("Adapting parameters to reduce memory usage...")

        # Handle based on which component is causing high memory
        if alert.component is not None:
            logger.warning(f"High memory component: {alert.component}")


class MemoryAwarePipeline:
    """Enhanced inference pipeline with integrated memory monitoring."""

    def __init__(
        self,
        model_id="mistralai/Mixtral-8x7B-v0.1",
        experts_to_load=None,
        knowledge_enabled=True,
        alert_threshold=0.85,
        sampling_interval=1.0,
    ):
        """
        Initialize the memory-aware pipeline.

        Args:
            model_id: Base model identifier
            experts_to_load: List of expert types to load or None for defaults
            knowledge_enabled: Whether to enable knowledge components
            alert_threshold: Memory utilization threshold for alerts (0-1)
            sampling_interval: Memory snapshot interval in seconds
        """
        self.model_id = model_id
        self.experts_to_load = experts_to_load or ["REASON", "EXPLAIN"]
        self.knowledge_enabled = knowledge_enabled

        # Create memory profiler
        self.profiler = MemoryProfiler(
            alert_threshold=alert_threshold,
            sampling_interval=sampling_interval,
            auto_start=False,
        )

        # Register memory alert handler
        self.profiler.register_alert_handler(memory_alert_handler)

        # Components
        self.pipeline = None
        self.knowledge_retriever = None
        self.expert_manager = None

        logger.info(
            "Memory-aware pipeline initialized with alert threshold: "
            f"{alert_threshold*100:.1f}%"
        )

    def initialize(self):
        """
        DEMONSTRATION ONLY: Simulate initializing components with memory monitoring.

        In a real implementation, this method would actually initialize the model,
        expert adapters, knowledge system, and inference pipeline.
        """
        # Start memory monitoring
        self.profiler.start_monitoring()
        logger.info("Memory monitoring started")

        try:
            # 1. Simulate initializing knowledge components if enabled
            if self.knowledge_enabled:
                logger.info("Simulating knowledge components initialization...")

                # Register "demo" components to demonstrate memory tracking
                self.profiler.register_component("knowledge_graph")
                self.profiler.register_component("knowledge_retriever")
                self.profiler.register_component("context_assembler")

            # 2. Simulate initializing expert adapter manager
            logger.info("Simulating expert adapter initialization...")
            self.profiler.register_component("expert_manager")

            # 3. Simulate initializing pipeline
            logger.info("Simulating pipeline initialization...")
            self.profiler.register_component("pipeline")

            # In a real implementation, we would initialize actual components:
            #
            # self.knowledge_graph = MTGKnowledgeGraph()
            # self.knowledge_retriever = HybridRetriever(vector_retriever, self.knowledge_graph)
            # self.context_assembler = ContextAssembler()
            #
            # self.expert_manager = ExpertAdapterManager(
            #    base_model=model,
            #    expert_types=self.experts_to_load
            # )
            #
            # self.pipeline = EnhancedMTGInferencePipeline(
            #     model=model,
            #     tokenizer=tokenizer,
            #     classifier=classifier,
            #     retriever=self.knowledge_retriever,
            #     data_loader=data_loader,
            #     expert_manager=self.expert_manager,
            # )

            # Create some fake memory usage for components
            for component in [
                "knowledge_graph",
                "knowledge_retriever",
                "context_assembler",
                "expert_manager",
                "pipeline",
            ]:
                size_mb = (
                    100 + hash(component) % 900
                )  # Pseudo-random size between 100-1000 MB
                self.profiler.component_memory_map[component]["current_usage"] = (
                    size_mb * 1024 * 1024
                )
                self.profiler.component_memory_map[component]["peak_usage"] = (
                    size_mb * 1.2 * 1024 * 1024
                )

            # Memory usage report after initialization
            memory_summary = self.profiler.get_memory_usage_summary()
            logger.info(f"Initialization complete. Memory usage:")
            logger.info(
                f"  Total GPU: {memory_summary.get('total_gpu_used_mb', 0):.1f}MB / "
                f"{memory_summary.get('total_gpu_available_mb', 0):.1f}MB "
                f"({memory_summary.get('gpu_utilization_percent', 0):.1f}%)"
            )

            # Component breakdown
            component_stats = self.profiler.get_component_memory_stats()
            for name, stats in component_stats.items():
                logger.info(f"  {name}: {stats.get('current_mb', 0):.1f}MB")

            # In simulation, we're initializing a fake pipeline
            self.pipeline = True  # Just a flag to indicate initialization
            return True

        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            # Stop monitoring on error
            self.profiler.stop_monitoring()
            raise

    def process_query(self, query, expert_type=None, max_tokens=512):
        """
        DEMONSTRATION ONLY: Simulate processing a query with memory monitoring.

        In a real implementation, this would pass the query to the actual
        inference pipeline.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        # Log the query processing request
        logger.info(f"Processing query with memory monitoring: '{query[:50]}...'")

        # Use a function to simulate processing and demonstrate memory profiling
        def simulate_query_processing():
            # Simulate some memory allocation during processing
            logger.info("Simulating response generation...")
            tensors = []

            # Simulate memory usage
            for i in range(3):
                # Create a tensor and add it to our list to simulate memory usage
                # In a real implementation, this would be actual model inference
                tensor_size = 50 * (i + 1)  # Increasing tensor sizes
                tensors.append(torch.ones((tensor_size, tensor_size)))
                time.sleep(0.5)  # Give profiler time to detect memory change

            # Simulate expert-specific processing
            if expert_type == "REASON":
                logger.info("Using REASON expert for query")
                # Simulate more intensive computation
                for i in range(2):
                    tensors.append(torch.ones((200, 200)))
                    time.sleep(0.5)  # Simulate processing time

            # Generate a simulated response
            response = f"Simulated response to query about {query.split()[0]}"

            # Cleanup (important for memory management!)
            del tensors

            return response

        # Create a unique ID for this query to appear in logs
        query_id = int(time.time())

        # Process the query with memory profiling
        try:
            # Note: In a real implementation, we would use the context manager:
            # with profile_memory(...): ...
            # But for this demo, we'll simulate it to avoid context manager issues

            logger.info(f"Starting memory profiling for Query_{query_id}")
            # Create a report path for the profile
            report_path = f"logs/memory/query_{query_id}.txt"

            # Simulate beginning memory profiling
            beginning_memory = self.profiler.get_memory_usage_summary()
            logger.info(
                f"Memory before query: {beginning_memory.get('gpu_utilization_percent', 0):.1f}% GPU"
            )

            # Run the actual query processing
            response = simulate_query_processing()

            # Simulate end of profiling context
            ending_memory = self.profiler.get_memory_usage_summary()
            logger.info(
                f"Memory after query: {ending_memory.get('gpu_utilization_percent', 0):.1f}% GPU"
            )

            # Create a basic memory report with start/end statistics
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                f.write(f"Memory Profile for Query_{query_id}\n")
                f.write(
                    f"Start: {beginning_memory.get('gpu_utilization_percent', 0):.1f}% GPU\n"
                )
                f.write(
                    f"End: {ending_memory.get('gpu_utilization_percent', 0):.1f}% GPU\n"
                )

            logger.info(f"Memory profile saved to {report_path}")

            # Check for memory leaks after processing
            leak_result = self.profiler.detect_memory_leaks()
            if leak_result and leak_result.get("detected", False):
                component = leak_result.get("component", "unknown")
                growth_rate = leak_result.get("growth_rate_mb_per_snapshot", 0)
                logger.warning(f"Memory leak detected in component: {component}")
                logger.warning(f"Growth rate: {growth_rate:.1f}MB per snapshot")

            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Error processing query: {str(e)}"

    def shutdown(self):
        """Clean up resources and stop memory monitoring."""
        logger.info("Shutting down memory-aware pipeline...")

        # Save final memory report
        self.profiler.save_memory_report("logs/memory/final_report.txt")

        # Stop the memory profiler
        self.profiler.stop_monitoring()

        # Clean up resources
        self.pipeline = None

        logger.info("Memory-aware pipeline shutdown complete")


def run_demo():
    """Run the memory-aware pipeline demonstration."""
    # Create logs directory
    os.makedirs("logs/memory", exist_ok=True)

    # Initialize pipeline
    memory_pipeline = MemoryAwarePipeline(
        experts_to_load=["REASON", "EXPLAIN"],  # Load just two experts to save memory
        alert_threshold=0.8,
        sampling_interval=0.5,
    )

    try:
        # Initialize components
        memory_pipeline.initialize()

        # Process several example queries
        queries = [
            (
                "What happens if I cast Lightning Bolt on a creature with protection from red?",
                "REASON",
            ),
            ("How does the stack work in Magic?", "EXPLAIN"),
            (
                "If I control Lurrus of the Dream-Den, can I cast permanent spells from my graveyard?",
                None,
            ),
        ]

        for query, expert_type in queries:
            print(f"\nQuery: {query}")
            print(f"Using expert: {expert_type or 'AUTO'}")

            # Process the query with memory monitoring
            response = memory_pipeline.process_query(query, expert_type)
            print(f"Response: {response}")

            # Report memory usage after each query
            stats = memory_pipeline.profiler.get_memory_usage_summary()
            print(
                f"Memory after query: {stats.get('gpu_utilization_percent', 0):.1f}% GPU utilization"
            )

            # Wait a moment to see memory changes
            time.sleep(1)

    finally:
        # Always shutdown properly
        memory_pipeline.shutdown()


if __name__ == "__main__":
    print("Memory-Aware MTG Inference Pipeline Demo")
    print("=======================================\n")

    # Print environment info
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available, using CPU only")
    print()

    # Run the demo
    run_demo()

    print("\nDemonstration completed!")
