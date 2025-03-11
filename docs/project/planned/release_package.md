# Integrated Professionalization and Hardening Plan

This document outlines the professionalization and hardening activities to be incorporated throughout the Hexcore development process. Rather than being postponed until after all features are complete, these activities will be integrated into each development phase to maintain code quality and reduce technical debt.

## Implementation Philosophy

The professionalization process will follow these key principles:

1. **Incremental Improvement**: Apply code quality improvements progressively alongside feature development
2. **Test-Driven Refactoring**: Create comprehensive tests before refactoring any component
3. **User-Centered Documentation**: Focus documentation on both developer and end-user needs
4. **MTG-Specific Structure**: Organize code and documentation around MTG domain concepts

## Phase 1: Core Quality Foundations

These initial quality improvements will be applied during Phase 1 of the development roadmap:

### 1.1 Error Handling Framework

```python
from typing import Optional, Dict, Any, Union, Tuple

# Define a MTG-specific error hierarchy
class HexcoreError(Exception):
    """Base class for Hexcore-specific exceptions with MTG context."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)

# Domain-specific errors
class RuleProcessingError(HexcoreError):
    """Error in MTG rule application or interpretation."""
    pass

class ExpertAdapterError(HexcoreError):
    """Error related to expert adapters."""
    pass

class KnowledgeRetrievalError(HexcoreError):
    """Error related to MTG knowledge retrieval."""
    pass

class MemoryError(HexcoreError):
    """Error related to memory management."""
    pass

# Example usage with graceful fallback
def safe_rule_processing(rule_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Apply MTG rules with proper error handling."""
    try:
        return rules_engine.process_rule(rule_id, context)
    except Exception as e:
        # Log detailed information
        logger.error(f"Rule processing failed for rule {rule_id}: {str(e)}")

        # Return fallback with error info
        return {
            "result": "error",
            "error_type": "rule_processing",
            "rule_id": rule_id,
            "error_message": str(e),
            "fallback_applied": True,
            "fallback_result": rules_engine.get_default_ruling(rule_id)
        }
```

### 1.2 Type Annotations for Core Components

Apply comprehensive type annotations to all critical path components:

```python
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, TypeVar, cast
import torch
from torch import nn, Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

# Example of properly annotated function
def apply_expert_adapter(
    model: PreTrainedModel,
    expert_type: str,
    adapter_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Apply expert adapter to model.

    Args:
        model: Base language model
        expert_type: Expert type to apply
        adapter_path: Optional path to adapter weights
        device: Optional device to place adapter on

    Returns:
        Tuple of (success, error_message)
    """
    try:
        # Implementation...
        return True, None
    except RuntimeError as e:
        # Handle runtime errors (device issues, memory issues)
        return False, f"Runtime error applying {expert_type} adapter: {str(e)}"
```

### 1.3 MTG-Specific Logging Structure

```python
import logging
from typing import Optional, Dict, Any

class MTGLogger:
    """Logger with MTG-specific structure and context."""

    # Categories specific to MTG domain
    CATEGORIES = {
        "RULES": "Rules application and interpretation",
        "CARDS": "Card information and interactions",
        "EXPERTS": "Expert module operations",
        "MEMORY": "Memory management operations",
        "KNOWLEDGE": "Knowledge retrieval operations"
    }

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize with MTG-specific configuration."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create MTG-specific formatter with category field
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(category)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Add console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        self.logger.addHandler(console)

    def log(self, level: int, category: str, message: str, context: Optional[Dict[str, Any]] = None):
        """Log with MTG-specific category and context."""
        # Validate category is a known MTG category
        if category not in self.CATEGORIES:
            category = "GENERAL"

        # Add extra fields for the formatter
        extra = {"category": category}

        # Format context information if provided
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            message = f"{message} [Context: {context_str}]"

        self.logger.log(level, message, extra=extra)

    # Convenience methods for MTG-specific categories
    def rules(self, message: str, context: Optional[Dict[str, Any]] = None, level: int = logging.INFO):
        """Log rules-related information."""
        self.log(level, "RULES", message, context)

    def cards(self, message: str, context: Optional[Dict[str, Any]] = None, level: int = logging.INFO):
        """Log card-related information."""
        self.log(level, "CARDS", message, context)

    def experts(self, message: str, context: Optional[Dict[str, Any]] = None, level: int = logging.INFO):
        """Log expert-related information."""
        self.log(level, "EXPERTS", message, context)
```

## Phase 2: Incremental Refactoring & Package Structure

These improvements will be integrated during Phase 2 of the development roadmap:

### 2.1 Evolved Package Structure

```text
hexcore/
├── pyproject.toml
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── src/
│   └── hexcore/
│       ├── __init__.py
│       ├── mtg/                 # MTG-specific domain logic
│       │   ├── __init__.py
│       │   ├── rules/           # Rules engine and processing
│       │   ├── cards/           # Card data and interactions
│       │   └── formats/         # Format-specific logic
│       ├── experts/             # Expert module management
│       │   ├── __init__.py
│       │   ├── adapters/        # Expert adapter management
│       │   ├── cross_expert/    # Expert collaboration
│       │   └── transaction/     # Transaction routing
│       ├── knowledge/           # Knowledge integration
│       │   ├── __init__.py
│       │   ├── graph/           # Knowledge graph components
│       │   ├── retrieval/       # Retrieval components
│       │   └── assembly/        # Context assembly
│       ├── inference/           # Inference pipeline
│       ├── training/            # Training components
│       └── utils/               # Utilities
│           ├── __init__.py
│           ├── memory/          # Memory management
│           ├── device/          # Device management
│           └── logging/         # Logging utilities
├── tests/
│   └── ...
└── docs/
    └── ...
```

### 2.2 Test Migration Strategy

For each component being refactored, follow this process:

1. **Create Integration Tests**: Before refactoring, create high-level tests that verify component behavior
2. **Maintain Test Compatibility**: Update tests alongside code changes
3. **Validate Behavior Preservation**: Ensure identical behavior before/after changes

```python
# Example test migration pattern
# Step 1: Create integration test for existing functionality
def test_expert_routing_behavior_before_refactor():
    """Capture existing expert routing behavior before refactoring."""
    # Arrange
    query = "How does Lightning Bolt interact with Hexproof?"
    expected_experts = ["REASON", "EXPLAIN"]

    # Act
    classifier = TransactionClassifier.load_from_pretrained(MODEL_PATH)
    result = classifier.classify(query)
    selected_experts = [expert for expert, conf in result.items() if conf > 0.5]

    # Assert
    assert set(selected_experts) == set(expected_experts)

    # Store result for comparison after refactoring
    with open("test_artifacts/expert_routing_results.json", "w") as f:
        json.dump(result, f, indent=2)

# Step 2: After refactoring, verify identical behavior
def test_expert_routing_behavior_after_refactor():
    """Verify expert routing behavior after refactoring matches previous behavior."""
    # Arrange
    query = "How does Lightning Bolt interact with Hexproof?"
    with open("test_artifacts/expert_routing_results.json", "r") as f:
        expected_results = json.load(f)

    # Act
    # Use refactored classifier with new package structure
    from hexcore.experts.transaction import ExpertRouter
    classifier = ExpertRouter.load_from_pretrained(MODEL_PATH)
    result = classifier.classify(query)

    # Assert
    for expert, expected_conf in expected_results.items():
        assert abs(result.get(expert, 0) - expected_conf) < 0.01
```

### 2.3 Enhanced setup.py with MTG-Specific Metadata

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="hexcore",
    version="0.1.0",
    author="John Morrissey",
    author_email="john@foundryside.dev",
    description="MTG AI Reasoning Assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tachyon-beep/hexcore",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment :: Board Games",  # Added for MTG relevance
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "bitsandbytes>=0.40.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "networkx>=2.8.0",
        "faiss-cpu>=1.7.4",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "flake8",
            "pytest",
            "pytest-cov",
        ],
        "mtg": [  # MTG-specific extras
            "scryfall", # For card data retrieval
            "mtgsdk",   # Alternative MTG data source
        ],
    },
    entry_points={
        "console_scripts": [
            "hexcore=hexcore.cli:main",
            "mtg-analyze=hexcore.mtg.cli:analyze_command",  # MTG-specific command
            "mtg-rules=hexcore.mtg.cli:rules_command",     # MTG-specific command
        ],
    },
)
```

## Phase 3: Production Readiness & Documentation

These improvements will be integrated during Phase 3 of the development roadmap:

### 3.1 MTG-Specific CLI Interface

```python
# hexcore/mtg/cli.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from hexcore.inference import HexcorePipeline
from hexcore.mtg.formats import Format

def analyze_command():
    """CLI command for analyzing MTG scenarios."""
    parser = argparse.ArgumentParser(description="Analyze MTG scenarios and rules interactions")

    # MTG-specific command line arguments
    parser.add_argument("--query", "-q", type=str, help="Question or scenario to analyze")
    parser.add_argument("--format", "-f", type=str, default="standard",
                        choices=["standard", "modern", "legacy", "commander", "limited"],
                        help="MTG format context for the analysis")
    parser.add_argument("--expert", "-e", type=str,
                        choices=["reason", "explain", "teach", "predict", "retrospect"],
                        help="Expert mode to use (default: auto-detect)")
    parser.add_argument("--cards", "-c", type=str, nargs="+",
                        help="Specific cards to consider in the analysis")
    parser.add_argument("--output", "-o", type=str, help="Output file for results")

    args = parser.parse_args()

    # Initialize the pipeline
    pipeline = HexcorePipeline()

    # Convert format string to enum
    mtg_format = Format[args.format.upper()]

    # Prepare additional context
    context = {
        "format": mtg_format.value,
    }

    if args.cards:
        # Fetch card data and add to context
        context["cards"] = fetch_card_data(args.cards)

    # Process the query
    result = pipeline.generate(
        query=args.query,
        expert_type=args.expert.upper() if args.expert else None,
        additional_context=context
    )

    # Output handling
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    else:
        # Pretty print the result
        print("\n" + "=" * 80)
        print(result["response"])
        print("=" * 80 + "\n")
        print(f"Expert(s) used: {', '.join(result['experts_used'])}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Processing time: {result['processing_time_ms']:.0f}ms")

def rules_command():
    """CLI command for MTG rules lookup and interpretation."""
    # Implementation similar to analyze_command
    pass

def fetch_card_data(card_names: List[str]) -> List[Dict[str, Any]]:
    """Fetch card data from Scryfall or local database."""
    # Implementation to retrieve card data
    pass
```

### 3.2 Comprehensive Documentation Structure

Create an expanded documentation structure with MTG-specific organization:

```text
docs/
├── index.md                  # Main documentation page
├── installation.md           # Installation instructions
├── quickstart.md             # Getting started guide
├── api/                      # API reference
│   ├── pipeline.md           # Inference pipeline documentation
│   ├── experts.md            # Expert documentation
│   ├── knowledge.md          # Knowledge system documentation
│   └── training.md           # Training documentation
├── mtg/                      # MTG-specific documentation
│   ├── reasoning_modes.md    # Guide to reasoning modes (REASON, EXPLAIN, etc.)
│   ├── rules_handling.md     # How to use the rules interpretation features
│   ├── card_interactions.md  # Documentation on card interaction analysis
│   └── formats.md            # Format-specific analysis features
├── guides/                   # User guides
│   ├── memory_optimization.md # Memory optimization guide
│   ├── custom_adapters.md    # Creating custom adapters
│   └── knowledge_extension.md # Extending the knowledge system
└── development/              # Developer documentation
    ├── architecture.md       # System architecture
    ├── contributing.md       # Contribution guidelines
    └── testing.md            # Testing guide
```

### 3.3 Performance Optimization Documentation

Create detailed documentation on performance optimization:

````markdown
# MTG AI Assistant Performance Optimization Guide

This guide provides recommendations for optimizing the Hexcore system for various hardware configurations, with a focus on memory management and inference performance.

## Hardware Configurations

### Recommended Setup

- **GPU**: 2x GPUs with 16GB VRAM each
- **CPU**: 8+ cores
- **RAM**: 32GB+
- **Disk**: SSD with 20GB+ free space

### Minimum Requirements

- **GPU**: 1x GPU with 16GB VRAM
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Disk**: 10GB free space

## Memory Optimization

### Quantization Settings

The system supports different quantization levels that trade off precision for memory efficiency:

| Quantization | VRAM Usage | Quality Impact | Use Case                         |
| ------------ | ---------- | -------------- | -------------------------------- |
| None (FP16)  | ~30GB      | None           | High-end setups with 40GB+ VRAM  |
| 8-bit        | ~15GB      | Minimal        | Good balance for 24GB setups     |
| 4-bit        | ~8GB       | Moderate       | Memory-constrained setups (16GB) |

### Expert Loading Strategies

Choose the expert loading strategy based on your needs:

| Strategy    | Description                         | Memory Usage | Performance      |
| ----------- | ----------------------------------- | ------------ | ---------------- |
| All Experts | Load all expert adapters at startup | Higher       | Faster switching |
| On-Demand   | Load experts as needed              | Lower        | Slower first use |
| Hybrid      | Keep frequently used experts loaded | Medium       | Balanced         |

Configure your expert loading strategy in `config.yaml`:

```yaml
expert_loading:
  strategy: "hybrid" # "all", "on_demand", or "hybrid"
  preload_experts: ["REASON", "EXPLAIN"] # Always keep these loaded
  max_cached_experts: 3 # Maximum number of inactive experts to keep in memory
```
````

## CPU Offloading

For systems with limited GPU memory but ample CPU RAM, enable CPU offloading:

```yaml
memory_management:
  enable_cpu_offloading: true
  offload_expert_threshold: 0.2 # Offload experts with usage probability < 20%
```

## MTG-Specific Optimizations

### Knowledge Retrieval Budget

Adjust the knowledge retrieval budget based on the complexity of MTG scenarios:

| Scenario Complexity | Latency Budget (ms) | Token Budget | Use Case              |
| ------------------- | ------------------- | ------------ | --------------------- |
| Simple              | 100                 | 512          | Basic rules questions |
| Medium              | 250                 | 1024         | Card interactions     |
| Complex             | 500                 | 2048         | Multi-card combos     |

Configure in `knowledge_config.yaml`:

```yaml
retrieval:
  complexity_detection: true # Auto-detect scenario complexity
  default_latency_budget_ms: 250
  max_tokens: 1024
```

```

## Relationship to Development Roadmap

The professionalization and hardening activities integrate directly with the three-phase development roadmap:

1. **Phase 1 (Core Completion)**: Implement error handling, type annotations, and logging
2. **Phase 2 (Feature Extensions)**: Refactor package structure, migrate tests, and enhance metadata
3. **Phase 3 (Performance & Distribution)**: Create user interfaces, documentation, and performance guides

This integrated approach ensures that code quality and professionalization efforts progress alongside feature development, avoiding technical debt and enabling more efficient collaboration.
```
