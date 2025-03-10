# Implementation Plan for Professionalization and Hardening

NOT TO BE IMPLEMENTED UNTIL ALL OTHER PLANNED WORK PACKAGES ARE COMPLETED.

## 1. Code Consolidation and Cleanup

First, let's identify and eliminate duplicate or unnecessary code:

```python
def consolidate_codebase():
    """
    Steps to consolidate the codebase:
    1. Identify and remove duplicate functionality
    2. Standardize interfaces between components
    3. Create clear boundaries between modules
    """
    # Steps for implementation:

    # 1. Analyze current codebase structure
    # - Map dependencies between modules
    # - Identify overlapping functionality

    # 2. Consolidate overlapping components
    # - Merge similar utility functions
    # - Create standardized interfaces
    # - Update import references

    # 3. Remove deprecated or unused code
    # - Run code coverage analysis
    # - Identify dead code paths
    # - Document removal decisions
```

### Implementation Strategy

Let's look at specific parts of your codebase to consolidate:

1. **Pipeline Consolidation**:

   - You have both `pipeline.py` and `enhanced_pipeline.py` in your inference directory
   - Create a single unified pipeline that includes all needed features

2. **Utility Function Consolidation**:
   - Create a centralized utilities module
   - Move common functions (device management, tensor operations, etc.)

Here's an example structure for the consolidated codebase:

```
src/
├── __init__.py
├── data/              # Data loading and management
├── inference/         # Single unified inference pipeline
├── knowledge/         # Knowledge retrieval system
├── models/            # Model definitions and expert management
├── training/          # Training pipeline
└── utils/             # Centralized utilities
    ├── __init__.py
    ├── device.py      # Device management and optimization
    ├── logging.py     # Standardized logging
    ├── memory.py      # Memory management
    ├── tensor.py      # Tensor operations
    └── validation.py  # Input and output validation
```

## 2. Standardized Pipeline Implementation

Let's create a single, well-documented pipeline that combines the best aspects of your current implementations:

```python
class HexcorePipeline:
    """
    Unified inference pipeline for the Hexcore MTG AI system.

    This pipeline integrates all necessary components:
    - Expert routing via transaction classification
    - Memory-optimized model execution
    - Knowledge retrieval and integration
    - Cross-expert collaboration
    - Optimized generation with KV caching

    Attributes:
        model (PreTrainedModel): Base language model
        tokenizer (PreTrainedTokenizer): Model tokenizer
        expert_manager (ExpertAdapterManager): Manages expert adapters
        knowledge_retriever (HybridRetriever): Retrieves MTG knowledge
        classifier (TransactionClassifier): Routes queries to experts
        cross_expert (CrossExpertAttention): Handles expert collaboration
        kv_cache_manager (KVCacheManager): Manages KV cache
    """

    def __init__(
        self,
        model_path="mistralai/Mixtral-8x7B-v0.1",
        expert_adapters_dir="adapters",
        device_map="auto",
        quantization_type="4bit",
        max_memory=None,
        knowledge_db_path=None,
        enable_memory_optimization=True,
    ):
        """
        Initialize the Hexcore pipeline.

        Args:
            model_path: Path to base model
            expert_adapters_dir: Directory with expert adapters
            device_map: Device mapping strategy
            quantization_type: Type of quantization (4bit or 8bit)
            max_memory: Maximum memory allocation
            knowledge_db_path: Path to knowledge database
            enable_memory_optimization: Whether to enable memory optimization
        """
        # Implementation details...

    def generate(
        self,
        query,
        max_new_tokens=1024,
        temperature=0.7,
        use_multiple_experts=True,
        latency_budget_ms=None,
        memory_budget_mb=None,
        **generation_kwargs
    ):
        """
        Generate a response to the query.

        Args:
            query: User query text
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            use_multiple_experts: Whether to use multiple experts
            latency_budget_ms: Maximum latency in milliseconds
            memory_budget_mb: Maximum memory usage in MB
            **generation_kwargs: Additional kwargs for generation

        Returns:
            Dictionary with response and metadata
        """
        # Implementation details...
```

This unified pipeline should have thorough documentation and clear interfaces. The implementation should enforce constraints (device compatibility, memory limits) and provide helpful error messages.

## 3. Type Annotations and Error Handling

Let's standardize type annotations and error handling throughout the codebase:

```python
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, TypeVar, cast

# Example of properly annotated function
def apply_expert_adapter(
    model: "PreTrainedModel",
    expert_type: str,
    adapter_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Apply expert adapter to model.

    Args:
        model: Base language model
        expert_type: Type of expert to apply
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
    except ValueError as e:
        # Handle value errors (invalid expert type, missing adapter)
        return False, f"Invalid input for {expert_type} adapter: {str(e)}"
    except Exception as e:
        # Fallback for other errors
        return False, f"Unexpected error applying {expert_type} adapter: {str(e)}"
```

### Standardized Error Handling

Create a consistent error handling approach:

```python
class HexcoreError(Exception):
    """Base class for Hexcore-specific exceptions."""
    pass

class ExpertAdapterError(HexcoreError):
    """Error related to expert adapters."""
    pass

class KnowledgeRetrievalError(HexcoreError):
    """Error related to knowledge retrieval."""
    pass

class MemoryError(HexcoreError):
    """Error related to memory management."""
    pass

class DeviceMappingError(HexcoreError):
    """Error related to device mapping."""
    pass

# Example usage
def safe_knowledge_retrieval(query: str) -> Dict[str, Any]:
    """Safely retrieve knowledge with proper error handling."""
    try:
        return retriever.retrieve(query)
    except Exception as e:
        # Graceful fallback
        logger.error(f"Knowledge retrieval failed: {str(e)}")
        # Return empty result with error info
        return {
            "results": [],
            "error": str(e),
            "error_type": type(e).__name__,
            "fallback_applied": True
        }
```

## 4. Package Structure and Requirements

Let's update the package structure to follow best practices:

### setup.py Improvements

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

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
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "flake8",
            "pytest",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "hexcore=hexcore.cli:main",
        ],
    },
)
```

### requirements.txt Updates

```
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
bitsandbytes>=0.40.0
accelerate>=0.20.0
peft>=0.4.0
networkx>=2.8.0
faiss-cpu>=1.7.4
numpy>=1.24.0
tqdm>=4.65.0
pandas>=2.0.0
scikit-learn>=1.2.0

# Optional dependencies
sentencepiece>=0.1.99
datasets>=2.12.0
wandb>=0.15.4

# Web server and API
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0

# Development and testing
pytest>=7.3.1
black>=23.3.0
isort>=5.12.0
mypy>=1.3.0
flake8>=6.0.0
```

### Package Structure

Reorganize to create a proper importable package:

```
hexcore/
├── pyproject.toml
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── src/
│   └── hexcore/
│       ├── __init__.py
│       ├── data/
│       ├── inference/
│       ├── knowledge/
│       ├── models/
│       ├── training/
│       └── utils/
├── tests/
│   └── ...
├── scripts/
│   └── ...
└── docs/
    └── ...
```

Add version tracking:

```python
# src/hexcore/__init__.py
__version__ = "0.1.0"
```

## 5. Code Quality Tools Integration

Let's integrate standard Python code quality tools:

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[tool.black]
line-length = 110
target-version = ['py38', 'py39', 'py310']
include = '\.pyi?$'
exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 110
multi_line_output = 3

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true

[[tool.mypy.overrides]]
module = [
    "transformers.*",
    "torch.*",
    "networkx.*",
    "faiss.*",
    "tqdm.*"
]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
addopts = "--cov=hexcore"
```

### Linting Scripts

Create a script to run all linting tools:

```python
# scripts/lint.py
#!/usr/bin/env python3
"""Run all linters on the codebase."""
import subprocess
import sys
from pathlib import Path

# Find project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src" / "hexcore"
TESTS_DIR = PROJECT_ROOT / "tests"

def run_command(cmd, description):
    """Run a command and print its output."""
    print(f"Running {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ {description} passed")
        if result.stdout.strip():
            print(result.stdout)
    else:
        print(f"❌ {description} failed")
        print(result.stdout)
        print(result.stderr)
        return False

    return True

def main():
    """Run all linters."""
    success = True

    # Run black
    success &= run_command(
        ["black", "--check", SRC_DIR, TESTS_DIR],
        "black code formatting"
    )

    # Run isort
    success &= run_command(
        ["isort", "--check", SRC_DIR, TESTS_DIR],
        "isort import sorting"
    )

    # Run flake8
    success &= run_command(
        ["flake8", SRC_DIR, TESTS_DIR],
        "flake8 linting"
    )

    # Run mypy
    success &= run_command(
        ["mypy", SRC_DIR],
        "mypy type checking"
    )

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## 6. Documentation Generation

Let's implement comprehensive documentation:

### Documentation Structure

```
docs/
├── index.md                  # Main documentation page
├── installation.md           # Installation instructions
├── quickstart.md             # Getting started guide
├── api/                      # API reference
│   ├── models.md             # Model documentation
│   ├── knowledge.md          # Knowledge system documentation
│   └── inference.md          # Inference pipeline documentation
├── guides/                   # User guides
│   ├── expert_modes.md       # Guide to expert modes
│   ├── memory_optimization.md # Memory optimization guide
│   └── custom_adapters.md    # Creating custom adapters
└── development/              # Developer documentation
    ├── architecture.md       # System architecture
    ├── contributing.md       # Contribution guidelines
    └── testing.md            # Testing guide
```

### Documentation Generation with MkDocs

Install MkDocs and set up automatic documentation generation:

```yaml
# mkdocs.yml
site_name: Hexcore Documentation
site_description: Documentation for the MTG AI Reasoning Assistant
site_author: Vren, the Relentless
repo_url: https://github.com/tachyon-beep/hexcore

theme:
  name: material
  palette:
    primary: indigo
    accent: purple
  features:
    - navigation.tabs
    - navigation.sections
    - toc.integrate
    - search.suggest
    - search.highlight

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.inlinehilite
  - pymdownx.tabbed
  - pymdownx.arithmatex
  - admonition
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Installation: installation.md
  - Quickstart: quickstart.md
  - API Reference:
      - Models: api/models.md
      - Knowledge: api/knowledge.md
      - Inference: api/inference.md
  - User Guides:
      - Expert Modes: guides/expert_modes.md
      - Memory Optimization: guides/memory_optimization.md
      - Custom Adapters: guides/custom_adapters.md
  - Development:
      - Architecture: development/architecture.md
      - Contributing: development/contributing.md
      - Testing: development/testing.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          selection:
            inherited_members: true
          rendering:
            show_source: true
            heading_level: 3
```

### Generate API Documentation Automatically

Create a script to generate API documentation from docstrings:

````python
# scripts/generate_docs.py
#!/usr/bin/env python3
"""Generate API documentation from code."""
import os
import importlib
import inspect
from pathlib import Path

# Find project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
SRC_DIR = PROJECT_ROOT / "src" / "hexcore"
DOCS_DIR = PROJECT_ROOT / "docs" / "api"

# Ensure docs directory exists
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def generate_module_docs(module_name, output_path):
    """Generate documentation for a module."""
    module = importlib.import_module(module_name)

    with open(output_path, "w") as f:
        f.write(f"# {module_name}\n\n")

        # Document module docstring
        if module.__doc__:
            f.write(f"{module.__doc__.strip()}\n\n")

        # Find all classes and functions
        for name, obj in inspect.getmembers(module):
            # Skip private members
            if name.startswith("_"):
                continue

            # Document classes
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                f.write(f"## {name}\n\n")

                # Class docstring
                if obj.__doc__:
                    f.write(f"{obj.__doc__.strip()}\n\n")

                f.write("```python\n")
                f.write(inspect.getsource(obj))
                f.write("```\n\n")

            # Document functions
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                f.write(f"## {name}\n\n")

                # Function docstring
                if obj.__doc__:
                    f.write(f"{obj.__doc__.strip()}\n\n")

                f.write("```python\n")
                f.write(inspect.getsource(obj))
                f.write("```\n\n")

def main():
    """Generate all module documentation."""
    modules = [
        "hexcore.models.model_loader",
        "hexcore.models.expert_adapters",
        "hexcore.models.cross_expert",
        "hexcore.models.transaction_classifier",
        "hexcore.knowledge.hybrid_retriever",
        "hexcore.knowledge.knowledge_graph",
        "hexcore.inference.pipeline",
    ]

    for module_name in modules:
        output_path = DOCS_DIR / f"{module_name.split('.')[-1]}.md"
        generate_module_docs(module_name, output_path)
        print(f"Generated docs for {module_name} at {output_path}")

if __name__ == "__main__":
    main()
````

## 7. Implementation Strategy and Timeline

Here's a recommended action plan for implementing these improvements:

### Week 1: Consolidation and Structure

1. Create a consolidated project structure
2. Remove duplicate code and standardize interfaces
3. Set up package structure and build system

### Week 2: Code Quality and Type Annotations

1. Add comprehensive type annotations
2. Implement error handling patterns
3. Set up linting and code quality tools
4. Fix identified issues

### Week 3: Documentation and Testing

1. Create documentation structure
2. Set up automatic API documentation generation
3. Improve test coverage
4. Create user guides and developer documentation

## Success Criteria and Benefits

By implementing these professionalization and hardening steps, you'll achieve:

1. **Better Code Quality**: Consistent style, reduced duplication, improved testing
2. **Enhanced Maintainability**: Clear structure, comprehensive documentation
3. **Easier Collaboration**: Well-defined interfaces and contribution guidelines
4. **Improved Reliability**: Robust error handling, type safety, memory management
5. **Professional Package**: Properly installable, versioned, and documented

These improvements will make Hexcore more resilient, easier to maintain, and more accessible to other developers, allowing you to focus on the innovative aspects of the MTG AI assistant rather than dealing with technical debt.

Would you like me to provide additional details on any of these aspects or help you implement a specific part of this professionalization plan?
