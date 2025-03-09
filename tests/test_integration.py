#!/usr/bin/env python
# tests/test_integration.py
"""
DEPRECATED: This file has been refactored into multiple specialized test files.

This integration test file has been split into multiple focused test files
in the tests/integration/ directory to better organize testing by functionality:

- tests/integration/test_memory_performance.py - Tests for memory usage
- tests/integration/test_performance_metrics.py - Tests for timing metrics
- tests/integration/test_stability.py - Tests for system stability
- tests/integration/test_multi_expert_pipeline.py - Tests for multi-expert functionality

Please use the new test files for all future testing. The shared fixture for all
integration tests has been moved to tests/integration/__init__.py.

This file will be removed in a future update.
"""

import sys
import warnings
from pathlib import Path

warnings.warn(
    "tests/test_integration.py is deprecated. Use specialized tests in tests/integration/ directory.",
    DeprecationWarning,
    stacklevel=2,
)

# Run all new integration tests if this file is executed directly
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main(["-xvs", "tests/integration/"]))
