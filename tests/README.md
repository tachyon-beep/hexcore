# Hexcore Test Suite

This directory contains the test suite for the Hexcore project, a Magic: The Gathering AI assistant that uses a Mixture-of-Experts architecture.

## Test Structure

The test suite is organized to mirror the structure of the source code:

```
tests/
├── models/             # Tests for model loading, expert adapters, cross-expert attention
├── data/               # Tests for data loading (cards, rules, etc.)
├── knowledge/          # Tests for retrieval system
├── inference/          # Tests for inference pipeline
├── utils/              # Tests for utilities
└── integration/        # Integration tests that span multiple components
```

## Running Tests

We use [pytest](https://docs.pytest.org/) as our testing framework. Tests can be run with:

```bash
# Run all tests
python -m pytest

# Run tests with verbose output
python -m pytest -v

# Run tests for a specific module
python -m pytest tests/models/

# Run a specific test file
python -m pytest tests/models/test_expert_adapters.py

# Run a specific test
python -m pytest tests/models/test_expert_adapters.py::TestExpertAdapterManager::test_init_loads_available_adapters
```

## Test Types

### Unit Tests

Each component has a corresponding unit test suite that verifies its functionality in isolation. We use mocking extensively to isolate components from their dependencies.

### Integration Tests

Integration tests verify that components work together correctly. These focus on the interactions between modules and are essential for ensuring that the system functions as a whole.

### Performance Tests

Performance tests evaluate the system's efficiency, response time, and resource usage. These are critical for ensuring the system meets its performance requirements.

## Mocking Strategy

For unit tests, we use Python's `unittest.mock` to create mocks and stubs for dependencies:

- **Model Mocking**: We use mock objects for the base LLM models to avoid loading large models during testing.
- **GPU Mocking**: For tests that involve GPU operations, we provide utilities to simulate GPU functionality when actual GPUs are not available.
- **Knowledge Base Mocking**: We use smaller, controlled knowledge bases for deterministic testing.

## Adding New Tests

When adding new tests, please follow these guidelines:

1. **Structure**: Place tests in the appropriate directory based on the component being tested.
2. **Naming**: Use the `test_*.py` naming convention for test files and `test_*` for test methods.
3. **Documentation**: Include docstrings explaining the purpose of each test.
4. **Fixtures**: Use pytest fixtures for setup and teardown to keep tests clean and DRY.
5. **Mocking**: Mock external dependencies to keep tests focused and fast.
6. **Parameterization**: Use pytest's parameterization for testing multiple scenarios.

Example test structure:

```python
class TestMyComponent:
    @pytest.fixture
    def my_fixture(self):
        # Setup code
        yield something
        # Teardown code

    def test_my_function(self, my_fixture):
        """Test that my_function behaves correctly in normal conditions."""
        # Test code here
        assert result == expected
```

## Completed Test Implementations

- ✅ `ExpertAdapterManager`: Tests for adapter loading, switching, and memory optimization
- ✅ `CrossExpertAttention`: Tests for attention mechanism and expert output combination

## Next Test Implementations

- Multi-expert query processing integration tests
- Memory usage monitoring for inference
- Transaction classifier tests
- Performance benchmarking for different query types

For a complete overview of the test strategy and roadmap, see the [Test Plan](../docs/project/test_plan.md).
