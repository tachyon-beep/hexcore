"""
Unit tests for the reasoning factory module.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.inference.reasoning_factory import ReasoningFactory, create_reasoning
from src.inference.base_reasoning import BaseReasoning
from src.inference.chain_of_thought import ChainOfThoughtReasoning
from src.inference.mcts_reasoning import MCTSReasoning
from src.inference.r1_reasoning import R1StyleReasoning


class TestReasoningFactory(unittest.TestCase):
    """Test cases for the ReasoningFactory class."""

    def test_create_chain_of_thought(self):
        """Test creating chain of thought reasoning."""
        reasoning = ReasoningFactory.create_reasoning("chain_of_thought")
        self.assertIsNotNone(reasoning)
        self.assertIsInstance(reasoning, ChainOfThoughtReasoning)
        self.assertEqual(reasoning.name, "chain_of_thought")

    def test_create_mcts(self):
        """Test creating MCTS reasoning."""
        reasoning = ReasoningFactory.create_reasoning("mcts")
        self.assertIsNotNone(reasoning)
        self.assertIsInstance(reasoning, MCTSReasoning)
        self.assertEqual(reasoning.name, "mcts")

    def test_create_r1_style(self):
        """Test creating R1-style reasoning."""
        reasoning = ReasoningFactory.create_reasoning("r1_style")
        self.assertIsNotNone(reasoning)
        self.assertIsInstance(reasoning, R1StyleReasoning)
        self.assertEqual(reasoning.name, "r1_style")

    def test_create_unknown_mode(self):
        """Test that creating unknown mode raises ValueError."""
        with self.assertRaises(ValueError):
            ReasoningFactory.create_reasoning("unknown_mode")

    def test_get_available_modes(self):
        """Test getting available modes."""
        modes = ReasoningFactory.get_available_modes()
        self.assertIsInstance(modes, list)
        self.assertIn("chain_of_thought", modes)
        self.assertIn("mcts", modes)
        self.assertIn("r1_style", modes)

    def test_register_new_reasoning(self):
        """Test registering a new reasoning implementation."""

        # Create a mock reasoning class
        class MockReasoning(BaseReasoning):
            def __init__(self):
                super().__init__("mock")

            def apply(self, query, inputs, knowledge_context, reasoning_config=None):
                return inputs

        # Register the mock reasoning
        ReasoningFactory.register_reasoning("mock", MockReasoning)

        # Verify it's available
        modes = ReasoningFactory.get_available_modes()
        self.assertIn("mock", modes)

        # Test creating it
        reasoning = ReasoningFactory.create_reasoning("mock")
        self.assertIsNotNone(reasoning)
        self.assertIsInstance(reasoning, MockReasoning)
        self.assertEqual(reasoning.name, "mock")

        # Clean up: remove the mock reasoning from the registry
        ReasoningFactory._registry.pop("mock", None)

    def test_register_invalid_reasoning(self):
        """Test that registering an invalid reasoning raises TypeError."""

        # Invalid class (not a subclass of BaseReasoning)
        class InvalidReasoning:
            def __init__(self):
                self.name = "invalid"

            def apply(self, query, inputs, knowledge_context, reasoning_config=None):
                return inputs

        # This should raise TypeError since InvalidReasoning is not a subclass of BaseReasoning
        with self.assertRaises(TypeError):
            ReasoningFactory.register_reasoning("invalid", InvalidReasoning)

    def test_create_reasoning_helper(self):
        """Test the create_reasoning helper function."""
        reasoning = create_reasoning("chain_of_thought")
        self.assertIsNotNone(reasoning)
        self.assertIsInstance(reasoning, ChainOfThoughtReasoning)
        self.assertEqual(reasoning.name, "chain_of_thought")


if __name__ == "__main__":
    unittest.main()
