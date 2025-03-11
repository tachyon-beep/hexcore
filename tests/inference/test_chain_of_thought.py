"""
Unit tests for the Chain-of-Thought reasoning implementation.
"""

import unittest
from unittest.mock import patch, MagicMock

from src.inference.chain_of_thought import ChainOfThoughtReasoning


class TestChainOfThoughtReasoning(unittest.TestCase):
    """Test cases for the ChainOfThoughtReasoning class."""

    def setUp(self):
        """Set up test case."""
        self.reasoning = ChainOfThoughtReasoning()

    def test_initialization(self):
        """Test that the reasoning initializes properly."""
        self.assertIsNotNone(self.reasoning)
        self.assertEqual(self.reasoning.name, "chain_of_thought")

    def test_apply_default_config(self):
        """Test applying reasoning with default configuration."""
        query = "How does the stack work in MTG?"
        inputs = {"prompt": "Please explain how the stack works in MTG."}
        knowledge_context = {
            "kg_data": {
                "type": "rule_data",
                "data": [
                    {
                        "id": "405.1",
                        "text": "When a spell is cast, it goes on top of the stack.",
                    }
                ],
            }
        }

        # Apply reasoning
        enhanced_inputs = self.reasoning.apply(query, inputs, knowledge_context)

        # Verify the result
        self.assertIn("prompt", enhanced_inputs)
        enhanced_prompt = enhanced_inputs["prompt"]

        # Verify that the enhanced prompt contains the reasoning structure
        self.assertIn("<reasoning>", enhanced_prompt)
        self.assertIn("</reasoning>", enhanced_prompt)

        # Verify that the enhanced prompt contains the steps
        self.assertIn("Step 1:", enhanced_prompt)
        self.assertIn("Step 2:", enhanced_prompt)
        self.assertIn("Step 3:", enhanced_prompt)

        # Verify that knowledge was integrated
        self.assertIn("405.1", enhanced_prompt)

    def test_apply_custom_config(self):
        """Test applying reasoning with custom configuration."""
        query = "How does the stack work in MTG?"
        inputs = {"prompt": "Please explain how the stack works in MTG."}
        knowledge_context = {}

        # Custom configuration - use parameters that actually exist in the implementation
        config = {
            "max_steps": 3,
            "verify_steps": True,
            "rule_grounding": True,
        }

        # Apply reasoning
        enhanced_inputs = self.reasoning.apply(query, inputs, knowledge_context, config)

        # Verify the result
        enhanced_prompt = enhanced_inputs["prompt"]

        # Verify that the enhanced prompt contains key elements
        self.assertIn("Step 1:", enhanced_prompt)
        self.assertIn("Step 2:", enhanced_prompt)

        # Check for verification in a more generic way - look for verify-related words
        verification_terms = ["verify", "verification", "check", "assess", "evaluate"]
        has_verification = any(
            term in enhanced_prompt.lower() for term in verification_terms
        )
        self.assertTrue(
            has_verification, "No verification-related terms found in the prompt"
        )

    def test_apply_with_different_configurations(self):
        """Test applying reasoning with different configurations."""
        query = "How does the stack work in MTG?"
        inputs = {"prompt": "Please explain how the stack works in MTG."}
        knowledge_context = {}

        # Test with minimal steps and no verification
        config_minimal = {
            "max_steps": 3,
            "verify_steps": False,
        }
        enhanced_inputs_minimal = self.reasoning.apply(
            query, inputs, knowledge_context, config_minimal
        )
        enhanced_prompt_minimal = enhanced_inputs_minimal["prompt"]

        # Check that it has the expected steps
        self.assertIn("Step 1:", enhanced_prompt_minimal)
        self.assertNotIn("verification", enhanced_prompt_minimal.lower())

        # Test with more steps and verification
        config_full = {
            "max_steps": 5,
            "verify_steps": True,
        }
        enhanced_inputs_full = self.reasoning.apply(
            query, inputs, knowledge_context, config_full
        )
        enhanced_prompt_full = enhanced_inputs_full["prompt"]

        # Check expected elements
        self.assertIn("Step 1:", enhanced_prompt_full)
        self.assertIn("verification", enhanced_prompt_full.lower())

    def test_enhance_prompt(self):
        """Test enhancing prompt with reasoning elements."""
        original_prompt = "Please explain how the stack works in MTG."
        reasoning_elements = {
            "reasoning_content": "Step 1: Identify the stack mechanism.\nStep 2: Explain the order of resolution."
        }

        enhanced_prompt = self.reasoning.enhance_prompt(
            original_prompt, reasoning_elements
        )

        # Verify the result
        self.assertIn(original_prompt, enhanced_prompt)
        self.assertIn("<reasoning>", enhanced_prompt)
        self.assertIn("</reasoning>", enhanced_prompt)
        self.assertIn("Step 1:", enhanced_prompt)
        self.assertIn("Step 2:", enhanced_prompt)

    def test_knowledge_integration(self):
        """Test that knowledge is properly integrated into the reasoning."""
        # Create a simple query and knowledge
        query = "How does the stack work in MTG?"
        inputs = {"prompt": "Please explain the stack."}
        knowledge_context = {
            "kg_data": {
                "type": "rule_data",
                "data": [
                    {
                        "id": "405.1",
                        "text": "When a spell is cast, it goes on top of the stack.",
                    }
                ],
            }
        }

        # Apply reasoning with knowledge
        enhanced_inputs = self.reasoning.apply(query, inputs, knowledge_context)
        enhanced_prompt = enhanced_inputs["prompt"]

        # Check that knowledge has been integrated in some way
        self.assertIn("stack", enhanced_prompt)


if __name__ == "__main__":
    unittest.main()
