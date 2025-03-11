"""
Advanced Reasoning Demo for MTG AI Reasoning Assistant.

This script demonstrates how to use the various reasoning methods
(Chain-of-Thought, MCTS, R1-Style) with the MTG AI Assistant.
"""

import logging
import json
import sys
from typing import Dict, Any

# Add the project root to the Python path
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.reasoning_selector import ReasoningModeSelector
from src.inference.reasoning_factory import create_reasoning

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main demonstration function."""
    # Sample queries demonstrating different types of questions
    sample_queries = [
        {
            "expert_type": "REASON",
            "query": "How does the stack work when multiple triggered abilities resolve at the same time?",
            "description": "Rules explanation query suitable for Chain-of-Thought reasoning",
        },
        {
            "expert_type": "PREDICT",
            "query": "What's the probability of drawing Lightning Bolt in my opening hand if I have 4 copies in a 60 card deck?",
            "description": "Probability-based query suitable for MCTS reasoning",
        },
        {
            "expert_type": "REASON",
            "query": "In a tournament, if my opponent controls Teferi, Time Raveler and I cast an instant during my main phase, can they respond with a counterspell?",
            "description": "Complex rules edge case suitable for R1-style reasoning",
        },
    ]

    # Initialize reasoning mode selector
    selector = ReasoningModeSelector()

    # Process each query
    for i, query_info in enumerate(sample_queries):
        query = query_info["query"]
        expert_type = query_info["expert_type"]
        description = query_info["description"]

        logger.info(f"\n\nQuery {i+1}: {query}")
        logger.info(f"Description: {description}")
        logger.info(f"Expert Type: {expert_type}")

        # Select reasoning mode
        # In a real system, the confidence score would come from the transaction classifier
        confidence_score = 0.85  # Example confidence score
        selected_mode, config = selector.select_reasoning_mode(
            query, expert_type, confidence_score
        )

        logger.info(f"Selected reasoning mode: {selected_mode}")
        logger.info(f"Configuration: {config}")

        # Create the reasoning implementation
        reasoning = create_reasoning(selected_mode)

        # Create a sample inputs dictionary (simplified for demonstration)
        inputs = {
            "prompt": f"Query: {query}\n\nPlease analyze this Magic: The Gathering question."
        }

        # Create a sample knowledge context (simplified for demonstration)
        knowledge_context = create_sample_knowledge_context(query)

        # Apply reasoning
        enhanced_inputs = reasoning.apply(query, inputs, knowledge_context, config)

        # Display the result (truncated for brevity)
        enhanced_prompt = enhanced_inputs.get("prompt", "")
        logger.info(f"Enhanced prompt (truncated):\n{enhanced_prompt[:500]}...\n")

        # In a real system, this enhanced prompt would be sent to the model for generation


def create_sample_knowledge_context(query: str) -> Dict[str, Any]:
    """
    Create a sample knowledge context for demonstration purposes.

    Args:
        query: The query text

    Returns:
        A sample knowledge context
    """
    # Extract card names from the query (very simplified)
    import re

    card_names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)

    if "stack" in query.lower():
        # Sample rule knowledge for stack-related queries
        return {
            "kg_data": {
                "type": "rule_data",
                "data": [
                    {
                        "id": "405.1",
                        "text": "When a spell is cast, it goes on top of the stack. When an ability is activated or triggers, it goes on top of the stack.",
                    },
                    {
                        "id": "405.2",
                        "text": "The stack resolves one object at a time, always taking the top object.",
                    },
                    {
                        "id": "405.5",
                        "text": "When all players pass in succession, the top object on the stack resolves.",
                    },
                ],
            }
        }
    elif "Teferi" in query:
        # Sample card knowledge for Teferi-related queries
        return {
            "kg_data": {
                "type": "card_data",
                "data": [
                    {
                        "name": "Teferi, Time Raveler",
                        "card_types": ["Planeswalker"],
                        "text": "Each opponent can cast spells only any time they could cast a sorcery.\n+1: Until your next turn, you may cast sorcery spells as though they had flash.\n−2: Return up to one target artifact, creature, or enchantment to its owner's hand. Draw a card.",
                        "loyalty": 4,
                    }
                ],
            }
        }
    elif "probability" in query.lower() or "opening hand" in query.lower():
        # Sample card knowledge for probability-related queries
        return {
            "kg_data": {
                "type": "card_data",
                "data": [
                    {
                        "name": "Lightning Bolt",
                        "card_types": ["Instant"],
                        "text": "Lightning Bolt deals 3 damage to any target.",
                    }
                ],
            }
        }
    else:
        # Generic empty knowledge context
        return {}


if __name__ == "__main__":
    main()
