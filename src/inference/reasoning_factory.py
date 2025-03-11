"""
Reasoning Factory for MTG AI Reasoning Assistant.

This module provides a factory mechanism to create the appropriate reasoning
implementation based on the selected reasoning mode. It supports the three
core reasoning methods (CoT, MCTS, R1) and allows for easy extension.
"""

import logging
from typing import Dict, Any, Optional, Type

from src.inference.base_reasoning import BaseReasoning
from src.inference.chain_of_thought import ChainOfThoughtReasoning
from src.inference.mcts_reasoning import MCTSReasoning
from src.inference.r1_reasoning import R1StyleReasoning

logger = logging.getLogger(__name__)


class ReasoningFactory:
    """
    Factory class for creating reasoning implementations.

    This factory instantiates the appropriate reasoning implementation
    based on the reasoning mode name.
    """

    # Registry of reasoning implementations
    _registry = {
        "chain_of_thought": ChainOfThoughtReasoning,
        "mcts": MCTSReasoning,
        "r1_style": R1StyleReasoning,
    }

    @classmethod
    def create_reasoning(cls, mode: str) -> BaseReasoning:
        """
        Create an instance of the appropriate reasoning implementation.

        Args:
            mode: Name of the reasoning mode to create

        Returns:
            An instance of the appropriate reasoning implementation

        Raises:
            ValueError: If the specified mode is not registered
        """
        if mode not in cls._registry:
            registered_modes = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown reasoning mode: '{mode}'. "
                f"Available modes are: {registered_modes}"
            )

        reasoning_class = cls._registry[mode]
        logger.debug(f"Creating reasoning implementation for mode: {mode}")
        return reasoning_class()

    @classmethod
    def register_reasoning(
        cls, mode: str, reasoning_class: Type[BaseReasoning]
    ) -> None:
        """
        Register a new reasoning implementation.

        Args:
            mode: Name of the reasoning mode
            reasoning_class: The reasoning implementation class

        Raises:
            TypeError: If reasoning_class is not a subclass of BaseReasoning
        """
        if not issubclass(reasoning_class, BaseReasoning):
            raise TypeError(
                f"Reasoning class must be a subclass of BaseReasoning, "
                f"got {reasoning_class.__name__}"
            )

        cls._registry[mode] = reasoning_class
        logger.debug(f"Registered reasoning implementation for mode: {mode}")

    @classmethod
    def get_available_modes(cls) -> list[str]:
        """
        Get a list of available reasoning modes.

        Returns:
            List of registered reasoning mode names
        """
        return list(cls._registry.keys())


def create_reasoning(mode: str) -> BaseReasoning:
    """
    Create an instance of the appropriate reasoning implementation.

    This function provides a simplified interface to the ReasoningFactory.

    Args:
        mode: Name of the reasoning mode to create

    Returns:
        An instance of the appropriate reasoning implementation
    """
    return ReasoningFactory.create_reasoning(mode)
