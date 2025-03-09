# src/utils/expert_config.py
"""
Centralized configuration for expert types and their settings.
This module provides a single source of truth for expert configurations,
making it easier to add or modify expert types without changing code in multiple places.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import json
import logging

logger = logging.getLogger(__name__)

# Default expert type definitions
DEFAULT_EXPERT_TYPES = [
    "REASON",  # Step-by-step logical reasoning through game states and rules
    "EXPLAIN",  # Clear articulation of MTG rules and decisions
    "TEACH",  # Breaking down concepts for learners
    "PREDICT",  # Simulating future game states and evaluating moves
    "RETROSPECT",  # Analyzing past plays to identify mistakes
]

# External configuration path (can be overridden by environment variable)
CONFIG_PATH = os.environ.get("HEXCORE_EXPERT_CONFIG", "config/expert_types.json")


def get_expert_types() -> List[str]:
    """
    Get the list of expert types, either from configuration file or defaults.

    Returns:
        List of expert type identifiers
    """
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                if "expert_types" in config and isinstance(
                    config["expert_types"], list
                ):
                    return config["expert_types"]
                else:
                    logger.warning(
                        f"Invalid expert_types configuration in {CONFIG_PATH}, using defaults"
                    )
        except Exception as e:
            logger.warning(
                f"Error loading expert configuration from {CONFIG_PATH}: {str(e)}, using defaults"
            )

    return DEFAULT_EXPERT_TYPES


def get_expert_id_mappings() -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Get mappings between expert IDs and expert types.

    Returns:
        Tuple of (id2label, label2id) dictionaries
    """
    expert_types = get_expert_types()
    id2label = {i: expert_type for i, expert_type in enumerate(expert_types)}
    label2id = {expert_type: i for i, expert_type in id2label.items()}

    return id2label, label2id


def get_expert_config(expert_type: str) -> Dict[str, Any]:
    """
    Get specific configuration for an expert type.

    Args:
        expert_type: The expert type identifier

    Returns:
        Dictionary of configuration settings for this expert
    """
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                if "expert_configs" in config and isinstance(
                    config["expert_configs"], dict
                ):
                    return config["expert_configs"].get(expert_type, {})
        except Exception:
            pass

    # Default configurations if no file exists or expert not in config
    default_configs = {
        "REASON": {
            "temperature": 0.3,
            "top_p": 0.8,
        },
        "EXPLAIN": {
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "TEACH": {
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "PREDICT": {
            "temperature": 0.3,
            "top_p": 0.8,
        },
        "RETROSPECT": {
            "temperature": 0.3,
            "top_p": 0.8,
        },
    }

    return default_configs.get(expert_type, {})
