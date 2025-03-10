"""
Training configurations for each expert adapter in the MTG AI system.

This module provides specialized configurations for fine-tuning the
different expert adapters (REASON, EXPLAIN, TEACH, PREDICT, RETROSPECT).
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ExpertTrainConfig:
    """Configuration for training a specific expert adapter."""

    # LoRA parameters
    lora_r: int = 16  # Rank of the update matrices
    lora_alpha: int = 32  # Alpha scaling parameter
    lora_dropout: float = 0.05

    # Target modules to apply LoRA
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    )

    # Training parameters
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_steps: int = 2000
    gradient_accumulation_steps: int = 8
    mixed_precision: bool = True

    # Dataset parameters
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4

    # Evaluation parameters
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 200

    # Expert-specific parameters
    expert_type: str = ""  # Will be set per expert
    training_data_path: str = ""  # Will be set per expert

    # Additional training parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary for the training API."""
        return {
            # LoRA parameters
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            # Training parameters
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "mixed_precision": self.mixed_precision,
            # Dataset parameters
            "max_seq_length": self.max_seq_length,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            # Evaluation parameters
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            # Expert-specific parameters
            "expert_type": self.expert_type,
            "training_data_path": self.training_data_path,
            # Additional parameters
            **self.additional_params,
        }


def get_expert_config(
    expert_type: str, override_params: Optional[Dict[str, Any]] = None
) -> ExpertTrainConfig:
    """
    Get the training configuration for a specific expert type.

    Args:
        expert_type: The type of expert (REASON, EXPLAIN, TEACH, PREDICT, RETROSPECT)
        override_params: Optional dictionary of parameters to override defaults

    Returns:
        Configuration object for the specified expert
    """
    # Base configurations for each expert type
    configs = {
        "REASON": ExpertTrainConfig(
            expert_type="REASON",
            training_data_path="data/training/reason_examples.jsonl",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            max_steps=2500,  # More steps for complex reasoning
            additional_params={
                "description": "Logical reasoning expert for MTG rules and interactions"
            },
        ),
        "EXPLAIN": ExpertTrainConfig(
            expert_type="EXPLAIN",
            training_data_path="data/training/explain_examples.jsonl",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.10,  # Higher dropout for explanation diversity
            additional_params={
                "description": "Clear explanation expert for MTG mechanics and concepts"
            },
        ),
        "TEACH": ExpertTrainConfig(
            expert_type="TEACH",
            training_data_path="data/training/teach_examples.jsonl",
            lora_r=8,  # Lower rank for teaching (less deviation from base model)
            lora_alpha=16,
            lora_dropout=0.05,
            max_seq_length=4096,  # Longer context for educational content
            additional_params={
                "description": "Teaching expert for MTG fundamentals and strategies",
                "use_pedagogical_formatting": True,
            },
        ),
        "PREDICT": ExpertTrainConfig(
            expert_type="PREDICT",
            training_data_path="data/training/predict_examples.jsonl",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            additional_params={
                "description": "Predictive expert for MTG game states and probabilities",
                "train_on_expected_utility": True,
            },
        ),
        "RETROSPECT": ExpertTrainConfig(
            expert_type="RETROSPECT",
            training_data_path="data/training/retrospect_examples.jsonl",
            lora_r=8,  # Lower rank for retrospective analysis
            lora_alpha=16,
            lora_dropout=0.05,
            additional_params={
                "description": "Analysis expert for reviewing MTG games and decisions",
                "focus_on_improvement_feedback": True,
            },
        ),
    }

    # Ensure the expert type is supported
    if expert_type not in configs:
        raise ValueError(
            f"Unsupported expert type: {expert_type}. "
            f"Supported types: {list(configs.keys())}"
        )

    # Get the base configuration
    config = configs[expert_type]

    # Apply any overrides
    if override_params:
        for key, value in override_params.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.additional_params[key] = value

    return config


def get_all_expert_configs() -> Dict[str, ExpertTrainConfig]:
    """
    Get configurations for all expert types.

    Returns:
        Dictionary mapping expert types to their configurations
    """
    return {
        expert_type: get_expert_config(expert_type)
        for expert_type in ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"]
    }
