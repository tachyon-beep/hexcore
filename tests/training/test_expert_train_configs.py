"""
Tests for the expert_train_configs module.

This module tests the configuration management for expert adapters,
including config generation and parameter overrides.
"""

import sys
import pytest
from pathlib import Path
from typing import Dict, Any

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.expert_train_configs import (
    ExpertTrainConfig,
    get_expert_config,
    get_all_expert_configs,
)


class TestExpertTrainConfig:
    """Test cases for the ExpertTrainConfig class."""

    def test_default_initialization(self):
        """Test initializing a config with default values."""
        # Execute
        config = ExpertTrainConfig()

        # Verify default values
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == pytest.approx(0.05)
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert config.learning_rate == pytest.approx(3e-4)
        assert config.max_seq_length == 2048
        assert config.expert_type == ""
        assert config.training_data_path == ""
        assert isinstance(config.additional_params, dict)
        assert len(config.additional_params) == 0

    def test_custom_initialization(self):
        """Test initializing a config with custom values."""
        # Execute
        config = ExpertTrainConfig(
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj"],
            learning_rate=5e-5,
            max_seq_length=4096,
            expert_type="TEST",
            training_data_path="data/test.jsonl",
            additional_params={"custom_param": "value"},
        )

        # Verify custom values
        assert config.lora_r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == pytest.approx(0.1)
        assert config.target_modules == ["q_proj", "k_proj"]
        assert config.learning_rate == pytest.approx(5e-5)
        assert config.max_seq_length == 4096
        assert config.expert_type == "TEST"
        assert config.training_data_path == "data/test.jsonl"
        assert config.additional_params["custom_param"] == "value"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        # Setup
        config = ExpertTrainConfig(
            expert_type="TEST", additional_params={"custom_param": "value"}
        )

        # Execute
        config_dict = config.to_dict()

        # Verify
        assert isinstance(config_dict, dict)
        assert config_dict["lora_r"] == 16
        assert config_dict["expert_type"] == "TEST"
        assert "custom_param" in config_dict
        assert config_dict["custom_param"] == "value"

        # Verify all base attributes are included
        for attr in [
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "learning_rate",
            "weight_decay",
            "max_steps",
            "max_seq_length",
            "expert_type",
        ]:
            assert attr in config_dict


class TestGetExpertConfig:
    """Test cases for the get_expert_config function."""

    def test_reason_expert_config(self):
        """Test getting config for REASON expert."""
        # Execute
        config = get_expert_config("REASON")

        # Verify
        assert config.expert_type == "REASON"
        assert config.training_data_path == "data/training/reason_examples.jsonl"
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == pytest.approx(0.05)
        assert config.max_steps == 2500  # Special value for REASON
        assert "description" in config.additional_params
        assert "Logical reasoning" in config.additional_params["description"]

    def test_teach_expert_config(self):
        """Test getting config for TEACH expert."""
        # Execute
        config = get_expert_config("TEACH")

        # Verify
        assert config.expert_type == "TEACH"
        assert config.training_data_path == "data/training/teach_examples.jsonl"
        assert config.lora_r == 8  # Special value for TEACH
        assert config.lora_alpha == 16  # Special value for TEACH
        assert config.max_seq_length == 4096  # Special value for TEACH
        assert config.additional_params["use_pedagogical_formatting"] is True

    def test_invalid_expert_type(self):
        """Test behavior with invalid expert type."""
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            get_expert_config("INVALID")

        # Check error message
        error_msg = str(exc_info.value)
        assert "Unsupported expert type: INVALID" in error_msg
        assert "REASON" in error_msg
        assert "EXPLAIN" in error_msg

    def test_override_params(self):
        """Test overriding parameters."""
        # Setup
        overrides = {
            "lora_r": 64,
            "learning_rate": 1e-5,
            "custom_param": "custom_value",
        }

        # Execute
        config = get_expert_config("EXPLAIN", override_params=overrides)

        # Verify
        assert config.expert_type == "EXPLAIN"  # Base value preserved
        assert config.lora_r == 64  # Overridden
        assert config.learning_rate == pytest.approx(1e-5)  # Overridden
        assert (
            config.additional_params["custom_param"] == "custom_value"
        )  # Added to additional_params


class TestGetAllExpertConfigs:
    """Test cases for the get_all_expert_configs function."""

    def test_get_all_configs(self):
        """Test getting configs for all expert types."""
        # Execute
        configs = get_all_expert_configs()

        # Verify
        assert isinstance(configs, dict)
        assert len(configs) == 5

        # Check all expert types are included
        for expert_type in ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"]:
            assert expert_type in configs
            assert configs[expert_type].expert_type == expert_type
            assert isinstance(configs[expert_type], ExpertTrainConfig)
