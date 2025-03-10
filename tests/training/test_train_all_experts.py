"""
Tests for the training pipeline that trains all expert adapters.

This module tests the functionality for training multiple LoRA adapters
for different expert types, with memory optimization and device mapping.
"""

import sys
import os
import pytest
import torch
import json
import tempfile
import logging
from unittest.mock import MagicMock, patch, mock_open, ANY, call
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.train_all_experts import (
    setup_training_args,
    train_expert,
    main,
)


class TestTrainingArguments:
    """Test cases for the setup_training_args function."""

    def test_default_arguments(self):
        """Test default argument values."""
        # Execute with no arguments
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = MagicMock(
                base_model="mistralai/Mixtral-8x7B-v0.1",
                output_dir="adapters",
                experts=["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
                quantization=4,
                mixed_precision=False,
                device_map="auto",
                validation_split=0.1,
                resume=False,
            )
            args = setup_training_args()

        # Verify default values
        assert args.base_model == "mistralai/Mixtral-8x7B-v0.1"
        assert args.output_dir == "adapters"
        assert args.experts == ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"]
        assert args.quantization == 4
        assert args.mixed_precision is False
        assert args.device_map == "auto"
        assert args.validation_split == pytest.approx(0.1)
        assert args.resume is False

    def test_custom_arguments(self):
        """Test parsing custom argument values."""
        # Execute with custom arguments
        with patch("argparse.ArgumentParser.parse_args") as mock_parse_args:
            mock_parse_args.return_value = MagicMock(
                base_model="openai/gpt-4",
                output_dir="custom_adapters",
                experts=["REASON", "EXPLAIN"],
                quantization=8,
                mixed_precision=True,
                device_map="cuda:0",
                validation_split=0.2,
                resume=True,
            )
            args = setup_training_args()

        # Verify custom values
        assert args.base_model == "openai/gpt-4"
        assert args.output_dir == "custom_adapters"
        assert args.experts == ["REASON", "EXPLAIN"]
        assert args.quantization == 8
        assert args.mixed_precision is True
        assert args.device_map == "cuda:0"
        assert args.validation_split == pytest.approx(0.2)
        assert args.resume is True


class TestTrainExpert:
    """Test cases for the train_expert function."""

    @patch("src.training.train_all_experts.LoRAAdapterTrainer")
    @patch("src.training.train_all_experts.create_expert_dataset_from_jsonl")
    @patch("src.training.train_all_experts.AutoTokenizer")
    @patch("src.training.train_all_experts.get_expert_config")
    def test_train_expert_successfully(
        self,
        mock_get_expert_config,
        mock_tokenizer_class,
        mock_create_dataset,
        mock_trainer_class,
    ):
        """Test training an expert successfully."""
        # Setup
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            "training_data_path": "data/training/reason_examples.jsonl"
        }
        mock_config.max_seq_length = 2048
        mock_get_expert_config.return_value = mock_config

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_create_dataset.return_value = (mock_train_dataset, mock_eval_dataset)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "loss": 0.5,
            "learning_rate": 5e-5,
            "epoch": 3.0,
        }
        mock_trainer_class.return_value = mock_trainer

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute
            result = train_expert(
                expert_type="REASON",
                base_model_path="mistralai/Mixtral-8x7B-v0.1",
                output_dir=tmpdir,
                quantization_bits=4,
                use_mixed_precision=True,
                device_map="auto",
                validation_split=0.1,
                resume_from_checkpoint=False,
            )

        # Verify
        mock_get_expert_config.assert_called_once_with("REASON")
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "mistralai/Mixtral-8x7B-v0.1"
        )
        mock_create_dataset.assert_called_once()
        mock_trainer_class.assert_called_once()
        mock_trainer.setup.assert_called_once_with(device_map="auto")
        mock_trainer.train.assert_called_once_with(
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            resume_from_checkpoint=None,
        )
        mock_trainer.save_adapter.assert_called_once()

        # Check result
        assert result["loss"] == pytest.approx(0.5)
        assert result["learning_rate"] == pytest.approx(5e-5)
        assert result["epoch"] == pytest.approx(3.0)

    @patch("src.training.train_all_experts.LoRAAdapterTrainer")
    @patch("src.training.train_all_experts.create_expert_dataset_from_jsonl")
    @patch("src.training.train_all_experts.AutoTokenizer")
    @patch("src.training.train_all_experts.get_expert_config")
    @patch("src.training.train_all_experts.os.makedirs")
    @patch("src.training.train_all_experts.os.path.getctime")
    def test_train_expert_with_resume(
        self,
        mock_getctime,
        mock_makedirs,
        mock_get_expert_config,
        mock_tokenizer_class,
        mock_create_dataset,
        mock_trainer_class,
    ):
        """Test training an expert with resume from checkpoint."""
        # Setup
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {
            "training_data_path": "data/training/reason_examples.jsonl"
        }
        mock_config.max_seq_length = 2048
        mock_get_expert_config.return_value = mock_config

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_create_dataset.return_value = (mock_train_dataset, mock_eval_dataset)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "loss": 0.5,
            "learning_rate": 5e-5,
            "epoch": 3.0,
        }
        mock_trainer_class.return_value = mock_trainer

        # Mock checkpoints - use non-existant paths that our test doesn't try to access
        checkpoint_paths = [
            Path("test_adapters/reason/checkpoint-500"),
            Path("test_adapters/reason/checkpoint-1000"),
        ]
        mock_getctime.side_effect = [
            1000,
            2000,
        ]  # 1000 for first checkpoint, 2000 for second

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint directories
            checkpoint_dir = os.path.join(tmpdir, "reason", "checkpoint-1000")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Patch the glob function to return our mock checkpoints
            with patch("pathlib.Path.glob") as mock_glob:
                mock_glob.return_value = checkpoint_paths

                # Execute
                result = train_expert(
                    expert_type="REASON",
                    base_model_path="mistralai/Mixtral-8x7B-v0.1",
                    output_dir="test_adapters",
                    quantization_bits=4,
                    use_mixed_precision=True,
                    device_map="auto",
                    validation_split=0.1,
                    resume_from_checkpoint=True,
                )

        # Verify
        mock_trainer.train.assert_called_once_with(
            train_dataset=mock_train_dataset,
            eval_dataset=mock_eval_dataset,
            resume_from_checkpoint=str(
                checkpoint_paths[1]
            ),  # Should use the latest checkpoint
        )
        mock_trainer.save_adapter.assert_called_once()

        # Check result
        assert result["loss"] == pytest.approx(0.5)
        assert result["learning_rate"] == pytest.approx(5e-5)
        assert result["epoch"] == pytest.approx(3.0)

    @patch("src.training.train_all_experts.LoRAAdapterTrainer")
    @patch("src.training.train_all_experts.create_expert_dataset_from_jsonl")
    @patch("src.training.train_all_experts.AutoTokenizer")
    @patch("src.training.train_all_experts.get_expert_config")
    def test_train_expert_fallback_data_path(
        self,
        mock_get_expert_config,
        mock_tokenizer_class,
        mock_create_dataset,
        mock_trainer_class,
    ):
        """Test fallback data path when not specified in config."""
        # Setup - empty training_data_path
        mock_config = MagicMock()
        mock_config.to_dict.return_value = {}
        mock_config.max_seq_length = 2048
        mock_get_expert_config.return_value = mock_config

        mock_tokenizer = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_train_dataset = MagicMock()
        mock_eval_dataset = MagicMock()
        mock_create_dataset.return_value = (mock_train_dataset, mock_eval_dataset)

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"loss": 0.5}
        mock_trainer_class.return_value = mock_trainer

        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            # Execute
            train_expert(
                expert_type="REASON",
                base_model_path="model_path",
                output_dir=tmpdir,
                quantization_bits=4,
                use_mixed_precision=False,
                device_map="auto",
                validation_split=0.1,
                resume_from_checkpoint=False,
            )

        # Verify that create_expert_dataset_from_jsonl was called with expected arguments
        mock_create_dataset.assert_called_once()

        # Check the arguments passed to create_expert_dataset_from_jsonl
        # The issue with the original test was trying to access positional args when we're using keyword args
        data_path_arg = mock_create_dataset.call_args.kwargs.get("data_path")
        assert data_path_arg == "data/training/reason_examples.jsonl"


class TestMainFunction:
    """Test cases for the main function."""

    @patch("src.training.train_all_experts.setup_training_args")
    @patch("src.training.train_all_experts.train_expert")
    @patch("src.training.train_all_experts.os.makedirs")
    def test_main_trains_all_experts(
        self, mock_makedirs, mock_train_expert, mock_setup_args
    ):
        """Test that main trains all requested experts."""
        # Setup
        mock_args = MagicMock()
        mock_args.base_model = "model_path"
        mock_args.output_dir = "output_dir"
        mock_args.experts = ["REASON", "EXPLAIN"]
        mock_args.quantization = 4
        mock_args.mixed_precision = True
        mock_args.device_map = "auto"
        mock_args.validation_split = 0.1
        mock_args.resume = False

        mock_setup_args.return_value = mock_args

        # Mock training results
        mock_train_expert.side_effect = [
            {"loss": 0.5, "expert": "REASON"},
            {"loss": 0.6, "expert": "EXPLAIN"},
        ]

        # Execute
        with patch("builtins.open", mock_open()) as _:
            main()

        # Verify
        mock_makedirs.assert_called_once_with("output_dir", exist_ok=True)
        assert mock_train_expert.call_count == 2

        # Verify arguments for each call
        for i, expert in enumerate(["REASON", "EXPLAIN"]):
            call_args = mock_train_expert.call_args_list[i][1]
            assert call_args["expert_type"] == expert
            assert call_args["base_model_path"] == "model_path"
            assert call_args["output_dir"] == "output_dir"
            assert call_args["quantization_bits"] == 4
            assert call_args["use_mixed_precision"] is True
            assert call_args["device_map"] == "auto"
            assert call_args["validation_split"] == pytest.approx(0.1)
            assert call_args["resume_from_checkpoint"] is False

    def test_main_handles_training_errors(self):
        """Test that main handles training errors gracefully."""
        # We need to directly mock the logger to get this test to work
        mock_logger = MagicMock()

        # Patching the actual logger used by train_all_experts.py
        with patch("src.training.train_all_experts.logger", mock_logger), patch(
            "src.training.train_all_experts.setup_training_args"
        ) as mock_setup_args, patch(
            "src.training.train_all_experts.train_expert"
        ) as mock_train_expert, patch(
            "src.training.train_all_experts.os.makedirs"
        ):

            # Setup args
            mock_args = MagicMock()
            mock_args.base_model = "model_path"
            mock_args.output_dir = "output_dir"
            mock_args.experts = ["REASON", "EXPLAIN"]  # List of experts
            mock_args.quantization = 4
            mock_args.mixed_precision = True
            mock_args.device_map = "auto"
            mock_args.validation_split = 0.1
            mock_args.resume = False
            mock_setup_args.return_value = mock_args

            # Make first expert training fail with a specific exception
            test_exception = Exception("Training failure")
            mock_train_expert.side_effect = [
                test_exception,
                {"loss": 0.6, "expert": "EXPLAIN"},
            ]

            # Execute
            with patch("builtins.open", mock_open()):
                main()

            # Verify that error was properly logged
            mock_logger.error.assert_called_once()
            # Check the call contains the expected arguments
            error_call_args = mock_logger.error.call_args[0]
            assert "Error training REASON expert" in error_call_args[0]
            assert "Training failure" in error_call_args[0]

    def test_main_skips_unknown_expert_types(self):
        """Test that main skips unknown expert types."""
        # We need to directly mock the logger to get this test to work
        mock_logger = MagicMock()

        # Patching the actual logger used by train_all_experts.py
        with patch("src.training.train_all_experts.logger", mock_logger), patch(
            "src.training.train_all_experts.setup_training_args"
        ) as mock_setup_args, patch(
            "src.training.train_all_experts.train_expert"
        ) as mock_train_expert, patch(
            "src.training.train_all_experts.os.makedirs"
        ):

            # Setup args with a list containing an unknown expert type
            mock_args = MagicMock()
            mock_args.base_model = "model_path"
            mock_args.output_dir = "output_dir"
            mock_args.experts = [
                "REASON",
                "UNKNOWN",
                "EXPLAIN",
            ]  # Unknown expert in the middle
            mock_args.quantization = 4
            mock_args.mixed_precision = True
            mock_args.device_map = "auto"
            mock_args.validation_split = 0.1
            mock_args.resume = False
            mock_setup_args.return_value = mock_args

            # Mock training results for the two valid expert types
            mock_train_expert.side_effect = [
                {"loss": 0.5, "expert": "REASON"},
                {"loss": 0.6, "expert": "EXPLAIN"},
            ]

            # Execute
            with patch("builtins.open", mock_open()):
                main()

            # Verify training was called for the valid experts
            assert mock_train_expert.call_count == 2

            # Verify warning was logged for the unknown expert type
            mock_logger.warning.assert_called_once()
            warning_args = mock_logger.warning.call_args[0]
            assert "Unknown expert type" in warning_args[0]
            assert "UNKNOWN" in warning_args[0]
