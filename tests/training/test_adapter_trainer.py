"""
Tests for the LoRAAdapterTrainer implementation.

This module tests the adapter training functionality with memory optimization,
mixed precision support, and LoRA integration.
"""

import sys
import os
import pytest
import torch
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from tempfile import TemporaryDirectory

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.adapter_trainer import LoRAAdapterTrainer
from src.training.mixed_precision import MixedPrecisionTrainer


class TestLoRAAdapterTrainer:
    """Test cases for the LoRAAdapterTrainer class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        model.print_trainable_parameters = MagicMock()
        model.save_pretrained = MagicMock()

        # Mock parameters
        param1 = MagicMock()
        param1.requires_grad = True
        param1.data = torch.randn(10, 10)

        param2 = MagicMock()
        param2.requires_grad = False
        param2.data = torch.randn(5, 5)

        model.parameters.return_value = [param1, param2]

        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 2
        tokenizer.save_pretrained = MagicMock()
        return tokenizer

    @pytest.fixture
    def mock_trainer(self):
        """Create a mock Trainer."""
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(
            metrics={"loss": 0.5, "learning_rate": 5e-5}
        )
        trainer.compute_loss = MagicMock()
        trainer.save_model = MagicMock()
        trainer.log_metrics = MagicMock()
        trainer.save_metrics = MagicMock()
        trainer.save_state = MagicMock()
        trainer.evaluate = MagicMock(return_value={"eval_loss": 0.6, "perplexity": 1.8})
        return trainer

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        dataset = MagicMock()
        dataset.__len__.return_value = 100
        return dataset

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        with TemporaryDirectory() as tmpdir:
            yield tmpdir

    @patch("src.training.adapter_trainer.get_expert_config")
    def test_init(self, mock_get_expert_config, temp_output_dir):
        """Test initialization with valid parameters."""
        # Setup
        mock_config = MagicMock()
        mock_config.lora_r = 16
        mock_config.lora_alpha = 32
        mock_config.lora_dropout = 0.05
        mock_config.target_modules = ["q_proj", "v_proj"]
        mock_config.per_device_train_batch_size = 8
        mock_config.gradient_accumulation_steps = 4
        mock_config.learning_rate = 2e-5
        mock_config.weight_decay = 0.01
        mock_config.max_steps = 1000
        mock_config.additional_params = {}

        mock_get_expert_config.return_value = mock_config

        # Execute
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="REASON",
            output_dir=temp_output_dir,
            quantization_bits=4,
            use_mixed_precision=True,
        )

        # Verify
        assert trainer.base_model_path == "models/mixtral-8x7b"
        assert trainer.expert_type == "REASON"
        assert trainer.output_dir == temp_output_dir
        assert trainer.quantization_bits == 4
        assert trainer.use_mixed_precision is True

        # Check LoRA config
        assert trainer.lora_config["r"] == 16
        assert trainer.lora_config["lora_alpha"] == 32
        assert trainer.lora_config["lora_dropout"] == pytest.approx(0.05)
        assert trainer.lora_config["target_modules"] == ["q_proj", "v_proj"]
        assert trainer.lora_config["bias"] == "none"
        assert trainer.lora_config["task_type"] == "CAUSAL_LM"

        # Check training params
        assert trainer.train_params["batch_size"] == 8
        assert trainer.train_params["gradient_accumulation_steps"] == 4
        assert trainer.train_params["learning_rate"] == pytest.approx(2e-5)
        assert trainer.train_params["weight_decay"] == pytest.approx(0.01)

        # Check instance variables
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.trainer is None
        assert isinstance(trainer.mp_trainer, MixedPrecisionTrainer)
        assert trainer.mp_trainer.use_amp is True

    @patch("src.training.adapter_trainer.get_expert_config")
    def test_init_with_overrides(self, mock_get_expert_config, temp_output_dir):
        """Test initialization with config overrides."""
        # Setup
        mock_config = MagicMock()
        mock_config.lora_r = 16
        mock_config.lora_alpha = 32
        mock_config.lora_dropout = 0.05
        mock_config.target_modules = ["q_proj", "v_proj"]
        mock_config.per_device_train_batch_size = 8
        mock_config.gradient_accumulation_steps = 4
        mock_config.learning_rate = 2e-5
        mock_config.weight_decay = 0.01
        mock_config.max_steps = 1000
        mock_config.additional_params = {}

        mock_get_expert_config.return_value = mock_config

        lora_override = {
            "r": 32,  # Override default rank
            "lora_dropout": 0.1,  # Override default dropout
        }

        train_override = {
            "batch_size": 4,  # Smaller batch size
            "learning_rate": 1e-5,  # Lower learning rate
        }

        # Execute
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="EXPLAIN",
            output_dir=temp_output_dir,
            override_lora_config=lora_override,
            override_train_params=train_override,
        )

        # Verify - overrides should take precedence
        assert trainer.lora_config["r"] == 32  # Overridden
        assert trainer.lora_config["lora_alpha"] == 32  # Original
        assert trainer.lora_config["lora_dropout"] == pytest.approx(0.1)  # Overridden

        # train_params should reflect expert_config, not override_train_params
        # because get_expert_config is mocked to return mock_config regardless of override_train_params
        assert trainer.train_params["batch_size"] == 8
        assert trainer.train_params["learning_rate"] == pytest.approx(2e-5)

    @patch("src.training.adapter_trainer.AutoTokenizer")
    @patch("src.training.adapter_trainer.AutoModelForCausalLM")
    @patch("src.training.adapter_trainer.prepare_model_for_kbit_training")
    @patch("src.training.adapter_trainer.LoraConfig")
    @patch("src.training.adapter_trainer.get_peft_model")
    @patch("src.training.adapter_trainer.get_expert_config")
    def test_setup(
        self,
        mock_get_expert_config,
        mock_get_peft_model,
        mock_lora_config,
        mock_prepare_model,
        mock_auto_model,
        mock_auto_tokenizer,
        mock_model,
        mock_tokenizer,
        temp_output_dir,
    ):
        """Test model and tokenizer setup."""
        # Setup
        mock_config = MagicMock()
        mock_config.lora_r = 16
        mock_config.lora_alpha = 32
        mock_config.lora_dropout = 0.05
        mock_config.target_modules = ["q_proj", "v_proj"]
        mock_config.per_device_train_batch_size = 8
        mock_config.gradient_accumulation_steps = 4
        mock_config.learning_rate = 2e-5
        mock_config.weight_decay = 0.01
        mock_config.max_steps = 1000
        mock_config.additional_params = {}

        mock_get_expert_config.return_value = mock_config
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_prepare_model.return_value = mock_model
        mock_get_peft_model.return_value = mock_model

        # Create trainer
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="REASON",
            output_dir=temp_output_dir,
        )

        # Execute setup
        trainer.setup()

        # Verify tokenizer setup
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "models/mixtral-8x7b"
        )
        assert trainer.tokenizer == mock_tokenizer

        # In the actual implementation, if pad_token_id is None, it should be set to eos_token_id
        # We're testing that the code sets this up correctly
        assert (
            mock_tokenizer.pad_token_id is not None
        ), "pad_token_id should be set by the implementation"
        assert mock_tokenizer.pad_token_id == mock_tokenizer.eos_token_id

        # Verify model setup
        mock_auto_model.from_pretrained.assert_called_once()
        mock_prepare_model.assert_called_once_with(mock_model)
        mock_lora_config.assert_called_once()
        mock_get_peft_model.assert_called_once()

        # Verify model configuration
        assert trainer.model == mock_model
        mock_model.gradient_checkpointing_enable.assert_called_once()
        mock_model.print_trainable_parameters.assert_called_once()

    @patch("src.training.adapter_trainer.os.makedirs")
    @patch("src.training.adapter_trainer.TrainingArguments")
    @patch("src.training.adapter_trainer.Trainer")
    @patch("src.training.adapter_trainer.get_expert_config")
    def test_train(
        self,
        mock_get_expert_config,
        mock_trainer_class,
        mock_training_args,
        mock_makedirs,
        mock_model,
        mock_trainer,
        mock_dataset,
        mock_tokenizer,
        temp_output_dir,
    ):
        """Test training process."""
        # Setup
        mock_config = MagicMock()
        mock_config.lora_r = 16
        mock_config.lora_alpha = 32
        mock_config.lora_dropout = 0.05
        mock_config.target_modules = ["q_proj", "v_proj"]
        mock_config.per_device_train_batch_size = 8
        mock_config.gradient_accumulation_steps = 4
        mock_config.learning_rate = 2e-5
        mock_config.weight_decay = 0.01
        mock_config.max_steps = 1000
        mock_config.additional_params = {}

        mock_get_expert_config.return_value = mock_config
        mock_trainer_class.return_value = mock_trainer

        # Create trainer and set model
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="REASON",
            output_dir=temp_output_dir,
        )
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer

        # Execute train
        result = trainer.train(train_dataset=mock_dataset, eval_dataset=None)

        # Verify
        mock_makedirs.assert_called_once_with(temp_output_dir, exist_ok=True)
        mock_training_args.assert_called_once()
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once_with(resume_from_checkpoint=None)
        mock_trainer.save_model.assert_called_once()
        mock_trainer.log_metrics.assert_called_once_with(
            "train", {"loss": 0.5, "learning_rate": 5e-5}
        )
        mock_trainer.save_metrics.assert_called_once_with(
            "train", {"loss": 0.5, "learning_rate": 5e-5}
        )
        mock_trainer.save_state.assert_called_once()

        # Verify returned metrics
        assert "loss" in result
        assert "mp_stats" in result

    @patch("builtins.open", new_callable=mock_open)
    @patch("src.training.adapter_trainer.json.dump")
    @patch("src.training.adapter_trainer.os.makedirs")
    @patch("src.training.adapter_trainer.get_expert_config")
    def test_save_adapter(
        self,
        mock_get_expert_config,
        mock_makedirs,
        mock_json_dump,
        mock_open_file,
        mock_model,
        mock_tokenizer,
        temp_output_dir,
    ):
        """Test saving adapter weights and configuration."""
        # Setup
        mock_config = MagicMock()
        mock_config.lora_r = 16
        mock_config.lora_alpha = 32
        mock_config.lora_dropout = 0.05
        mock_config.target_modules = ["q_proj", "v_proj"]
        mock_config.per_device_train_batch_size = 8
        mock_config.gradient_accumulation_steps = 4
        mock_config.learning_rate = 2e-5
        mock_config.weight_decay = 0.01
        mock_config.max_steps = 1000
        mock_config.additional_params = {"description": "REASON expert adapter"}

        mock_get_expert_config.return_value = mock_config

        # Create trainer and set model/tokenizer
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="REASON",
            output_dir=temp_output_dir,
        )
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer

        # Execute save
        save_path = os.path.join(temp_output_dir, "saved_adapter")
        trainer.save_adapter(path=save_path)

        # Verify
        mock_makedirs.assert_called_once_with(save_path, exist_ok=True)
        mock_model.save_pretrained.assert_called_once_with(save_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(save_path)

        # Verify config was saved
        mock_open_file.assert_called_once_with(
            os.path.join(save_path, "expert_config.json"), "w"
        )
        mock_json_dump.assert_called_once()

        # Check saved config content (extract first positional arg of json.dump)
        saved_config = mock_json_dump.call_args[0][0]
        assert saved_config["expert_type"] == "REASON"
        assert saved_config["description"] == "REASON expert adapter"
        assert "lora_config" in saved_config
        assert "training_params" in saved_config

    @patch("src.training.adapter_trainer.get_expert_config")
    def test_evaluate(
        self, mock_get_expert_config, mock_trainer, mock_dataset, temp_output_dir
    ):
        """Test evaluation process."""
        # Setup
        mock_config = MagicMock()
        mock_config.additional_params = {}
        mock_get_expert_config.return_value = mock_config

        # Create trainer and set trainer instance
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="REASON",
            output_dir=temp_output_dir,
        )
        trainer.trainer = mock_trainer

        # Execute evaluate
        result = trainer.evaluate(eval_dataset=mock_dataset)

        # Verify
        mock_trainer.evaluate.assert_called_once_with(eval_dataset=mock_dataset)
        mock_trainer.log_metrics.assert_called_once_with(
            "eval", {"eval_loss": 0.6, "perplexity": 1.8}
        )
        mock_trainer.save_metrics.assert_called_once_with(
            "eval", {"eval_loss": 0.6, "perplexity": 1.8}
        )

        # Verify returned metrics
        assert "eval_loss" in result
        assert "perplexity" in result

    @patch("src.training.adapter_trainer.get_expert_config")
    def test_evaluate_without_trainer(self, mock_get_expert_config, temp_output_dir):
        """Test evaluation without trainer initialization."""
        # Setup
        mock_config = MagicMock()
        mock_config.additional_params = {}
        mock_get_expert_config.return_value = mock_config

        # Create trainer without setting trainer instance
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="REASON",
            output_dir=temp_output_dir,
        )

        # Execute and verify exception is raised
        with pytest.raises(ValueError, match="Trainer not initialized"):
            trainer.evaluate(eval_dataset=MagicMock())

    @patch("src.training.adapter_trainer.AutoTokenizer")
    @patch("src.training.adapter_trainer.AutoModelForCausalLM")
    @patch("src.training.adapter_trainer.get_expert_config")
    def test_setup_with_custom_device_map(
        self,
        mock_get_expert_config,
        mock_auto_model,
        mock_auto_tokenizer,
        mock_model,
        mock_tokenizer,
        temp_output_dir,
    ):
        """Test setup with custom device mapping."""
        # Setup
        mock_config = MagicMock()
        mock_config.additional_params = {}
        mock_get_expert_config.return_value = mock_config
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        # Create trainer
        trainer = LoRAAdapterTrainer(
            base_model_path="models/mixtral-8x7b",
            expert_type="REASON",
            output_dir=temp_output_dir,
        )

        # Setup with custom device map
        custom_device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 0,
            "model.layers.2": 1,
            "model.layers.3": 1,
            "model.norm": 1,
            "lm_head": 1,
        }

        # Execute setup with custom device map
        with patch(
            "src.training.adapter_trainer.prepare_model_for_kbit_training",
            return_value=mock_model,
        ), patch("src.training.adapter_trainer.LoraConfig"), patch(
            "src.training.adapter_trainer.get_peft_model", return_value=mock_model
        ):
            trainer.setup(device_map=custom_device_map)

        # Verify custom device map was used
        call_kwargs = mock_auto_model.from_pretrained.call_args[1]
        assert call_kwargs["device_map"] == custom_device_map

    @patch(
        "src.training.adapter_trainer.logger"
    )  # Add logger patch to avoid actual logging
    def test_mixed_precision_hooks(self, mock_logger, temp_output_dir):
        """Test mixed precision training hooks."""
        # Create a trainer with mixed precision enabled
        with patch(
            "src.training.adapter_trainer.get_expert_config"
        ) as mock_get_expert_config:
            mock_config = MagicMock()
            mock_config.additional_params = {}
            mock_get_expert_config.return_value = mock_config

            trainer = LoRAAdapterTrainer(
                base_model_path="models/mixtral-8x7b",
                expert_type="REASON",
                output_dir=temp_output_dir,
                use_mixed_precision=True,
            )

        # Create mock trainer and mp_trainer
        trainer.trainer = MagicMock()
        trainer.mp_trainer = MagicMock()
        trainer.mp_trainer.get_ctx_manager.return_value = MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock()
        )
        trainer.mp_trainer.backward = MagicMock()

        # Set up the original compute_loss method with the correct signature
        def original_compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            if return_outputs:
                return torch.tensor(1.0), {"logits": torch.tensor([1.0])}
            return torch.tensor(1.0)

        trainer.trainer.compute_loss = original_compute_loss

        # Apply mixed precision hooks
        trainer._setup_mixed_precision_hooks()

        # Get the wrapped compute_loss function
        wrapped_compute_loss = trainer.trainer.compute_loss

        # Test with return_outputs=False
        model = MagicMock()
        inputs = {"input_ids": torch.ones(2, 10)}

        # Call the wrapped function
        loss = wrapped_compute_loss(model, inputs)

        # Verify the loss is returned and backward was called
        assert isinstance(loss, torch.Tensor)
        trainer.mp_trainer.backward.assert_called_once_with(loss)

        # Reset the mock for the next test
        trainer.mp_trainer.backward.reset_mock()

        # Test with return_outputs=True
        loss, outputs = wrapped_compute_loss(model, inputs, return_outputs=True)

        # Verify both loss and outputs are returned and backward was called
        assert isinstance(loss, torch.Tensor)
        assert isinstance(outputs, dict)
        trainer.mp_trainer.backward.assert_called_once_with(loss)
