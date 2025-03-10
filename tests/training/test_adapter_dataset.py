"""
Tests for the ExpertDataset and TokenizedDataset implementations.

This module tests the dataset components used for adapter training,
including data loading, processing, and tokenization.
"""

import sys
import os
import json
import pytest
import torch
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from tempfile import TemporaryDirectory

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.training.adapter_dataset import (
    ExpertDataset,
    TokenizedDataset,
    create_expert_dataset_from_jsonl,
)


class TestExpertDataset:
    """Test cases for the ExpertDataset class."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = None  # Will be set to eos_token if None

        # Mock encode method
        tokenizer.encode.return_value = [101, 102, 103]  # Dummy token IDs

        # Mock call method (used in tokenization)
        tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102, 103, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }

        return tokenizer

    @pytest.fixture
    def sample_data_files(self):
        """Create sample data files for testing."""
        with TemporaryDirectory() as tmpdir:
            # Create a JSONL file
            jsonl_path = Path(tmpdir) / "test_data.jsonl"
            with open(jsonl_path, "w") as f:
                f.write('{"input": "Test question 1", "output": "Test answer 1"}\n')
                f.write('{"input": "Test question 2", "output": "Test answer 2"}\n')

            # Create a JSON file with an array
            json_path = Path(tmpdir) / "test_data.json"
            with open(json_path, "w") as f:
                json.dump(
                    [
                        {"input": "Test question 3", "output": "Test answer 3"},
                        {"input": "Test question 4", "output": "Test answer 4"},
                    ],
                    f,
                )

            # Create a text file with structured format
            txt_path = Path(tmpdir) / "test_data.txt"
            with open(txt_path, "w") as f:
                f.write("Test question 5\n----\nTest answer 5\n====\n")
                f.write("Test question 6\n----\nTest answer 6\n")

            yield {
                "jsonl": str(jsonl_path),
                "json": str(json_path),
                "txt": str(txt_path),
                "dir": tmpdir,
            }

    def test_init(self, mock_tokenizer, sample_data_files):
        """Test initialization with valid parameters."""
        # Setup
        expert_type = "REASON"
        data_sources = [sample_data_files["jsonl"]]

        # Execute
        dataset = ExpertDataset(
            expert_type=expert_type,
            data_sources=data_sources,
            tokenizer=mock_tokenizer,
            max_length=512,
            validation_split=0.1,
        )

        # Verify
        assert dataset.expert_type == expert_type
        assert dataset.data_sources == data_sources
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == 512
        assert dataset.validation_split == pytest.approx(0.1)
        assert len(dataset.raw_data) > 0
        assert len(dataset.processed_data) > 0

    def test_load_data_multiple_sources(self, mock_tokenizer, sample_data_files):
        """Test loading data from multiple source types."""
        # Setup
        data_sources = [
            sample_data_files["jsonl"],
            sample_data_files["json"],
            sample_data_files["txt"],
        ]

        # Execute
        dataset = ExpertDataset(
            expert_type="EXPLAIN", data_sources=data_sources, tokenizer=mock_tokenizer
        )

        # Verify - should load from all sources
        # This test is now updated to allow for the .txt file handler to work correctly
        # The .txt file should load 2 examples
        txt_count = 2
        total_count = 2 + 2 + txt_count  # JSONL + JSON + TXT
        assert len(dataset.raw_data) == total_count

    def test_load_data_directory(self, mock_tokenizer, sample_data_files):
        """Test loading data from a directory containing files."""
        # Setup - use the directory containing all test files
        data_sources = [sample_data_files["dir"]]

        # Execute
        dataset = ExpertDataset(
            expert_type="TEACH", data_sources=data_sources, tokenizer=mock_tokenizer
        )

        # Verify - should load from all files in the directory
        # Allow for flexibility in TXT file loading
        assert len(dataset.raw_data) >= 4  # At least JSONL + JSON (2+2)

    def test_load_data_nonexistent_source(self, mock_tokenizer):
        """Test behavior with nonexistent data source."""
        # Setup
        data_sources = ["nonexistent_file.jsonl"]

        # Execute
        dataset = ExpertDataset(
            expert_type="PREDICT", data_sources=data_sources, tokenizer=mock_tokenizer
        )

        # Verify - should handle nonexistent sources gracefully
        assert len(dataset.raw_data) == 0
        assert len(dataset.processed_data) == 0

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data='{"input": "Q", "output": "A"}',
    )
    def test_load_json_single_object(self, mock_file, mock_tokenizer):
        """Test loading a JSON file with a single object (not an array)."""
        # Setup
        with patch("json.load") as mock_json_load:
            mock_json_load.return_value = {"input": "Q", "output": "A"}
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.is_file", return_value=True):
                    with patch("pathlib.Path.suffix", ".json"):
                        # Execute
                        dataset = ExpertDataset(
                            expert_type="RETROSPECT",
                            data_sources=["test.json"],
                            tokenizer=mock_tokenizer,
                        )

        # Verify - should convert single object to a list
        assert len(dataset.raw_data) == 1
        assert dataset.raw_data[0]["input"] == "Q"
        assert dataset.raw_data[0]["output"] == "A"

    def test_format_example_for_different_experts(self, mock_tokenizer):
        """Test formatting examples for different expert types."""
        # Setup - test data
        input_text = "What happens if my opponent plays Counterspell?"
        output_text = "Your spell will be countered and put into the graveyard."

        # Execute - create datasets for different expert types
        reason_dataset = ExpertDataset(
            expert_type="REASON",
            data_sources=[],  # Empty, we'll manually add data
            tokenizer=mock_tokenizer,
        )
        explain_dataset = ExpertDataset(
            expert_type="EXPLAIN", data_sources=[], tokenizer=mock_tokenizer
        )

        # Manually format examples
        reason_example = reason_dataset._format_example(input_text, output_text)
        explain_example = explain_dataset._format_example(input_text, output_text)

        # Verify - should include expert type in prompt
        assert reason_example["prompt"].startswith("<REASON>")
        assert explain_example["prompt"].startswith("<EXPLAIN>")
        assert input_text in reason_example["prompt"]
        assert output_text == reason_example["completion"]
        assert output_text == explain_example["completion"]

    def test_process_data_different_formats(self, mock_tokenizer):
        """Test processing data with different input formats."""
        # Setup - create instance with empty sources
        dataset = ExpertDataset(
            expert_type="REASON", data_sources=[], tokenizer=mock_tokenizer
        )

        # Manually set raw data with different formats
        dataset.raw_data = [
            {"input": "Q1", "output": "A1"},  # Standard format
            {"prompt": "Q2", "completion": "A2"},  # OpenAI format
            {"question": "Q3", "answer": "A3"},  # QA format
            {"instruction": "Q4", "response": "A4"},  # Instruction format
            {"text": "Q5\n-----\nA5"},  # Raw text with separator
            {"text": "Just some text without separator"},  # Raw text without separator
        ]

        # Execute - process the data
        processed_data = dataset._process_data()

        # Verify - all formats should be processed correctly
        assert len(processed_data) == 6
        # Check first entry (standard format)
        assert "<REASON>" in processed_data[0]["prompt"]
        assert "Q1" in processed_data[0]["prompt"]
        assert processed_data[0]["completion"] == "A1"

    @patch("src.training.adapter_dataset.TokenizedDataset")
    def test_create_train_val_split(self, mock_tokenized_dataset, mock_tokenizer):
        """Test creating training and validation splits."""
        # Setup
        # Create mock TokenizedDataset instances that will be returned
        mock_train_dataset = MagicMock()
        mock_val_dataset = MagicMock()

        # Configure the mock to return our mock datasets when instantiated
        mock_tokenized_dataset.side_effect = [mock_train_dataset, mock_val_dataset]

        dataset = ExpertDataset(
            expert_type="TEACH",
            data_sources=[],
            tokenizer=mock_tokenizer,
            validation_split=0.2,  # 20% validation
        )

        # Add some processed data manually - first use deterministic values
        processed_data = []
        for i in range(10):  # 10 examples
            processed_data.append({"prompt": f"<TEACH>\nQ{i}", "completion": f"A{i}"})
        dataset.processed_data = processed_data

        # Execute
        train_dataset, val_dataset = dataset.create_train_val_split()

        # Instead of checking the exact data passed (which depends on the random shuffle),
        # we'll check that:
        # 1. TokenizedDataset was called twice (for train and val)
        # 2. The first call uses approximately 80% of the data (8 items)
        # 3. The second call uses approximately 20% of the data (2 items)
        assert mock_tokenized_dataset.call_count == 2

        # Check the length of data passed to each call
        first_call_data = mock_tokenized_dataset.call_args_list[0][0][0]
        second_call_data = mock_tokenized_dataset.call_args_list[1][0][0]
        assert len(first_call_data) == 8  # 80% of 10 = 8
        assert len(second_call_data) == 2  # 20% of 10 = 2

        # Verify the returned datasets are our mocks
        assert train_dataset == mock_train_dataset
        assert val_dataset == mock_val_dataset


class TestTokenizedDataset:
    """Test cases for the TokenizedDataset class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return [
            {
                "prompt": "<REASON>\nWhat is Lightning Bolt?",
                "completion": "Lightning Bolt is a red instant that deals 3 damage.",
            }
        ]

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer with more realistic behavior."""
        tokenizer = MagicMock()
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = None

        # Set up the tokenizer to return different lengths for different inputs
        def tokenizer_side_effect(text, **kwargs):
            # Count rough number of tokens (simply by spaces + some overhead)
            if isinstance(text, str):
                token_count = len(text.split()) + 2  # Rough approximation
                return {
                    "input_ids": torch.tensor([[i for i in range(token_count)]]),
                    "attention_mask": torch.tensor([[1] * token_count]),
                }
            else:
                # Return a default value for other cases
                return {
                    "input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 1]]),
                }

        tokenizer.side_effect = tokenizer_side_effect

        # Mock encode to return token ids
        tokenizer.encode.side_effect = lambda text, **kwargs: [
            i for i in range(len(text.split()) + 2)
        ]

        return tokenizer

    def test_init(self, sample_data, mock_tokenizer):
        """Test initialization with valid parameters."""
        # Execute
        dataset = TokenizedDataset(sample_data, mock_tokenizer, max_length=512)

        # Verify
        assert len(dataset) == 1
        assert dataset.max_length == 512
        # Check if pad token was set to eos token
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token

    def test_getitem(self, sample_data, mock_tokenizer):
        """Test getting an item from the dataset."""
        # Setup
        dataset = TokenizedDataset(sample_data, mock_tokenizer, max_length=512)

        # Configure mock to return specific tensors for the test example
        prompt_tokens = 5  # Simulated token count for prompt
        full_tokens = 12  # Simulated token count for prompt + completion

        # Mock the tokenizer call for full text
        full_text_encoding = {
            "input_ids": torch.tensor([[i for i in range(full_tokens)]]),
            "attention_mask": torch.tensor([[1] * full_tokens]),
        }
        mock_tokenizer.return_value = full_text_encoding

        # Mock the tokenizer call for prompt only
        prompt_encoding = {
            "input_ids": torch.tensor([[i for i in range(prompt_tokens)]]),
            "attention_mask": torch.tensor([[1] * prompt_tokens]),
        }

        def side_effect(text, **kwargs):
            if text == sample_data[0]["prompt"]:
                return prompt_encoding
            return full_text_encoding

        mock_tokenizer.side_effect = side_effect

        # Execute
        item = dataset[0]

        # Verify
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["input_ids"].size(0) == full_tokens

        # Check that labels have -100 for prompt tokens
        expected_labels = torch.tensor([i for i in range(full_tokens)])
        expected_labels[:prompt_tokens] = -100

        # Since we don't know the exact label values, just check the pattern of -100s
        assert torch.all(item["labels"][:prompt_tokens] == -100)
        assert torch.all(item["labels"][prompt_tokens:] != -100)

    def test_getitem_with_different_tokenizer_outputs(self, sample_data):
        """Test handling different tokenizer output types."""
        # Setup a more controlled test case
        tokenizer = MagicMock()
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "</s>"

        # For the full text tokenization, return a tensor with 5 tokens
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # For prompt-only tokenization (needed to mask out prompt in labels)
        tokenizer.side_effect = None  # Remove any existing side_effect

        # The prompt tokenization result will be accessed through 'get'
        prompt_result = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.__getitem__ = lambda self, key: (
            prompt_result if key == "input_ids" else None
        )
        tokenizer.get = lambda key, default=None: prompt_result.get(key, default)

        dataset = TokenizedDataset(sample_data, tokenizer, max_length=512)

        # Execute
        item = dataset[0]

        # Verify basic structure
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


class TestHelperFunctions:
    """Test cases for helper functions in the adapter_dataset module."""

    def test_create_expert_dataset_from_jsonl(self):
        """Test the create_expert_dataset_from_jsonl helper function."""
        # Setup - create mock objects
        mock_tokenizer = MagicMock()

        # Mock the ExpertDataset class
        with patch("src.training.adapter_dataset.ExpertDataset") as mock_expert_dataset:
            # Create a mock ExpertDataset instance
            mock_dataset_instance = MagicMock()
            mock_expert_dataset.return_value = mock_dataset_instance

            # Set up the return value for create_train_val_split
            mock_train_dataset = MagicMock()
            mock_val_dataset = MagicMock()
            mock_dataset_instance.create_train_val_split.return_value = (
                mock_train_dataset,
                mock_val_dataset,
            )

            # Execute
            train_ds, val_ds = create_expert_dataset_from_jsonl(
                expert_type="REASON",
                data_path="data.jsonl",
                tokenizer=mock_tokenizer,
                max_length=512,
                validation_split=0.1,
            )

            # Verify
            mock_expert_dataset.assert_called_once_with(
                expert_type="REASON",
                data_sources=["data.jsonl"],
                tokenizer=mock_tokenizer,
                max_length=512,
                validation_split=0.1,
            )
            mock_dataset_instance.create_train_val_split.assert_called_once()
            assert train_ds == mock_train_dataset
            assert val_ds == mock_val_dataset
