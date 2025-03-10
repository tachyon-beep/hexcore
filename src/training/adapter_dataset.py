"""
Dataset utilities for expert adapter training with specialized MTG knowledge.

This module provides dataset preparation tools for training LoRA adapters
for different expert types (REASON, EXPLAIN, TEACH, PREDICT, RETROSPECT).
"""

import os
import json
import logging
import torch
import random
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, TypeVar, cast
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding

logger = logging.getLogger(__name__)

# Define a type alias for both tokenizer types
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class ExpertDataset:
    """
    Dataset manager for loading, processing, and preparing expert-specific datasets.

    Features:
    - Loads data from multiple sources (files, directories)
    - Formats data for specific expert types
    - Creates train/validation splits
    - Handles tokenization
    """

    def __init__(
        self,
        expert_type: str,
        data_sources: List[str],
        tokenizer: TokenizerType,
        max_length: int = 2048,
        validation_split: float = 0.1,
    ):
        """
        Initialize dataset manager.

        Args:
            expert_type: Expert type (REASON, EXPLAIN, TEACH, PREDICT, RETROSPECT)
            data_sources: List of data source paths (files or directories)
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length for tokenization
            validation_split: Portion of data to use for validation (0.0 to 1.0)
        """
        self.expert_type = expert_type
        self.data_sources = data_sources
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.validation_split = validation_split

        # Load data
        self.raw_data = self._load_data()

        # Process for specific expert type
        self.processed_data = self._process_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from all sources.

        Returns:
            List of data entries
        """
        all_data = []

        for source in self.data_sources:
            source_path = Path(source)

            if not source_path.exists():
                logger.warning(f"Data source does not exist: {source}")
                continue

            if source_path.is_file():
                # Single file
                file_data = self._load_file(source_path)
                all_data.extend(file_data)
            elif source_path.is_dir():
                # Directory - load all JSON files
                for file_path in source_path.glob("**/*.json"):
                    file_data = self._load_file(file_path)
                    all_data.extend(file_data)

                # Also check for JSONL files
                for file_path in source_path.glob("**/*.jsonl"):
                    file_data = self._load_file(file_path)
                    all_data.extend(file_data)

                # Check for TXT files with formatted examples
                for file_path in source_path.glob("**/*.txt"):
                    file_data = self._load_text_file(file_path)
                    all_data.extend(file_data)

        logger.info(
            f"Loaded {len(all_data)} raw data entries from {len(self.data_sources)} sources"
        )
        return all_data

    def _load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load data from a file.

        Args:
            file_path: Path to the file

        Returns:
            List of data entries
        """
        try:
            # Check file extension
            suffix = file_path.suffix.lower()

            if suffix == ".json":
                # JSON file - could be an array or a single object
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Convert to list if it's a single object
                if isinstance(data, dict):
                    data = [data]

                return data

            elif suffix == ".jsonl":
                # JSONL file - one JSON object per line
                data = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            entry = json.loads(line)
                            data.append(entry)
                return data

            elif suffix == ".txt":
                # TXT file - use specialized loader
                return self._load_text_file(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return []

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return []

    def _load_text_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load structured data from a text file. Expects a format with separators
        for input/output pairs.

        Args:
            file_path: Path to text file

        Returns:
            List of data entries
        """
        try:
            data = []
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Split by example separator (assuming examples are separated by '====' or similar)
            examples = content.split("====")
            for i, example in enumerate(examples):
                example = example.strip()
                if not example:
                    continue

                # Try to split input and output
                parts = example.split("----")

                if len(parts) == 2:
                    input_text = parts[0].strip()
                    output_text = parts[1].strip()

                    data.append(
                        {
                            "input": input_text,
                            "output": output_text,
                            "source": str(file_path),
                            "id": f"{file_path.stem}_{i}",
                        }
                    )
                else:
                    # If no clear separator, treat entire text as a single example
                    data.append(
                        {
                            "text": example,
                            "source": str(file_path),
                            "id": f"{file_path.stem}_{i}",
                        }
                    )

            return data

        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []

    def _process_data(self) -> List[Dict[str, str]]:
        """
        Process raw data for the specific expert type.

        Returns:
            List of processed data entries with prompt and completion fields
        """
        processed = []

        for entry in self.raw_data:
            try:
                # Try different common formats
                if "input" in entry and "output" in entry:
                    # Standard input/output format
                    processed_entry = self._format_example(
                        entry["input"], entry["output"]
                    )
                    processed.append(processed_entry)

                elif "prompt" in entry and "completion" in entry:
                    # OpenAI-style format
                    processed_entry = self._format_example(
                        entry["prompt"], entry["completion"]
                    )
                    processed.append(processed_entry)

                elif "question" in entry and "answer" in entry:
                    # QA format
                    processed_entry = self._format_example(
                        entry["question"], entry["answer"]
                    )
                    processed.append(processed_entry)

                elif "instruction" in entry and "response" in entry:
                    # Instruction format
                    processed_entry = self._format_example(
                        entry["instruction"], entry["response"]
                    )
                    processed.append(processed_entry)

                elif "text" in entry:
                    # Raw text format - if it contains special markers, try to split it
                    text = entry["text"]
                    if "-----" in text:
                        parts = text.split("-----", 1)
                        processed_entry = self._format_example(
                            parts[0].strip(), parts[1].strip()
                        )
                        processed.append(processed_entry)
                    else:
                        # Use the text as completion only with a generic prompt
                        processed_entry = self._format_example(
                            f"You are an MTG AI Assistant specialized in {self.expert_type} mode. Respond to the following request:",
                            text,
                        )
                        processed.append(processed_entry)

                else:
                    # Unknown format, try to parse it based on structure
                    logger.warning(f"Unknown data format in entry: {entry.keys()}")

            except Exception as e:
                logger.error(f"Error processing entry: {str(e)}")

        logger.info(f"Processed {len(processed)} entries for {self.expert_type} expert")
        return processed

    def _format_example(self, input_text: str, output_text: str) -> Dict[str, str]:
        """
        Format an example for the specific expert type.

        Args:
            input_text: Input text (question, prompt)
            output_text: Output text (answer, completion)

        Returns:
            Dictionary with prompt and completion fields
        """
        # Add expert tag to the prompt
        prompt = f"<{self.expert_type}>\n{input_text}"

        return {"prompt": prompt, "completion": output_text}

    def create_train_val_split(self) -> Tuple[Dataset, Dataset]:
        """
        Create training and validation datasets.

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Shuffle data deterministically for testing consistency
        data = self.processed_data.copy()
        random.seed(42)
        random.shuffle(data)

        # Split data
        split_idx = int(len(data) * (1 - self.validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        logger.info(
            f"Split data into {len(train_data)} training and {len(val_data)} validation examples"
        )

        # Create datasets
        train_dataset = TokenizedDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = TokenizedDataset(val_data, self.tokenizer, self.max_length)

        return train_dataset, val_dataset


class TokenizedDataset(Dataset):
    """
    Dataset class that tokenizes prompt and completion pairs.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: TokenizerType,
        max_length: int = 2048,
    ):
        """
        Initialize dataset.

        Args:
            data: List of dictionaries with prompt and completion fields
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Check if tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tokenized sample with properly formatted labels for causal LM training.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        example = self.data[idx]
        prompt, completion = example["prompt"], example["completion"]

        # Safely add EOS token if it exists and is a string
        if isinstance(self.tokenizer.eos_token, str):
            completion_with_eos = completion + self.tokenizer.eos_token
        else:
            # Fallback if eos_token is None or not a string
            completion_with_eos = completion

        # Tokenize prompt and completion separately
        prompt_tokens = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True, return_tensors="pt"
        )

        # Use Any type to bypass the type checking for dictionary access
        prompt_ids = cast(Any, prompt_tokens)["input_ids"]
        if not isinstance(prompt_ids, torch.Tensor):
            # Use Any again to bypass type checking for indexing
            prompt_ids = torch.tensor(cast(Any, prompt_ids)[0]).unsqueeze(0)

        # Do the same for completion tokens
        completion_tokens = self.tokenizer(
            completion_with_eos,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        completion_ids = cast(Any, completion_tokens)["input_ids"]
        if not isinstance(completion_ids, torch.Tensor):
            completion_ids = torch.tensor(cast(Any, completion_ids)[0]).unsqueeze(0)

        # Get lengths from tensors
        prompt_length = prompt_ids.size(1)
        completion_length = completion_ids.size(1)

        # Ensure we don't exceed max_length when combined
        total_length = prompt_length + completion_length
        if total_length > self.max_length:
            # Prioritize completion by truncating prompt if needed
            prompt_length = max(1, self.max_length - completion_length)

        # Extract the parts we need
        prompt_part = prompt_ids[0, :prompt_length]
        completion_part = completion_ids[
            0, : min(completion_length, self.max_length - prompt_length)
        ]

        # Combine into a single sequence
        input_ids = torch.cat([prompt_part, completion_part], dim=0)

        # Create attention mask
        attention_mask = torch.ones_like(input_ids)

        # Pad if needed to reach max_length
        if input_ids.size(0) < self.max_length:
            padding_length = self.max_length - input_ids.size(0)
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else 0
            )

            input_ids = torch.cat(
                [
                    input_ids,
                    torch.ones(padding_length, dtype=torch.long) * pad_token_id,
                ],
                dim=0,
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(padding_length, dtype=torch.long)],
                dim=0,
            )

        # Create labels: -100 for prompt tokens (they won't contribute to loss),
        # actual ids for completion tokens
        labels = input_ids.clone()
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_expert_dataset_from_jsonl(
    expert_type: str,
    data_path: str,
    tokenizer: TokenizerType,
    max_length: int = 2048,
    validation_split: float = 0.1,
) -> Tuple[Dataset, Dataset]:
    """
    Create expert dataset from a JSONL file.

    Args:
        expert_type: The type of expert (REASON, EXPLAIN, etc.)
        data_path: Path to JSONL file
        tokenizer: Tokenizer for the model
        max_length: Maximum sequence length
        validation_split: Portion to use for validation

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Create dataset manager
    dataset_manager = ExpertDataset(
        expert_type=expert_type,
        data_sources=[data_path],
        tokenizer=tokenizer,
        max_length=max_length,
        validation_split=validation_split,
    )

    return dataset_manager.create_train_val_split()
