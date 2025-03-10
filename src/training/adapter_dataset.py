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
        Get a tokenized sample.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        example = self.data[idx]
        prompt, completion = example["prompt"], example["completion"]

        # Construct full text (prompt + completion)
        text = f"{prompt}{completion}{self.tokenizer.eos_token}"

        # Tokenize
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Extract tensors - handle both regular and fast tokenizers
        input_ids = encodings["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.squeeze(0)
        else:
            # For other tokenizer types, convert to tensor if needed
            input_ids = torch.tensor(input_ids)

        attention_mask = encodings["attention_mask"]
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.squeeze(0)
        else:
            # For other tokenizer types, convert to tensor if needed
            attention_mask = torch.tensor(attention_mask)

        # Create labels for causal LM (same as input_ids)
        labels = input_ids.clone()

        # Mask out the prompt part in labels (set to -100)
        prompt_tokens = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True
        )

        # Safely determine prompt length for different tokenizer outputs
        prompt_length = 0

        # Extract input_ids from the prompt tokenization
        prompt_input_ids = prompt_tokens.get("input_ids", [])

        # Determine length based on the type of output
        if isinstance(prompt_input_ids, torch.Tensor):
            prompt_length = (
                prompt_input_ids.size(1)
                if prompt_input_ids.dim() > 1
                else prompt_input_ids.size(0)
            )
        elif isinstance(prompt_input_ids, list):
            # For list outputs
            if prompt_input_ids and isinstance(prompt_input_ids[0], list):
                prompt_length = len(prompt_input_ids[0])
            elif prompt_input_ids:
                prompt_length = len(prompt_input_ids)

        # Ensure prompt_length is within bounds
        prompt_length = min(prompt_length, labels.size(0))

        # Mask prompt tokens in the labels
        if prompt_length > 0:
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
