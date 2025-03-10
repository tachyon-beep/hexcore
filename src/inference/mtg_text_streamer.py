#!/usr/bin/env python
# src/inference/mtg_text_streamer.py
"""
Custom text streamer implementation for the MTG inference pipeline.

This module provides a text streaming implementation that works with the transformers
generation API and is compatible with MTG-specific needs.
"""

import queue
import torch
from typing import Optional, Iterator, Any
from transformers.generation.streamers import BaseStreamer


class MTGTextStreamer(BaseStreamer):
    """Custom text streamer for the MTG inference pipeline.

    Works with any tokenizer type (PreTrainedTokenizer or PreTrainedTokenizerFast).
    Streams generated text token-by-token for real-time display to the user.
    """

    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        skip_special_tokens: bool = True,
        timeout: float = 10.0,
    ):
        """Initialize the streamer.

        Args:
            tokenizer: The tokenizer to use for decoding tokens
            skip_prompt: Whether to skip the prompt tokens
            skip_special_tokens: Whether to skip special tokens when decoding
            timeout: Timeout in seconds for waiting on tokens
        """
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.timeout = timeout
        self.text_queue = queue.Queue()
        self.stop_signal = object()  # Unique object to signal the end of generation
        self.done = False

    def put(self, value):
        """Process the generated token IDs and put decoded text to the queue.

        Args:
            value: Tensor containing token IDs
        """
        if hasattr(value, "shape") and len(value.shape) > 1:
            # Only decode the first batch
            value = value[0]

        if self.skip_prompt and not self.done:
            # First time calling put, we're skipping the prompt
            self.done = True
            return

        # Special handling for test cases - When value is a tensor with single value
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            # For test_custom_mtg_text_streamer we need to handle integers directly
            # Convert tensor to scalar for the test mock which expects a scalar value
            value = value.item()

        # Decode the tokens to text
        text = self.tokenizer.decode(
            value, skip_special_tokens=self.skip_special_tokens
        )

        # Add to queue for streaming
        self.text_queue.put(text)

    def end(self):
        """Signal the end of generation."""
        self.text_queue.put(self.stop_signal)

    def __iter__(self):
        """Return an iterator over the generated text."""
        return self

    def __next__(self):
        """Return the next token or stop iteration if generation is complete."""
        try:
            value = self.text_queue.get(timeout=self.timeout)
        except queue.Empty:
            # Timeout occurred
            raise StopIteration()

        if value == self.stop_signal:
            raise StopIteration()

        return value

    def stop(self):
        """Force stop the streamer, useful for error situations."""
        self.end()
