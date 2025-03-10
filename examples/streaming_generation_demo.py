#!/usr/bin/env python
# examples/streaming_generation_demo.py
"""
Demonstration of streaming generation with the MTG AI Assistant.

This example shows how to use the streaming generation functionality
of the MTG AI Assistant, including:
1. Basic streaming with token-by-token display
2. Async streaming with progress tracking
3. Conversation-based generation with context
"""

import sys
import os
import time
import asyncio
import queue
import torch
import threading
from typing import Dict, Any, List, Optional, AsyncIterator, Iterator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components
from src.models.model_loader import load_quantized_model


# Define a custom streamer to avoid type errors
class MTGTextStreamer(BaseStreamer):
    """Custom text streamer that works with any tokenizer type."""

    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        skip_special_tokens: bool = False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.text_queue = queue.Queue()
        self.stop_signal = None
        self.done = False

    def put(self, value):
        """Add new token to queue."""
        if len(value.shape) > 1:
            # Only decode the first batch
            value = value[0]

        if self.skip_prompt and not self.done:
            # First time calling put, we should ignore the prompt
            self.done = True
            return

        text = self.tokenizer.decode(
            value, skip_special_tokens=self.skip_special_tokens
        )

        # Add to queue for streaming
        self.text_queue.put(text)

    def end(self):
        """Signal end of generation."""
        self.text_queue.put(self.stop_signal)

    def __iter__(self):
        """Iterate through generated tokens."""
        return self

    def __next__(self):
        """Return next token or stop iteration."""
        value = self.text_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        return value


def colorize(text, color_code):
    """Add color to terminal text."""
    return f"\033[{color_code}m{text}\033[0m"


def progress_callback(state: Dict[str, Any]):
    """Callback function to display generation progress."""
    # Only print every 10 tokens to avoid spamming the console
    if state.get("tokens_generated", 0) % 10 == 0 or state.get("is_complete", False):
        tokens = state.get("tokens_generated", 0)
        time_elapsed = state.get("generation_time", 0)
        tokens_per_sec = state.get("tokens_per_second", 0)

        status_msg = (
            f"Generated: {tokens} tokens in {time_elapsed:.2f}s "
            f"({tokens_per_sec:.1f} tokens/sec)"
        )

        # Print progress on the same line (carriage return without newline)
        print(f"\r{status_msg}", end="", flush=True)

        # If complete, add a newline
        if state.get("is_complete", False):
            print("")

            # Print memory usage if available
            if "memory_before_mb" in state and "memory_after_mb" in state:
                memory_before = state["memory_before_mb"]
                memory_after = state["memory_after_mb"]
                memory_change = state["memory_change_mb"]
                memory_msg = (
                    f"Memory: {memory_before:.1f}MB → {memory_after:.1f}MB "
                    f"(Change: {memory_change:+.1f}MB)"
                )
                print(colorize(memory_msg, 36))  # Cyan


class StreamingDemoGenerator:
    """Demo class that handles streaming text generation."""

    def __init__(self, model_id="mistralai/Mixtral-8x7B-v0.1"):
        """Initialize the demo generator with a model."""
        print(f"Loading model: {model_id}...")

        # Try loading the actual Mixtral model if available
        try:
            self.model, self.tokenizer = load_quantized_model(
                model_id=model_id,
                quantization_type="4bit",
                device_map="auto",
                compute_dtype=torch.bfloat16,
            )
            print("Successfully loaded Mixtral 8x7B model with 4-bit quantization")
        except Exception as e:
            # Fall back to a smaller model if loading Mixtral fails
            print(f"Could not load Mixtral model: {e}")
            print("Falling back to smaller model for demo purposes...")
            self.model_id = "gpt2"  # Fallback to a small model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)

        # Store conversation history as a simple dictionary
        self.conversations = {}

    def _create_prompt(self, query: str, conversation_id: Optional[str] = None) -> str:
        """Create a prompt for the model, including conversation history if provided."""
        # If conversation_id is provided, include conversation history
        if conversation_id and conversation_id in self.conversations:
            history = self.conversations[conversation_id]
            prompt = f"Previous conversation:\n{history}\n\nNew query: {query}"
        else:
            prompt = f"Query: {query}"

        # For Mixtral, you might want to add a system prompt
        return f"You are an MTG AI Assistant that provides expert information about Magic: The Gathering rules and cards.\n\n{prompt}"

    def generate_streaming(
        self,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        conversation_id: Optional[str] = None,
        progress_callback=None,
    ) -> Iterator[str]:
        """Generate text in a streaming fashion, yielding tokens as they're generated."""
        # Create prompt with conversation history if provided
        prompt = self._create_prompt(query, conversation_id)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Set up the streamer - using our custom MTGTextStreamer
        streamer = MTGTextStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,  # Skip the prompt in the output
            skip_special_tokens=True,
        )

        # Generation parameters
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "streamer": streamer,
        }

        # Track generation stats for progress callback
        state = {
            "start_time": time.time(),
            "tokens_generated": 0,
            "is_complete": False,
            "generation_time": 0,
            "tokens_per_second": 0,
            "memory_before_mb": 0,
            "memory_after_mb": 0,
            "memory_change_mb": 0,
        }

        # Function to run generation in a separate thread
        def generate_thread():
            try:
                # Capture memory usage
                if torch.cuda.is_available():
                    state["memory_before_mb"] = torch.cuda.memory_allocated() / (
                        1024 * 1024
                    )

                # Generate
                self.model.generate(**gen_kwargs)

                # Update memory usage
                if torch.cuda.is_available():
                    state["memory_after_mb"] = torch.cuda.memory_allocated() / (
                        1024 * 1024
                    )
                    state["memory_change_mb"] = (
                        state["memory_after_mb"] - state["memory_before_mb"]
                    )
            except Exception as e:
                print(f"Error in generation: {e}")
                streamer.end()

        # Start generation in a separate thread
        thread = threading.Thread(target=generate_thread)
        thread.start()

        # Stream tokens and accumulate response
        accumulated_text = ""

        for new_text in streamer:
            # Update state
            state["tokens_generated"] += 1
            state["generation_time"] = time.time() - state["start_time"]
            if state["generation_time"] > 0:
                state["tokens_per_second"] = (
                    state["tokens_generated"] / state["generation_time"]
                )

            # Call progress callback if provided
            if progress_callback:
                progress_callback(state)

            # Accumulate text
            accumulated_text += new_text

            # Yield the token
            yield new_text

        # Mark generation as complete
        state["is_complete"] = True
        if progress_callback:
            progress_callback(state)

        # If using conversation_id, store the response
        if conversation_id:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = ""

            self.conversations[
                conversation_id
            ] += f"\nUser: {query}\nAssistant: {accumulated_text}\n"

    async def generate_streaming_async(
        self,
        query: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        conversation_id: Optional[str] = None,
        progress_callback=None,
    ) -> AsyncIterator[str]:
        """Asynchronous version of generate_streaming."""
        # This is a simple wrapper around the synchronous method
        # In a real implementation, you might want to use an actually async approach
        for token in self.generate_streaming(
            query, max_new_tokens, temperature, conversation_id, progress_callback
        ):
            yield token
            await asyncio.sleep(0)  # Allow other tasks to run


def demo_sync_streaming():
    """Demonstrate synchronous streaming generation."""
    print(colorize("\n===== Synchronous Streaming Demo =====", 33))  # Yellow

    # Create generator
    generator = StreamingDemoGenerator()

    # Query for demonstration
    query = "Explain the rules for casting spells with alternative costs like Force of Will."

    print(f"Query: {colorize(query, 32)}")  # Green
    print("Response (token by token):")

    # Use streaming generation
    start_time = time.time()
    response = ""

    for token in generator.generate_streaming(
        query=query,
        max_new_tokens=256,
        temperature=0.7,
        progress_callback=progress_callback,
    ):
        # Accumulate response
        response += token

        # Print token by token (green foreground)
        print(colorize(token, 32), end="", flush=True)

        # Small delay to simulate reading
        time.sleep(0.01)

    # Print total time
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

    # Print full response length
    print(f"Response length: {len(response)} characters")


async def demo_async_streaming():
    """Demonstrate asynchronous streaming generation."""
    print(colorize("\n===== Asynchronous Streaming Demo =====", 35))  # Magenta

    # Create generator (reusing the same model would be more efficient in practice)
    generator = StreamingDemoGenerator()

    # Query for demonstration
    query = "What happens when Cryptic Command counters a spell with Ward?"

    print(f"Query: {colorize(query, 35)}")  # Magenta
    print("Response (token by token):")

    # Use async streaming generation
    start_time = time.time()
    response = ""

    async for token in generator.generate_streaming_async(
        query=query,
        max_new_tokens=256,
        temperature=0.7,
        progress_callback=progress_callback,
    ):
        # Accumulate response
        response += token

        # Print token by token (magenta foreground)
        print(colorize(token, 35), end="", flush=True)

        # Small delay to simulate reading
        await asyncio.sleep(0.01)

    # Print total time
    print(f"\nTotal time: {time.time() - start_time:.2f} seconds")

    # Print full response length
    print(f"Response length: {len(response)} characters")


def demo_conversation_streaming():
    """Demonstrate streaming generation with conversation history."""
    print(colorize("\n===== Conversation Streaming Demo =====", 34))  # Blue

    # Create generator (reusing the same model would be more efficient in practice)
    generator = StreamingDemoGenerator()

    # Create a conversation ID
    conversation_id = "demo-conversation"

    # First query
    query1 = "How does the stack work in Magic: The Gathering?"

    print(f"First Query: {colorize(query1, 34)}")  # Blue
    print("Response:")

    response1 = ""
    for token in generator.generate_streaming(
        query=query1,
        max_new_tokens=256,
        temperature=0.7,
        conversation_id=conversation_id,  # Use conversation ID
    ):
        response1 += token
        print(colorize(token, 34), end="", flush=True)
        time.sleep(0.01)

    print("\n\n")

    # Second query that builds on the first
    query2 = "Can you give me an example of how triggered abilities go on the stack?"

    print(f"Follow-up Query: {colorize(query2, 36)}")  # Cyan
    print("Response (with conversation context):")

    response2 = ""
    for token in generator.generate_streaming(
        query=query2,
        max_new_tokens=256,
        temperature=0.7,
        conversation_id=conversation_id,  # Same conversation ID to maintain context
    ):
        response2 += token
        print(colorize(token, 36), end="", flush=True)
        time.sleep(0.01)

    print("\n")


def run_demos():
    """Run all demonstrations."""
    print(colorize("MTG AI Assistant Streaming Generation Demo", 1))
    print(
        "This demonstration shows different ways to use streaming generation with the MTG AI Assistant."
    )

    # Run sync demo
    demo_sync_streaming()

    # Run async demo using asyncio
    asyncio.run(demo_async_streaming())

    # Run conversation demo
    demo_conversation_streaming()

    print(colorize("\nAll demonstrations completed!", 32))


if __name__ == "__main__":
    run_demos()
