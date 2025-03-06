# test_model_loading.py

import logging
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models.model_loader import load_quantized_model, create_optimized_device_map
from utils.gpu_utils import monitor_memory, optimize_memory, get_gpu_info

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Test loading the Mixtral model with our custom memory optimization."""

    # First, check available GPUs
    logger.info("Checking GPU availability...")
    gpu_info = get_gpu_info()
    logger.info(f"GPU information: {gpu_info}")

    if not gpu_info.get("devices"):
        logger.error("No GPUs detected. This test requires GPUs to run.")
        return

    # Log initial memory state
    logger.info("Initial memory state:")
    initial_memory = monitor_memory()
    for device, stats in initial_memory.items():
        logger.info(f"  {device}: {stats}")

    try:
        # Create custom device map for dual GPUs
        logger.info("Creating optimized device map...")
        device_map = create_optimized_device_map()

        # Load the model with quantization
        logger.info("Loading Mixtral model with quantization...")
        model, tokenizer = load_quantized_model(device_map=device_map)

        # Log memory after loading
        logger.info("Memory after model loading:")
        loaded_memory = monitor_memory()
        for device, stats in loaded_memory.items():
            logger.info(f"  {device}: {stats}")

        # Perform a simple inference test
        logger.info("Testing basic inference...")
        input_text = "Magic: The Gathering is a card game where"

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        if "cuda:0" in str(next(model.parameters()).device):
            input_ids = input_ids.to("cuda:0")

        import torch

        with torch.no_grad():
            output = model.generate(
                input_ids, max_length=100, do_sample=True, temperature=0.7
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")

        # Cleanup
        logger.info("Cleaning up memory...")
        optimize_memory()

        logger.info("Test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Error during model loading test: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    main()
