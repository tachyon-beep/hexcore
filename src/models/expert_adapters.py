# src/models/expert_adapters.py

from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft.peft_model import PeftModel
import torch
import os
import logging
import time
import gc
from collections import OrderedDict, defaultdict
from transformers import PreTrainedModel
from typing import Dict, Any, Optional, List, Union, cast, Set, Tuple

logger = logging.getLogger(__name__)


class ExpertAdapterManager:
    """
    Manages LoRA adapters for different experts in the MTG AI Assistant with advanced
    memory optimization through aggressive offloading and LRU caching.
    """

    def __init__(
        self, base_model, adapters_dir: str = "adapters", max_gpu_experts: int = 2
    ):
        """
        Initialize the expert adapter manager with memory optimization features.

        Args:
            base_model: Base language model to apply adapters to
            adapters_dir: Directory where adapter weights are stored
            max_gpu_experts: Maximum number of experts to keep in GPU memory at once
                            (smaller values save more memory but may increase latency)
        """
        self.base_model = base_model
        self.adapters_dir = adapters_dir
        self.current_adapter = None
        self.expert_configs = self._get_expert_configs()
        self.max_gpu_experts = max_gpu_experts

        # Map of expert types to their loaded adapter models
        self.expert_adapters = {}

        # Track expert usage for LRU caching
        self.expert_usage = OrderedDict()

        # Track memory statistics for each expert
        self.expert_memory_stats = {}

        # Track when experts were last used
        self.last_used_timestamp = {}

        # Track experts we predict might be needed soon
        self.prefetch_candidates = set()

        # Load all available adapters
        self._load_available_adapters()

    def _get_expert_configs(self) -> Dict[str, LoraConfig]:
        """
        Define LoRA configurations for each expert type.

        These configurations define the architecture of each expert adapter.
        """
        return {
            "REASON": LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "EXPLAIN": LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.10,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "TEACH": LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "PREDICT": LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
            "RETROSPECT": LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        }

    def _load_available_adapters(self):
        """
        Load all available adapters from the adapters directory.

        Unlike earlier versions, this doesn't automatically load all adapters into GPU memory.
        Instead, it just notes which adapters are available for loading when needed.
        """
        os.makedirs(self.adapters_dir, exist_ok=True)

        # Track which adapters are available on disk
        self.available_adapters = set()

        # Only load the first expert to GPU initially
        loaded_count = 0

        for expert_type in self.expert_configs.keys():
            adapter_path = os.path.join(self.adapters_dir, expert_type.lower())
            if os.path.exists(adapter_path):
                try:
                    # Record this adapter as available
                    self.available_adapters.add(expert_type)

                    # Only load to GPU if we haven't reached our limit
                    if loaded_count < self.max_gpu_experts:
                        logger.info(f"Loading adapter for {expert_type} to GPU")
                        self.expert_adapters[expert_type] = PeftModel.from_pretrained(
                            self.base_model, adapter_path
                        )
                        # Record initial usage for LRU
                        self._update_expert_usage(expert_type)
                        loaded_count += 1
                        logger.info(f"Successfully loaded adapter for {expert_type}")
                    else:
                        logger.info(
                            f"Adapter {expert_type} available but not preloaded to preserve memory"
                        )
                except Exception as e:
                    logger.error(f"Failed to load adapter for {expert_type}: {str(e)}")

    def _update_expert_usage(self, expert_type: str):
        """
        Update the LRU cache tracking for a given expert.

        Args:
            expert_type: The expert type that was just used
        """
        # Update LRU tracking
        if expert_type in self.expert_usage:
            # Remove and re-add to move to the end (most recently used)
            self.expert_usage.pop(expert_type, None)

        # Add to the end of the OrderedDict (most recently used position)
        self.expert_usage[expert_type] = True

        # Update timestamp
        self.last_used_timestamp[expert_type] = time.time()

        # Log the current LRU order for debugging
        lru_order = list(self.expert_usage.keys())
        logger.debug(f"Updated LRU order: {lru_order}")

    def _estimate_adapter_memory(self, expert_type: str) -> float:
        """
        Estimate the memory footprint of an expert adapter in GB.

        Args:
            expert_type: The expert type to estimate memory for

        Returns:
            Estimated memory usage in GB
        """
        # If we've measured this expert before, use that value
        if expert_type in self.expert_memory_stats:
            return self.expert_memory_stats[expert_type]

        # Otherwise, make an estimate based on the LoRA config
        if expert_type in self.expert_configs:
            config = self.expert_configs[expert_type]
            # Basic heuristic: rank * target modules * 4 bytes per parameter * 2 (up/down projections)
            # This is a simplified estimate - real usage will vary
            rank = getattr(config, "r", 16)  # Default to 16 if not specified
            target_module_count = len(
                getattr(config, "target_modules", ["q_proj", "v_proj"])
            )
            # Very rough estimate of parameters per module in millions
            params_per_module_M = 10
            estimated_params = (
                rank * target_module_count * params_per_module_M * 1_000_000 * 2
            )
            # Convert to GB (4 bytes per param)
            estimated_gb = (estimated_params * 4) / (1024 * 1024 * 1024)
            return estimated_gb

        # If all else fails, return a default estimate
        return 0.25  # 250MB default estimate

    def _measure_adapter_memory(self, expert_type: str) -> None:
        """
        Measure the actual memory footprint of an expert adapter.

        Args:
            expert_type: The expert type to measure memory for
        """
        if expert_type not in self.expert_adapters:
            return

        try:
            # Record starting memory
            torch.cuda.empty_cache()
            start_mem = torch.cuda.memory_allocated()

            # Move adapter to CPU to measure its size
            model = self.expert_adapters[expert_type]
            model.to("cpu")

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            # Record memory after removal
            end_mem = torch.cuda.memory_allocated()

            # Calculate difference in GB
            memory_diff_gb = (start_mem - end_mem) / (1024 * 1024 * 1024)
            if memory_diff_gb > 0:
                self.expert_memory_stats[expert_type] = memory_diff_gb
                logger.info(
                    f"Measured memory for {expert_type}: {memory_diff_gb:.2f} GB"
                )

            # Move back to original device if needed
            if self.current_adapter == expert_type:
                device = next(self.base_model.parameters()).device
                model.to(device)
        except Exception as e:
            logger.error(f"Error measuring adapter memory for {expert_type}: {str(e)}")

    def apply_adapter(
        self, expert_type: str, target_device: Optional[torch.device] = None
    ) -> bool:
        """
        Apply the adapter for the specified expert type, with enhanced memory management.

        Args:
            expert_type: The expert type to activate
            target_device: Optional specific device to place the adapter on
                           (if None, uses the base model's device)

        Returns:
            True if adapter was successfully applied, False otherwise
        """
        # If this expert is already active, just update usage tracking and return
        if self.current_adapter == expert_type:
            self._update_expert_usage(expert_type)
            return True

        # Determine the target device
        if target_device is None:
            # Try to get device from base model
            try:
                device = next(self.base_model.parameters()).device
            except (StopIteration, AttributeError):
                # If base_model has no parameters or is not a Module
                if torch.cuda.is_available():
                    device = torch.device("cuda:0")
                else:
                    device = torch.device("cpu")
                    logger.warning("No CUDA available, using CPU for adapter")
        else:
            device = target_device

        # First aggressively offload inactive experts to save memory
        # We pass our LRU data to make smart decisions about what to keep
        self.offload_inactive_experts(
            active_expert_type=expert_type,
            target_device=device,
            keep_recent=min(
                self.max_gpu_experts - 1, 1
            ),  # Keep some recently used experts
        )

        # Update usage tracking for LRU
        self._update_expert_usage(expert_type)

        # Check if adapter is already loaded to memory
        if expert_type in self.expert_adapters:
            # Use the pre-loaded adapter
            model = self.expert_adapters[expert_type]

            # Get current model device
            try:
                current_device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                current_device = None

            # Move model to target device if needed
            if current_device is None or current_device != device:
                logger.info(f"Moving adapter model for {expert_type} to {device}")
                try:
                    start_time = time.time()
                    model = model.to(device)
                    transfer_time = time.time() - start_time
                    logger.info(f"Device transfer took {transfer_time:.2f}s")

                    # Force memory cleanup after device transfer
                    torch.cuda.empty_cache()
                    self.expert_adapters[expert_type] = model  # Update the stored model
                except Exception as e:
                    logger.error(f"Failed to move adapter to {device}: {str(e)}")
                    return False

            # Replace the active model
            self.base_model = model
            self.current_adapter = expert_type

            # Verify all inputs and model parameters are on the same device
            self._verify_device_consistency(model, device)

            return True
        else:
            # Check if this adapter is available on disk but not loaded
            if expert_type in self.available_adapters:
                # Load the adapter from disk directly to the target device
                try:
                    logger.info(
                        f"Loading adapter for {expert_type} from disk to {device}"
                    )
                    adapter_path = os.path.join(self.adapters_dir, expert_type.lower())

                    start_time = time.time()
                    # Load directly to the target device
                    with torch.device(device):
                        model = PeftModel.from_pretrained(
                            self.base_model,
                            adapter_path,
                            torch_dtype=torch.float16,  # Use half precision to save memory
                        )

                    load_time = time.time() - start_time
                    logger.info(f"Adapter loading took {load_time:.2f}s")

                    self.expert_adapters[expert_type] = model
                    self.base_model = model
                    self.current_adapter = expert_type

                    # Measure memory usage for future reference
                    if expert_type not in self.expert_memory_stats:
                        self._estimate_adapter_memory(expert_type)

                    return True
                except Exception as e:
                    logger.error(
                        f"Failed to load adapter from disk for {expert_type}: {str(e)}"
                    )
                    # Fall back to creating a new adapter if loading fails

            # If not available on disk or loading failed, try to create a new adapter
            if expert_type in self.expert_configs:
                try:
                    config = self.expert_configs[expert_type]

                    # Get the original base model if current model is a PeftModel
                    base_model = self.base_model
                    if isinstance(base_model, PeftModel):
                        # Extract the base model to avoid type issues
                        base_model = base_model.get_base_model()

                    # Create new adapter with the proper base model
                    adapted_model = get_peft_model(
                        cast(PreTrainedModel, base_model), config
                    )
                    self.expert_adapters[expert_type] = adapted_model
                    self.base_model = adapted_model
                    self.current_adapter = expert_type
                    return True
                except Exception as e:
                    logger.error(
                        f"Failed to create adapter for {expert_type}: {str(e)}"
                    )
                    return False
            else:
                logger.warning(f"No configuration for expert type {expert_type}")
                return False

    def get_active_model(self):
        """Get the currently active model with applied adapter."""
        return self.base_model

    def get_memory_usage_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for expert adapters.

        Returns:
            Dictionary with memory usage statistics
        """
        # Count experts by device type safely
        experts_in_gpu = 0
        experts_in_cpu = 0

        for e in self.expert_adapters:
            try:
                # Get a fresh iterator each time to avoid StopIteration issues
                device_type = next(self.expert_adapters[e].parameters()).device.type
                if device_type == "cuda":
                    experts_in_gpu += 1
                elif device_type == "cpu":
                    experts_in_cpu += 1
            except (StopIteration, AttributeError):
                # Skip this expert if we can't determine the device
                continue

        stats = {
            "experts_in_gpu": experts_in_gpu,
            "experts_in_cpu": experts_in_cpu,
            "active_expert": self.current_adapter,
            "expert_memory_stats": self.expert_memory_stats,
            "lru_order": list(self.expert_usage.keys()),
        }

        # Add GPU memory usage if available
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)

        return stats

    def create_adapter(self, expert_type: str, training_data: Any) -> bool:
        """
        Create and train a new adapter for the specified expert type.

        Args:
            expert_type: The expert type to create an adapter for
            training_data: Training data for the adapter

        Returns:
            True if adapter was successfully created, False otherwise

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            f"Training adapter for {expert_type} is not implemented yet. "
            f"This feature will be available in a future version. "
            f"The system currently supports using pre-trained adapters only."
        )

    def prefetch_expert(self, expert_type: str) -> bool:
        """
        Prefetch an expert adapter to GPU memory in anticipation of future use.

        Args:
            expert_type: The expert type to prefetch

        Returns:
            True if prefetching was successful, False otherwise
        """
        # Skip if this expert is already active or loaded
        if self.current_adapter == expert_type:
            return True

        if expert_type in self.expert_adapters:
            # Check if it's already on GPU
            try:
                model = self.expert_adapters[expert_type]
                current_device = next(model.parameters()).device
                if current_device.type == "cuda":
                    logger.debug(
                        f"Expert {expert_type} already on GPU, no need to prefetch"
                    )
                    return True
            except Exception:
                pass

        # Update prefetch candidates
        self.prefetch_candidates.add(expert_type)

        # Check if we have room to prefetch (based on max_gpu_experts setting)
        gpu_expert_count = 0
        for e in self.expert_adapters:
            try:
                # Safely check device type - using a new parameter iterator each time to avoid StopIteration
                device_type = next(self.expert_adapters[e].parameters()).device.type
                if device_type == "cuda":
                    gpu_expert_count += 1
            except (StopIteration, AttributeError):
                # Skip this expert if we can't determine the device
                continue

        if gpu_expert_count >= self.max_gpu_experts:
            logger.debug(
                f"Not prefetching {expert_type} - already at max GPU experts ({gpu_expert_count})"
            )
            return False

        # Prefetch the expert if it's available
        if expert_type in self.available_adapters:
            try:
                logger.info(f"Prefetching expert {expert_type} to GPU")

                # Check if we already loaded to CPU
                if expert_type in self.expert_adapters:
                    # Move to GPU
                    model = self.expert_adapters[expert_type]
                    if torch.cuda.is_available():
                        device = torch.device("cuda:0")
                        model = model.to(device)
                        self.expert_adapters[expert_type] = model
                else:
                    # Load from disk
                    adapter_path = os.path.join(self.adapters_dir, expert_type.lower())
                    if torch.cuda.is_available():
                        device = torch.device("cuda:0")
                        with torch.device(device):
                            model = PeftModel.from_pretrained(
                                self.base_model, adapter_path, torch_dtype=torch.float16
                            )
                        self.expert_adapters[expert_type] = model

                # Update usage for LRU tracking
                self._update_expert_usage(expert_type)
                return True
            except Exception as e:
                logger.error(f"Error prefetching expert {expert_type}: {str(e)}")
                return False
        return False

    def offload_inactive_experts(
        self,
        active_expert_type,
        target_device=None,
        keep_recent: int = 0,
        force_offload: bool = False,
    ):
        """
        Aggressively offload inactive experts to CPU to save GPU memory, with LRU caching.

        Args:
            active_expert_type: Type of the currently active expert
            target_device: Device where the active expert should reside
            keep_recent: Number of recently used experts to keep in GPU memory
            force_offload: Force offloading even recently used experts
        """
        # Determine target device info for logging
        device_desc = "CPU" if target_device is None else str(target_device)
        logger.info(
            f"Offloading inactive experts (keeping {active_expert_type} on {device_desc})"
        )

        # Get experts to keep in GPU memory based on LRU caching
        experts_to_keep = set([active_expert_type])

        if not force_offload and keep_recent > 0:
            # Add the most recently used experts up to the keep_recent limit
            for expert_type in reversed(list(self.expert_usage.keys())):
                if (
                    expert_type != active_expert_type
                    and len(experts_to_keep) < keep_recent + 1
                ):
                    experts_to_keep.add(expert_type)

            # Also include any prefetch candidates if we have room
            for expert_type in list(self.prefetch_candidates):
                if len(experts_to_keep) < self.max_gpu_experts:
                    experts_to_keep.add(expert_type)

        logger.debug(f"Keeping experts in GPU: {experts_to_keep}")

        # First phase: identify experts to offload
        experts_to_offload = {}
        for expert_type, adapter_model in self.expert_adapters.items():
            if expert_type not in experts_to_keep:
                try:
                    # Check current device
                    current_device = next(adapter_model.parameters()).device

                    # Only offload if currently on GPU
                    if current_device.type == "cuda":
                        experts_to_offload[expert_type] = adapter_model
                except Exception as e:
                    logger.error(f"Error checking device for {expert_type}: {str(e)}")

        # Second phase: Aggressively offload the experts in batch
        if experts_to_offload:
            logger.info(f"Offloading {len(experts_to_offload)} experts to CPU")

            # First measure memory usage before offloading
            before_mem = 0.0
            if torch.cuda.is_available():
                before_mem = torch.cuda.memory_allocated() / (1024**3)

            for expert_type, adapter_model in experts_to_offload.items():
                try:
                    # Remember current memory usage to estimate expert size
                    before_expert = 0.0
                    if (
                        expert_type not in self.expert_memory_stats
                        and torch.cuda.is_available()
                    ):
                        before_expert = torch.cuda.memory_allocated() / (1024**3)

                    # Move to CPU
                    adapter_model.to("cpu")
                    logger.debug(f"Expert {expert_type} moved to CPU")

                    # Record memory usage for this expert
                    if (
                        expert_type not in self.expert_memory_stats
                        and torch.cuda.is_available()
                        and before_expert > 0.0
                    ):
                        after_expert = torch.cuda.memory_allocated() / (1024**3)
                        expert_size = before_expert - after_expert
                        if expert_size > 0:
                            self.expert_memory_stats[expert_type] = expert_size
                            logger.info(
                                f"Estimated memory for {expert_type}: {expert_size:.2f} GB"
                            )
                except Exception as e:
                    logger.error(f"Error moving expert {expert_type} to CPU: {str(e)}")

            # Measure memory saved
            if torch.cuda.is_available() and before_mem > 0.0:
                after_mem = torch.cuda.memory_allocated() / (1024**3)
                saved = before_mem - after_mem
                logger.info(f"Offloading saved {saved:.2f} GB of GPU memory")

            # Force aggressive garbage collection to free GPU memory
            gc.collect()
            torch.cuda.empty_cache()

            # Clear prefetch candidates that were not kept
            self.prefetch_candidates = set(
                [c for c in self.prefetch_candidates if c in experts_to_keep]
            )

    def _verify_device_consistency(self, model, target_device):
        """
        Verify that all model parameters are on the expected device.

        Args:
            model: The model to check
            target_device: The expected device for all parameters
        """
        try:
            inconsistent_modules = []
            for name, module in model.named_modules():
                if list(module.parameters()):  # Only check modules with parameters
                    param_device = next(module.parameters()).device
                    if param_device != target_device:
                        inconsistent_modules.append((name, str(param_device)))

            if inconsistent_modules:
                logger.warning(
                    f"Found {len(inconsistent_modules)} modules on unexpected devices:"
                )
                for name, device in inconsistent_modules[:5]:  # Show first 5 only
                    logger.warning(f"  {name}: {device} (expected {target_device})")
                if len(inconsistent_modules) > 5:
                    logger.warning(f"  ...and {len(inconsistent_modules) - 5} more")

                # Try to fix critical modules (like embedding)
                critical_modules = ["embed_tokens", "lm_head"]
                for module_name in critical_modules:
                    for name, module in model.named_modules():
                        if module_name in name and list(module.parameters()):
                            param_device = next(module.parameters()).device
                            if param_device != target_device:
                                logger.info(
                                    f"Moving critical module {name} to {target_device}"
                                )
                                module.to(target_device)
        except Exception as e:
            logger.error(f"Error during device consistency check: {str(e)}")
