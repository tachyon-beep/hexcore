"""
MixedPrecisionTrainer for memory-efficient training of expert adapters.

This module provides support for automatic mixed precision (AMP) training to
optimize memory usage while maintaining training stability.
"""

import torch
import math
from contextlib import nullcontext
from typing import Dict, List, Optional, Any


class MixedPrecisionTrainer:
    """
    Provides safe mixed precision training with automatic fallback mechanisms.

    Key features:
    - Dynamic loss scaling to prevent underflow
    - Fallback to full precision for unstable operations
    - Training stability metrics
    """

    def __init__(self, use_amp=True, scale_factor=2**16, growth_interval=2000):
        """
        Initialize mixed precision training support.

        Args:
            use_amp: Whether to use automatic mixed precision
            scale_factor: Initial scale factor for gradients
            growth_interval: Steps between scale factor growth attempts
        """
        self.use_amp = use_amp and torch.cuda.is_available()
        self.steps = 0
        self.grad_norm_history = []
        self.fp32_operations = set()

        # Create scaler for AMP
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=scale_factor,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=growth_interval,
            enabled=self.use_amp,
        )

    def backward(self, loss):
        """Scale loss and perform backward pass with appropriate precision."""
        if self.use_amp:
            # Scale loss to prevent underflow
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer):
        """Update weights with gradient scaling if using AMP."""
        if self.use_amp:
            # Unscale gradients for clipping
            self.scaler.unscale_(optimizer)

            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]["params"], max_norm=1.0
            )

            # Record norm for stability monitoring
            if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                self.grad_norm_history.append(grad_norm.item())

            # Update weights with scaled gradients
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard update
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]["params"], max_norm=1.0
            )
            if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                self.grad_norm_history.append(grad_norm.item())
            optimizer.step()

        self.steps += 1

    def register_fp32_operation(self, operation_name):
        """Register an operation that requires full precision."""
        self.fp32_operations.add(operation_name)

    def get_ctx_manager(self):
        """Get appropriate context manager for forward pass."""
        if self.use_amp:
            return torch.cuda.amp.autocast()
        else:
            return nullcontext()

    def get_statistics(self):
        """Get training stability statistics."""
        if not self.grad_norm_history:
            return {"status": "No training steps recorded"}

        recent_history = (
            self.grad_norm_history[-100:]
            if len(self.grad_norm_history) > 100
            else self.grad_norm_history
        )

        return {
            "amp_enabled": self.use_amp,
            "steps": self.steps,
            "current_scale": self.scaler.get_scale() if self.use_amp else 1.0,
            "fp32_operations": list(self.fp32_operations),
            "mean_grad_norm": sum(recent_history) / len(recent_history),
            "max_grad_norm": max(recent_history),
            "min_grad_norm": min(recent_history),
            "training_stability": self._assess_stability(),
        }

    def _assess_stability(self):
        """Assess training stability based on gradient norms."""
        if len(self.grad_norm_history) < 100:
            return "Insufficient data"

        # Check for NaNs
        if any(math.isnan(x) for x in self.grad_norm_history[-100:]):
            return "Unstable - NaN values detected"

        # Check for explosion
        if max(self.grad_norm_history[-20:]) > 100:
            return "Potentially unstable - high gradient norms"

        # Check for vanishing
        if max(self.grad_norm_history[-20:]) < 1e-6:
            return "Potentially unstable - vanishing gradients"

        return "Stable"
