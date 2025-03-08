# src/models/cross_expert.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional


class CrossExpertAttention(nn.Module):
    """
    Implements attention mechanism for combining outputs from multiple experts.
    """

    def __init__(self, hidden_size=4096, num_heads=16, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Projection layers for cross-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scaling = self.head_dim**-0.5

    def forward(self, expert_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Apply cross-expert attention to combine outputs from multiple experts.

        Args:
            expert_outputs: List of tensors [batch_size, seq_len, hidden_size] from each expert

        Returns:
            Combined output tensor after cross-expert attention
        """
        # If only one expert, return directly
        if len(expert_outputs) == 1:
            return expert_outputs[0]

        batch_size, seq_len, _ = expert_outputs[0].shape
        num_experts = len(expert_outputs)

        # Stack expert outputs along a new dimension
        # Shape: [batch_size, num_experts, seq_len, hidden_size]
        stacked_outputs = torch.stack(expert_outputs, dim=1)

        # Project to queries, keys, and values
        # Shape: [batch_size, num_experts, seq_len, hidden_size]
        queries = self.q_proj(stacked_outputs)
        keys = self.k_proj(stacked_outputs)
        values = self.v_proj(stacked_outputs)

        # Reshape for multi-head attention
        # Shape: [batch_size, num_experts, seq_len, num_heads, head_dim]
        queries = queries.view(
            batch_size, num_experts, seq_len, self.num_heads, self.head_dim
        )
        keys = keys.view(
            batch_size, num_experts, seq_len, self.num_heads, self.head_dim
        )
        values = values.view(
            batch_size, num_experts, seq_len, self.num_heads, self.head_dim
        )

        # Transpose to [batch_size, seq_len, num_experts, num_heads, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Reshape for attention computation
        # Shape: [batch_size * seq_len, num_experts, num_heads, head_dim]
        queries = queries.reshape(
            batch_size * seq_len, num_experts, self.num_heads, self.head_dim
        )
        keys = keys.reshape(
            batch_size * seq_len, num_experts, self.num_heads, self.head_dim
        )
        values = values.reshape(
            batch_size * seq_len, num_experts, self.num_heads, self.head_dim
        )

        # Transpose queries and keys for attention calculation
        # Shape: [batch_size * seq_len, num_heads, num_experts, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2).transpose(2, 3)  # Prepare for matmul
        values = values.transpose(1, 2)

        # Compute attention scores
        # Shape: [batch_size * seq_len, num_heads, num_experts, num_experts]
        scores = torch.matmul(queries, keys) * self.scaling

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        # Shape: [batch_size * seq_len, num_heads, num_experts, head_dim]
        attn_output = torch.matmul(attn_weights, values)

        # Reshape back
        # Shape: [batch_size, seq_len, num_heads, num_experts, head_dim]
        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads, num_experts, self.head_dim
        )

        # Average over expert dimension
        # Shape: [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.mean(dim=3)

        # Final reshape to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Final projection
        output = self.o_proj(attn_output)

        return output
