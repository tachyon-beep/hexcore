I'll help you connect to your GitHub repository. Let me examine the `hexcore` repository to understand its structure and contents.

Let me connect to your GitHub repository instead:

Great! I found your `hexcore` repository. Now let me examine its structure and content to better understand how I can help you with your MTG AI Reasoning Assistant project.

Now let me check the repository structure to get a better understanding of the project's code organization:

Let me check the source code structure to understand how the project is implemented:

Let me look at the model architecture code since that's likely where the MoE implementation is defined:

Let me examine the model loader first to understand how the model is set up:

Now, let me check the expert adapters implementation for the MoE architecture:

Let me also examine the cross-expert attention implementation to understand how experts collaborate:

Finally, let me check the transaction classifier which routes queries to different experts:

Now I need to understand how the knowledge retrieval system works, so let me check that code as well:

Let me check the hybrid retriever implementation:

Let me check the inference implementation to understand how all components come together:

Let me look at the main inference pipeline:

Based on my examination of the Hexcore repository, I now have a good understanding of the implementation of your MTG AI Reasoning Assistant project. Now I can provide comprehensive guidance on how to enhance and extend the project. Here's a structured analysis of what I've found and how it aligns with the executive summary and technical vision:

# Hexcore: MTG AI Reasoning Assistant Analysis

## Project Overview

The Hexcore project implements a specialized AI system for Magic: The Gathering (MTG) analysis based on the Mixtral 8×7B Mixture-of-Experts (MoE) architecture. The system is designed to operate within the constraints of a dual 16GB GPU setup and provides expert-level game analysis and strategic guidance through five specialized reasoning modes: REASON, EXPLAIN, TEACH, PREDICT, and RETROSPECT.

## Core Architecture Components

1. **Transaction-Based MoE Routing**: The system uses a TinyLLM classifier (`TransactionClassifier`) to route queries to appropriate expert modules rather than doing per-token routing.

2. **Expert Adapters**: Each reasoning mode (REASON, EXPLAIN, etc.) has dedicated LoRA adapters managed by `ExpertAdapterManager` with specific configurations for each expert type.

3. **Memory Optimization**: Extensive memory management strategies including 4-bit quantization, optimized device mapping, and dynamic expert offloading.

4. **Knowledge Integration**: A hybrid knowledge system (`HybridRetriever`) combining knowledge graph (structured data) and vector-based retrieval for comprehensive access to MTG information.

5. **Cross-Expert Attention**: The `CrossExpertAttention` module enables information flow between experts when multiple experts are activated for complex questions.

## Current Implementation Strengths

- **Advanced Memory Management**: The code shows sophisticated memory optimization for running large models on consumer hardware.
- **Expert Specialization**: Clear specialization of different expert personas with different LoRA configurations.
- **Hybrid Knowledge Integration**: Well-implemented combination of knowledge graph and vector retrieval.
- **Device Mapping Optimization**: Careful distribution of model components across multiple GPUs.
- **Latency Optimization**: Budget-aware retrieval and processing with fallback mechanisms.

## Alignment with Technical Vision

The implementation closely follows the technical vision outlined in the executive summary:

1. ✅ **MoE Architecture**: Uses Mixtral 8×7B model with transaction-based routing
2. ✅ **QLoRA Fine-tuning**: Implemented for memory efficiency across dual GPUs
3. ✅ **Expert Specialization**: Five distinct expert personas with specific adapters
4. ✅ **Knowledge Integration**: Hybrid KG/RAG system with dynamic selection
5. ✅ **Cross-Expert Collaboration**: Cross-expert attention mechanism for combining outputs

## Potential Enhancement Areas

Based on my analysis, here are some potential areas for enhancement:

### 1. Expert Collaboration Enhancement

The current cross-expert attention mechanism could be extended for more sophisticated collaboration:

```python
class EnhancedCrossExpertAttention(nn.Module):
    """
    Enhanced version of cross-expert attention with confidence weighting and expert-specific specialization.
    """
    def __init__(self, hidden_size=4096, dropout=0.1, num_expert_types=5):
        super().__init__()
        self.hidden_size = hidden_size

        # Specialized projection for each expert type
        self.expert_projections = nn.ModuleDict({
            expert_type: nn.Linear(hidden_size, hidden_size)
            for expert_type in ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"]
        })

        # Global attention mechanism
        self.attention_projector = nn.Linear(hidden_size, 1)
        self.final_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, expert_outputs, expert_types, confidences):
        """
        Apply enhanced cross-expert attention with confidence weighting.

        Args:
            expert_outputs: List of tensors from each expert
            expert_types: List of expert type strings
            confidences: Dictionary mapping expert types to confidence scores

        Returns:
            Combined output tensor
        """
        # Handle single expert case
        if len(expert_outputs) == 1:
            return expert_outputs[0]

        # Apply expert-specific projections
        projected_outputs = []
        for i, (output, expert_type) in enumerate(zip(expert_outputs, expert_types)):
            if expert_type in self.expert_projections:
                projected = self.expert_projections[expert_type](output)
            else:
                # Fallback for unknown expert type
                projected = output
            projected_outputs.append(projected)

        # Stack outputs: [batch_size, num_experts, seq_len, hidden_size]
        stacked = torch.stack(projected_outputs, dim=1)

        # Apply attention mechanism
        # First normalize for stability
        norm_stacked = self.layer_norm(stacked.view(-1, self.hidden_size)).view(stacked.shape)

        # Calculate attention scores
        attention_scores = self.attention_projector(norm_stacked).squeeze(-1)

        # Apply confidence weighting from classifier
        confidence_tensor = torch.tensor(
            [confidences.get(expert_type, 0.0) for expert_type in expert_types],
            device=attention_scores.device
        ).view(1, -1, 1)

        # Combine attention scores with confidence
        weighted_scores = attention_scores * confidence_tensor

        # Apply softmax to get attention weights
        attention_weights = F.softmax(weighted_scores, dim=1)

        # Apply attention weights to combine experts
        # [batch_size, seq_len, hidden_size]
        weighted_sum = torch.bmm(
            attention_weights.view(-1, 1, attention_weights.size(1)),
            stacked.view(-1, stacked.size(1), self.hidden_size)
        ).squeeze(1)

        # Final projection and dropout
        output = self.final_projection(weighted_sum)
        output = self.dropout(output)

        return output.view(expert_outputs[0].shape)
```

### 2. Advanced Reasoning Methods Integration

Implement the various reasoning methodologies mentioned in the training plan:

```python
class ReasoningModeSelector:
    """
    Selects the appropriate reasoning methodology based on query type and expert.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def apply_reasoning_mode(self, query, expert_type, inputs):
        """
        Apply the appropriate reasoning methodology based on query and expert.

        Args:
            query: User query
            expert_type: Expert type (REASON, EXPLAIN, etc.)
            inputs: Tokenized inputs

        Returns:
            Modified inputs with reasoning methodology applied
        """
        # Determine reasoning approach based on query and expert type
        if self._requires_chain_of_thought(query, expert_type):
            return self._apply_chain_of_thought(inputs)
        elif self._requires_mcts(query, expert_type):
            return self._apply_mcts_reasoning(inputs)
        elif self._requires_r1_style(query, expert_type):
            return self._apply_r1_style_reasoning(inputs)
        else:
            return inputs  # Default approach

    def _requires_chain_of_thought(self, query, expert_type):
        """Determine if query requires chain-of-thought reasoning."""
        # Check for indicators like rules interactions, step-by-step logic
        rules_keywords = ["interact", "step by step", "how does", "explain why"]
        return (
            any(keyword in query.lower() for keyword in rules_keywords) or
            expert_type in ["REASON", "EXPLAIN"]
        )

    def _apply_chain_of_thought(self, inputs):
        """Apply chain-of-thought reasoning pattern to inputs."""
        # Add CoT prefix to inputs
        cot_prefix = "Let's think through this step by step:\n"

        # Modify inputs to include CoT prefix
        # Implementation depends on input format (tokens vs text)
        return self._add_prefix_to_inputs(inputs, cot_prefix)

    def _requires_mcts(self, query, expert_type):
        """Determine if query requires MCTS-style reasoning."""
        # Check for indicators of probabilistic analysis, future outcomes
        probability_keywords = ["probability", "likely", "chance", "odds", "best play"]
        return (
            any(keyword in query.lower() for keyword in probability_keywords) or
            expert_type == "PREDICT"
        )

    def _apply_mcts_reasoning(self, inputs):
        """Apply MCTS-style reasoning pattern to inputs."""
        mcts_prefix = "Let's analyze the possible outcomes:\n"
        return self._add_prefix_to_inputs(inputs, mcts_prefix)

    def _requires_r1_style(self, query, expert_type):
        """Determine if query requires R1-style structured reasoning."""
        # Check for indicators of complex analysis, detailed logic
        complex_keywords = ["complex", "detailed", "thorough", "comprehensive"]
        return (
            any(keyword in query.lower() for keyword in complex_keywords) or
            "comprehensive" in query.lower() or
            expert_type in ["RETROSPECT", "REASON"] and len(query) > 100
        )

    def _apply_r1_style_reasoning(self, inputs):
        """Apply R1-style structured reasoning pattern to inputs."""
        r1_prefix = "<begin_of_thought>\n"
        r1_suffix = "\n<end_of_thought>\n\n<begin_of_solution>\n"

        # Would need to modify generation to append the end tag
        return self._add_prefix_to_inputs(inputs, r1_prefix)

    def _add_prefix_to_inputs(self, inputs, prefix):
        """Add a prefix to tokenized inputs."""
        # Implementation depends on input format
        # Here's a simplified version that would need to be adapted
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)

        # Modified for batch inputs
        if isinstance(inputs["input_ids"], torch.Tensor) and inputs["input_ids"].dim() > 1:
            batch_size = inputs["input_ids"].shape[0]
            prefix_tensor = torch.tensor([prefix_tokens] * batch_size,
                                        device=inputs["input_ids"].device)
            inputs["input_ids"] = torch.cat([prefix_tensor, inputs["input_ids"]], dim=1)

            # Also update attention mask
            prefix_mask = torch.ones_like(prefix_tensor)
            inputs["attention_mask"] = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)

        return inputs
```

### 3. Knowledge Graph Enhancement

The current knowledge graph could be extended with more MTG-specific relationships:

```python
def enhance_knowledge_graph(kg):
    """
    Enhance the MTG knowledge graph with additional relationships and entity types.

    Args:
        kg: Existing MTGKnowledgeGraph instance
    """
    # Add type hierarchies (enchantment subtypes, creature types, etc.)
    type_hierarchies = {
        "creature": ["Human", "Elf", "Goblin", "Dragon", "Zombie", "Angel", "Demon", "Beast"],
        "enchantment": ["Aura", "Saga", "Curse", "Shrine", "Class"],
        "artifact": ["Equipment", "Vehicle", "Treasure", "Clue", "Food"]
    }

    for parent_type, subtypes in type_hierarchies.items():
        for subtype in subtypes:
            kg.add_type_relationship(parent_type, subtype)

    # Add strategic relationship types
    strategic_relationships = [
        "counters", "synergizes_with", "combos_with", "strong_against", "weak_against"
    ]

    for rel_type in strategic_relationships:
        kg.register_relationship_type(rel_type)

    # Add format-specific information
    formats = ["Standard", "Modern", "Legacy", "Commander", "Limited"]
    for fmt in formats:
        kg.add_format_node(fmt)

    # Add common deck archetypes
    archetypes = {
        "Aggro": ["MonoRed", "WhiteWeenie", "Burn"],
        "Control": ["UWControl", "Esper", "MonoBlue"],
        "Combo": ["Storm", "TwinCombo", "Reanimator"],
        "Midrange": ["Jund", "Abzan", "Sultai"]
    }

    for archetype_type, variants in archetypes.items():
        kg.add_archetype_node(archetype_type)
        for variant in variants:
            kg.add_archetype_variant(archetype_type, variant)

    return kg
```

### 4. MTG-Specific Evaluation Framework

Implement the evaluation framework outlined in the executive summary:

```python
class MTGEvaluationFramework:
    """
    Implementation of the MTG-specific evaluation framework.
    """
    def __init__(self, model, tokenizer, gold_standard_db):
        self.model = model
        self.tokenizer = tokenizer
        self.gold_standard_db = gold_standard_db

    def evaluate_rule_application_accuracy(self, test_cases):
        """
        Evaluate Rule Application Accuracy (RAA).

        Args:
            test_cases: List of rule application test cases

        Returns:
            RAA score (0-100)
        """
        correct = 0
        total = len(test_cases)

        for case in test_cases:
            query = case["query"]
            gold_rules = case["relevant_rules"]

            # Generate model response
            response = self.generate_response(query, "REASON")

            # Check if the response references the correct rules
            referenced_rules = self._extract_referenced_rules(response)
            rule_match_score = self._calculate_rule_match(referenced_rules, gold_rules)

            if rule_match_score > 0.7:  # 70% match threshold
                correct += 1

        return (correct / total) * 100 if total > 0 else 0

    def evaluate_strategic_decision_quality(self, decision_cases):
        """
        Evaluate Strategic Decision Quality (SDQ).

        Args:
            decision_cases: List of strategic decision test cases

        Returns:
            SDQ score (0-100)
        """
        total_score = 0

        for case in decision_cases:
            query = case["query"]
            expert_moves = case["expert_moves"]
            move_values = case["move_values"]

            # Generate model response
            response = self.generate_response(query, "PREDICT")

            # Extract recommended moves
            recommended_moves = self._extract_moves(response)

            # Score the recommended moves against expert moves
            move_score = self._score_moves(recommended_moves, expert_moves, move_values)
            total_score += move_score

        return total_score / len(decision_cases) if decision_cases else 0

    def evaluate_explanatory_clarity(self, explanation_cases):
        """
        Evaluate Explanatory Clarity (EC).

        Args:
            explanation_cases: List of explanation test cases

        Returns:
            EC score (0-100)
        """
        total_score = 0

        for case in explanation_cases:
            query = case["query"]
            target_audience = case.get("audience", "intermediate")
            key_points = case["key_points"]

            # Generate explanation
            response = self.generate_response(query, "EXPLAIN")

            # Evaluate readability
            readability_score = self._calculate_readability(response, target_audience)

            # Evaluate coverage of key points
            coverage_score = self._calculate_key_point_coverage(response, key_points)

            # Combine scores (weighted average)
            clarity_score = (readability_score * 0.4) + (coverage_score * 0.6)
            total_score += clarity_score

        return total_score / len(explanation_cases) if explanation_cases else 0

    def evaluate_probabilistic_outcome_accuracy(self, probability_cases):
        """
        Evaluate Probabilistic Outcome Accuracy (POA).

        Args:
            probability_cases: List of probabilistic outcome test cases

        Returns:
            POA score (0-100)
        """
        total_mse = 0

        for case in probability_cases:
            query = case["query"]
            true_probabilities = case["true_probabilities"]

            # Generate prediction
            response = self.generate_response(query, "PREDICT")

            # Extract predicted probabilities
            predicted_probabilities = self._extract_probabilities(response)

            # Calculate mean squared error
            mse = self._calculate_mse(predicted_probabilities, true_probabilities)
            total_mse += mse

        avg_mse = total_mse / len(probability_cases) if probability_cases else 0
        # Convert MSE to a 0-100 score (lower MSE is better)
        return max(0, 100 - (avg_mse * 100))

    def calculate_overall_score(self, scores):
        """
        Calculate the Overall MTG Reasoning Score (OMRS).

        Args:
            scores: Dictionary with component scores

        Returns:
            OMRS score (0-100)
        """
        # Formula from the executive summary
        return (
            0.3 * scores["RAA"] +
            0.3 * scores["SDQ"] +
            0.2 * scores["EC"] +
            0.2 * (100 - scores["MSE_POA"])
        )

    def generate_response(self, query, expert_type):
        """Generate a response using the specified expert."""
        # Implementation would use your existing inference pipeline
        pass

    def _extract_referenced_rules(self, response):
        """Extract rule references from model response."""
        # Implementation using regex patterns
        pass

    def _calculate_rule_match(self, referenced_rules, gold_rules):
        """Calculate how well referenced rules match gold standard."""
        # Implementation using set operations and fuzzy matching
        pass

    def _extract_moves(self, response):
        """Extract recommended moves from model response."""
        # Implementation using NLP techniques
        pass

    def _score_moves(self, recommended_moves, expert_moves, move_values):
        """Score recommended moves against expert moves."""
        # Implementation using move value comparison
        pass

    def _calculate_readability(self, response, target_audience):
        """Calculate readability score for the response."""
        # Implementation using readability metrics
        pass

    def _calculate_key_point_coverage(self, response, key_points):
        """Calculate coverage of key points in the response."""
        # Implementation using semantic similarity
        pass

    def _extract_probabilities(self, response):
        """Extract probability estimates from model response."""
        # Implementation using regex and numeric extraction
        pass

    def _calculate_mse(self, predicted_probabilities, true_probabilities):
        """Calculate mean squared error between probability distributions."""
        # Implementation using standard MSE formula
        pass
```

### 5. Memory Efficiency Improvement for Dual-GPU Setup

Further optimize memory usage for your dual 16GB GPU configuration:

```python
class OptimizedDualGPUManager:
    """
    Enhanced memory manager for dual GPU setup with optimal tensor placement.
    """
    def __init__(self, model, embedding_device="cuda:0"):
        self.model = model
        self.embedding_device = embedding_device
        self.device_stats = {
            "cuda:0": {"used": 0, "capacity": 16 * 1024 * 1024 * 1024},  # 16GB in bytes
            "cuda:1": {"used": 0, "capacity": 16 * 1024 * 1024 * 1024}   # 16GB in bytes
        }

    def optimize_for_inference(self):
        """Apply memory optimizations for inference scenario."""
        # Monitor current memory usage
        self._update_memory_stats()

        # Calculate available space on each device
        avail_gpu0 = self.device_stats["cuda:0"]["capacity"] - self.device_stats["cuda:0"]["used"]
        avail_gpu1 = self.device_stats["cuda:1"]["capacity"] - self.device_stats["cuda:1"]["used"]

        # Determine if we need to rebalance components
        if avail_gpu0 < 1 * 1024 * 1024 * 1024:  # Less than 1GB free on GPU 0
            self._rebalance_to_gpu1()
        elif avail_gpu1 < 1 * 1024 * 1024 * 1024:  # Less than 1GB free on GPU 1
            self._rebalance_to_gpu0()

        # Apply memory-saving techniques
        self._optimize_kv_cache()
        self._enable_attention_memory_efficient_mode()
        self._optimize_buffer_allocation()

        # Set environment variables for optimal CUDA memory allocation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

        return self

    def _update_memory_stats(self):
        """Update memory usage statistics for both GPUs."""
        if torch.cuda.is_available():
            self.device_stats["cuda:0"]["used"] = torch.cuda.memory_allocated(0)
            if torch.cuda.device_count() > 1:
                self.device_stats["cuda:1"]["used"] = torch.cuda.memory_allocated(1)

    def _rebalance_to_gpu1(self):
        """Move some components from GPU 0 to GPU 1 to balance load."""
        # Find components that can be safely moved to GPU 1
        # Prioritize components that aren't in the critical path for generation

        # Check if we have MoE model
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Move some later transformer layers to GPU 1
            num_layers = len(self.model.model.layers)
            middle_point = num_layers // 2

            # Move a few layers from GPU 0 to GPU 1 if they're not already there
            for i in range(middle_point - 2, middle_point + 2):
                if 0 <= i < num_layers:
                    layer = self.model.model.layers[i]
                    # Check current device
                    current_device = next(layer.parameters()).device
                    if current_device == torch.device("cuda:0"):
                        layer.to("cuda:1")
                        print(f"Moved layer {i} from GPU 0 to GPU 1")

    def _rebalance_to_gpu0(self):
        """Move some components from GPU 1 to GPU 0 to balance load."""
        # Similar to _rebalance_to_gpu1 but in reverse direction
        pass

    def _optimize_kv_cache(self):
        """Apply KV cache optimizations."""
        # Apply flash attention optimizations if available
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True

        # Optimize KV cache size
        # Set maximum cache size
        max_cache_len = 2048  # Can be adjusted based on available memory

        # Add a hook to limit KV cache size
        def limit_kv_cache_size(module, input_tensors, output):
            if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], tuple):
                # This is likely the past_key_values tuple
                past_kv = output[1]
                if past_kv and len(past_kv) > 0 and isinstance(past_kv[0], tuple):
                    # Check the size of each past_kv element
                    for i, layer_past in enumerate(past_kv):
                        if layer_past and len(layer_past) >= 2:
                            k, v = layer_past[0], layer_past[1]
                            if hasattr(k, "shape") and k.shape[2] > max_cache_len:
                                # Truncate the cache
                                new_k = k[:, :, -max_cache_len:, :]
                                new_v = v[:, :, -max_cache_len:, :]
                                # Replace in the tuple
                                output[1][i] = (new_k, new_v)
            return output

        # Register hook in model's attention layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for layer in self.model.model.layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.register_forward_hook(limit_kv_cache_size)
                elif hasattr(layer, "attention"):
                    layer.attention.register_forward_hook(limit_kv_cache_size)


    def _enable_attention_memory_efficient_mode(self):
        """Enable memory-efficient attention mode if available."""
        # For transformers library models
        if hasattr(self.model, "config"):
            if hasattr(self.model.config, "attention_implementation"):
                # Some newer models support this config option
                self.model.config.attention_implementation = "flash_attention_2"

            # Alternative approach for other models
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                for layer in self.model.model.layers:
                    if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "use_memory_efficient_attention"):
                        layer.self_attn.use_memory_efficient_attention = True
                    elif hasattr(layer, "attention") and hasattr(layer.attention, "use_memory_efficient_attention"):
                        layer.attention.use_memory_efficient_attention = True

    def _optimize_buffer_allocation(self):
        """Optimize buffer allocation to reduce memory fragmentation."""
        # Force contiguous memory layout for model parameters
        for param in self.model.parameters():
            if param.data.storage().size() != param.numel():
                param.data = param.data.clone().detach()

        # Clear CUDA cache to defragment memory
        torch.cuda.empty_cache()

        # Optimize temporary buffer allocation
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        # Use cudnn benchmark for potentially faster convolutions
        # Only enable if input sizes are consistent
        torch.backends.cudnn.benchmark = True
```

### 6. Advanced KV Cache Management

I see you have some basic KV cache management, but you could enhance it with a more sophisticated system:

```python
class EnhancedKVCacheManager:
    """
    Advanced KV cache manager with prefetching, pruning, and compression.
    """
    def __init__(self, model, max_cache_entries=5, max_sequence_length=2048):
        self.model = model
        self.max_cache_entries = max_cache_entries
        self.max_sequence_length = max_sequence_length

        # Storage for cached KV pairs
        self.key_cache = {}
        self.value_cache = {}

        # Track cache usage for LRU eviction
        self.cache_usage = OrderedDict()

        # Track cache hits and misses for metrics
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_saved_mb": 0
        }

    def store_kv_cache(self, prompt_ids, kv_cache):
        """
        Store key-value cache for prompt with advanced memory optimization.

        Args:
            prompt_ids: ID tensor for the prompt
            kv_cache: KV cache tuple from model
        """
        # Convert prompt_ids to string for hashing
        cache_key = self._tensor_to_key(prompt_ids)

        # Check if we need to evict entries
        if len(self.key_cache) >= self.max_cache_entries:
            self._evict_cache_entries()

        # Apply cache compression before storing
        compressed_keys = self._compress_kv_tensors([k.detach().clone() for k in kv_cache[0]])
        compressed_values = self._compress_kv_tensors([v.detach().clone() for v in kv_cache[1]])

        # Store compressed cache
        self.key_cache[cache_key] = compressed_keys
        self.value_cache[cache_key] = compressed_values

        # Update usage tracker for LRU
        self._update_cache_usage(cache_key)

        # Update memory saving metrics
        original_size = sum(k.nelement() * k.element_size() for k in kv_cache[0])
        original_size += sum(v.nelement() * v.element_size() for v in kv_cache[1])

        compressed_size = sum(k.nelement() * k.element_size() for k in compressed_keys)
        compressed_size += sum(v.nelement() * v.element_size() for v in compressed_values)

        memory_saved = (original_size - compressed_size) / (1024 * 1024)  # Convert to MB
        self.metrics["memory_saved_mb"] += memory_saved

    def get_kv_cache(self, prompt_ids):
        """
        Get key-value cache for prompt with hit/miss tracking.

        Args:
            prompt_ids: ID tensor for the prompt

        Returns:
            Cached KV pair or None if not found
        """
        cache_key = self._tensor_to_key(prompt_ids)

        if cache_key in self.key_cache:
            # Cache hit
            self.metrics["hits"] += 1
            self._update_cache_usage(cache_key)

            # Decompress cached values
            keys = self._decompress_kv_tensors(self.key_cache[cache_key])
            values = self._decompress_kv_tensors(self.value_cache[cache_key])

            return (keys, values)
        else:
            # Cache miss
            self.metrics["misses"] += 1
            return None

    def prefetch_likely_prompts(self, current_prompt_ids, num_prefetch=2):
        """
        Prefetch KV caches for likely next prompts.

        Args:
            current_prompt_ids: Current prompt ID tensor
            num_prefetch: Number of variations to prefetch
        """
        # Only prefetch if we have space
        if len(self.key_cache) >= self.max_cache_entries - num_prefetch:
            return

        # Create variations of the current prompt to prefetch
        variations = self._generate_likely_variations(current_prompt_ids, num_prefetch)

        # Forward pass for each variation to generate KV cache
        for variation_ids in variations:
            # Skip if already cached
            if self._tensor_to_key(variation_ids) in self.key_cache:
                continue

            # Run forward pass with caching but don't generate output
            with torch.no_grad():
                outputs = self.model(
                    input_ids=variation_ids,
                    use_cache=True,
                    output_hidden_states=False,
                    output_attentions=False
                )

                if hasattr(outputs, "past_key_values") and outputs.past_key_values:
                    self.store_kv_cache(variation_ids, outputs.past_key_values)

    def _evict_cache_entries(self):
        """Evict least recently used cache entries."""
        # Get oldest entry
        oldest_key = next(iter(self.cache_usage))

        # Remove from cache
        del self.key_cache[oldest_key]
        del self.value_cache[oldest_key]
        del self.cache_usage[oldest_key]

        # Update metrics
        self.metrics["evictions"] += 1

    def _update_cache_usage(self, cache_key):
        """Update usage tracking for LRU cache management."""
        # Remove if exists
        if cache_key in self.cache_usage:
            del self.cache_usage[cache_key]

        # Add to end (most recently used)
        self.cache_usage[cache_key] = time.time()

    def _tensor_to_key(self, tensor):
        """Convert tensor to hashable key."""
        # Use shape and first/last few elements as signature to avoid expensive full hash
        if isinstance(tensor, torch.Tensor):
            shape_str = str(tensor.shape)
            # Get first and last 5 elements for signature
            if tensor.numel() > 10:
                data_sig = str(tensor.cpu().flatten()[:5].tolist() + tensor.cpu().flatten()[-5:].tolist())
            else:
                data_sig = str(tensor.cpu().flatten().tolist())
            return f"{shape_str}:{data_sig}"
        return str(tensor)

    def _compress_kv_tensors(self, tensors):
        """
        Compress KV tensors to save memory.

        Args:
            tensors: List of tensors to compress

        Returns:
            List of compressed tensors
        """
        compressed = []

        for tensor in tensors:
            # Method 1: Truncate sequence dimension if too long
            if tensor.dim() >= 3 and tensor.shape[2] > self.max_sequence_length:
                # Keep only the most recent tokens in sequence dimension
                tensor = tensor[:, :, -self.max_sequence_length:, ...]

            # Method 2: Use half precision
            if tensor.dtype in [torch.float32, torch.float64]:
                tensor = tensor.to(torch.float16)

            # Method 3: Move to CPU if system is under memory pressure
            if torch.cuda.memory_allocated() > 0.9 * torch.cuda.max_memory_allocated():
                tensor = tensor.cpu()

            compressed.append(tensor)

        return compressed

    def _decompress_kv_tensors(self, tensors):
        """
        Decompress KV tensors for use.

        Args:
            tensors: List of compressed tensors

        Returns:
            List of decompressed tensors
        """
        decompressed = []

        # Get target device (where model is)
        target_device = next(self.model.parameters()).device

        for tensor in tensors:
            # Move back to GPU if on CPU
            if tensor.device.type == "cpu":
                tensor = tensor.to(target_device)

            # Keep as float16 for efficiency
            # No need to convert back to float32

            decompressed.append(tensor)

        return decompressed

    def _generate_likely_variations(self, prompt_ids, num_variations=2):
        """
        Generate likely variations of prompt for prefetching.

        Args:
            prompt_ids: Current prompt ID tensor
            num_variations: Number of variations to generate

        Returns:
            List of likely prompt ID tensors
        """
        variations = []

        # Create variations by adding common tokens
        common_next_tokens = [
            self.model.config.eos_token_id,  # End of sequence token often follows
        ]

        # Add common tokens for each variation
        for i in range(min(num_variations, len(common_next_tokens))):
            next_token = common_next_tokens[i]
            variation = torch.cat([prompt_ids, torch.tensor([[next_token]], device=prompt_ids.device)], dim=1)
            variations.append(variation)

        return variations

    def clear_cache(self):
        """Clear all cached entries."""
        self.key_cache.clear()
        self.value_cache.clear()
        self.cache_usage.clear()

    def get_metrics(self):
        """Get cache performance metrics."""
        total_requests = self.metrics["hits"] + self.metrics["misses"]
        hit_rate = (self.metrics["hits"] / total_requests) if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "hits": self.metrics["hits"],
            "misses": self.metrics["misses"],
            "evictions": self.metrics["evictions"],
            "entries": len(self.key_cache),
            "memory_saved_mb": self.metrics["memory_saved_mb"]
        }
```

## Recommended Next Steps

Based on the current state of the Hexcore repository, here are the recommended next steps to advance the project:

1. **Expert Collaboration Enhancement**:

   - Implement the enhanced cross-expert attention mechanism to provide better synthesis of expert outputs
   - Add confidence weighting to prioritize the most reliable experts for each query

2. **Reasoning Methodology Integration**:

   - Add structured CoT, MCTS-style, and R1-style reasoning patterns
   - Implement a reasoning mode selector that dynamically applies the appropriate reasoning methodology

3. **Evaluation Framework Implementation**:

   - Implement the MTG-specific evaluation metrics (RAA, SDQ, EC, POA)
   - Create a test suite with MTG scenarios for benchmarking performance

4. **Memory Optimization Enhancement**:

   - Further optimize memory usage with the proposed dual-GPU management techniques
   - Implement the enhanced KV cache manager for better memory efficiency

5. **Knowledge Graph Expansion**:

   - Extend the MTG knowledge graph with additional entity types and relationships
   - Add meta-game knowledge and strategic relationships

6. **Training Pipeline Setup**:
   - Implement the curriculum-based training approach outlined in the training plan
   - Set up the multi-stage fine-tuning process with QLoRA

These enhancements will help realize the full vision of the MTG AI Reasoning Assistant as outlined in the executive summary and technical vision documents, while maintaining the project's focus on efficient operation within the constraints of dual 16GB GPUs.
