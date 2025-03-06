# Technical Implementation Guide: MTG AI Reasoning Assistant

## 1. Detailed Memory Allocation & Hardware Requirements

### 1.1 Hardware Requirements

The system is designed for dual 16GB GPUs (32GB VRAM total) with the following baseline specifications:

- 2 × GPUs with 16GB VRAM each (e.g., RTX 3080, RTX 4060 Ti, or similar)
- 32GB+ system RAM
- Minimum 8-core CPU
- 100GB+ SSD storage for model weights, adaptation layers, and knowledge bases

### 1.2 Memory Allocation Strategy

The Mixtral 8×7B model contains approximately 46.7 billion parameters, which would require around 180GB in FP32 precision. Through aggressive 4-bit quantization and tensor parallelism, we distribute these parameters across two GPUs as follows:

#### GPU 0 Memory Allocation

- **Base Model Components (11.75GB)**:
  - 4 experts (Expert 0, 1, 2, 3) quantized to 4-bit: ~6GB
  - Shared transformer layers (half): ~4.5GB
  - Embedding layers: ~1.25GB
- **Router & Classification Components (0.5GB)**:
  - TinyLLM transaction classifier: ~0.3GB
  - Router weights and buffers: ~0.2GB
- **Working Memory (3.25GB)**:
  - KV cache for generation: ~1.5GB
  - Temporary tensors and activations: ~1GB
  - RAG retrieved context embeddings: ~0.5GB
  - Gradients during fine-tuning: ~0.25GB
- **Reserved (0.5GB)**:
  - Memory buffer to prevent OOM errors

#### GPU 1 Memory Allocation

- **Base Model Components (11.75GB)**:
  - 4 experts (Expert 4, 5, 6, 7) quantized to 4-bit: ~6GB
  - Shared transformer layers (half): ~4.5GB
  - Output and normalization layers: ~1.25GB
- **Expert-Specific Components (0.75GB)**:
  - LoRA adapters for experts 4-7: ~0.5GB
  - Expert collaboration attention mechanism: ~0.25GB
- **Working Memory (3GB)**:
  - KV cache for generation: ~1.5GB
  - Temporary tensors and activations: ~1GB
  - Optimizer states during fine-tuning: ~0.5GB
- **Reserved (0.5GB)**:
  - Memory buffer to prevent OOM errors

The memory allocation is visualized below:

```text
GPU 0 (16GB)                              GPU 1 (16GB)
+---------------------------+             +---------------------------+
| Quantized Experts 0-3     |             | Quantized Experts 4-7     |
| (6GB)                     |             | (6GB)                     |
+---------------------------+             +---------------------------+
| Shared Transformer Layers |             | Shared Transformer Layers |
| First Half (4.5GB)        |             | Second Half (4.5GB)       |
+---------------------------+             +---------------------------+
| Embedding Layers (1.25GB) |             | Output Layers (1.25GB)    |
+---------------------------+             +---------------------------+
| TinyLLM Router (0.5GB)    |             | LoRA Adapters (0.75GB)    |
+---------------------------+             +---------------------------+
| KV Cache & Working (3.25GB)|            | KV Cache & Working (3GB)  |
+---------------------------+             +---------------------------+
| Reserved (0.5GB)          |             | Reserved (0.5GB)          |
+---------------------------+             +---------------------------+
```

### 1.3 Expert Distribution

Experts are distributed by reasoning type:

- **GPU 0**:
  - Expert 0: REASON (Primary) - Rules and mechanics analysis
  - Expert 1: REASON (Secondary) - Edge case handling
  - Expert 2: EXPLAIN (Primary) - Clear explanations
  - Expert 3: EXPLAIN (Secondary) - Simplification
- **GPU 1**:
  - Expert 4: TEACH - Educational outputs
  - Expert 5: PREDICT - Gameplay forecasting
  - Expert 6: RETROSPECT - Post-game analysis
  - Expert 7: General/multi-purpose expertise

This allocation ensures that commonly paired experts (REASON+PREDICT or EXPLAIN+TEACH) reside on different GPUs to maximize parallel computation.

## 2. Byte-Level Memory Optimization Strategies

### 2.1 4-bit Quantization Specifics

We implement NF4 (Normal Float 4-bit) quantization for all model weights using the bitsandbytes library:

```python
import torch
from bitsandbytes.nn import Linear4bit
from transformers import AutoModelForCausalLM

def load_quantized_model(model_id="mistralai/Mixtral-8x7B-v0.1"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",  # Automatic device mapping
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # Double quantization for further compression
    )
    return model
```

NF4 quantization provides better precision for weights close to zero compared to int4, which is crucial for maintaining reasoning performance.

### 2.2 Layer Distribution Strategy

Transformer layers are distributed using a customized device map that assigns modules to specific GPUs:

```python
def create_optimized_device_map(model_id="mistralai/Mixtral-8x7B-v0.1"):
    from transformers.utils.device import infer_auto_device_map

    # First load model config without loading weights
    model_config = AutoConfig.from_pretrained(model_id)

    # Create a detailed max memory map
    max_memory = {
        0: "15GB",  # GPU 0: 15GB (reserve 1GB for system)
        1: "15GB",  # GPU 1: 15GB (reserve 1GB for system)
        "cpu": "32GB"  # CPU: Use as overflow if needed
    }

    # Custom device map
    device_map = {
        "model.embed_tokens": 0,
        "model.layers.0": 0,
        "model.layers.1": 0,
        "model.layers.2": 0,
        "model.layers.3": 0,
        "model.layers.4": 0,
        "model.layers.5": 0,
        "model.layers.6": 0,
        "model.layers.7": 0,
        "model.layers.8": 0,
        "model.layers.9": 0,
        "model.layers.10": 0,
        "model.layers.11": 0,
        "model.layers.12": 0,
        "model.layers.13": 0,
        "model.layers.14": 1,
        "model.layers.15": 1,
        "model.layers.16": 1,
        "model.layers.17": 1,
        "model.layers.18": 1,
        "model.layers.19": 1,
        "model.layers.20": 1,
        "model.layers.21": 1,
        "model.layers.22": 1,
        "model.layers.23": 1,
        "model.layers.24": 1,
        "model.layers.25": 1,
        "model.layers.26": 1,
        "model.layers.27": 1,
        "model.norm": 1,
        "lm_head": 1
    }

    # For each layer, distribute MoE experts
    for i in range(28):
        layer_name = f"model.layers.{i}"
        gpu_id = 0 if i < 14 else 1

        # Distribute experts across GPUs
        for j in range(8):
            expert_gpu = 0 if j < 4 else 1
            device_map[f"{layer_name}.block_sparse_moe.experts.{j}"] = expert_gpu

        # Router stays with the layer's GPU
        device_map[f"{layer_name}.block_sparse_moe.gate"] = gpu_id

    return device_map
```

This mapping ensures balanced memory usage and minimizes cross-device communication during forward passes.

### 2.3 Memory Efficient Attention Implementation

To optimize attention computation, we implement FlashAttention:

```python
from flash_attn import flash_attn_func

# Modified attention forward function
def optimized_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    # Regular QKV projection
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape for FlashAttention
    batch_size, q_len, hidden_size = query_states.size()
    head_dim = hidden_size // self.num_heads

    query_states = query_states.view(batch_size, q_len, self.num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len, self.num_heads, head_dim).transpose(1, 2)

    # Causal mask is automatically handled by FlashAttention
    attn_output = flash_attn_func(
        query_states, key_states, value_states,
        causal=True
    )

    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, None
```

FlashAttention reduces memory usage by up to 20% compared to standard attention implementations by avoiding explicit storage of the attention matrix.

### 2.4 Gradient Checkpointing

During fine-tuning, we implement gradient checkpointing to trade computation for memory:

```python
def prepare_model_for_training(model):
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Use contiguous parameters for better memory efficiency
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.contiguous()

    # Explicitly clean up unnecessary buffers
    torch.cuda.empty_cache()

    return model
```

This reduces memory usage during backpropagation by recomputing activations during the backward pass rather than storing them.

### 2.5 CPU Offloading Strategy

For components not needed during every forward pass, we implement strategic CPU offloading:

```python
def strategic_cpu_offloading(model):
    # Identify modules that aren't used in every forward pass
    unused_expert_indices = []  # To be populated during inference based on router

    # Temporarily move unused experts to CPU
    for i in unused_expert_indices:
        for layer_idx in range(model.config.num_hidden_layers):
            expert = model.model.layers[layer_idx].block_sparse_moe.experts[i]
            expert.to("cpu")

    # Move them back when needed
    def move_expert_to_gpu(expert_idx, device="cuda:0"):
        for layer_idx in range(model.config.num_hidden_layers):
            expert = model.model.layers[layer_idx].block_sparse_moe.experts[expert_idx]
            expert.to(device)

    return move_expert_to_gpu
```

This function dynamically offloads unused experts to CPU and provides a mechanism to bring them back to GPU when needed.

## 3. QLoRA Fine-Tuning Configuration

### 3.1 LoRA Adapter Configuration

We use QLoRA (Quantized Low-Rank Adaptation) to efficiently fine-tune the model:

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def configure_qlora(model):
    # Prepare model for QLoRA training
    model = prepare_model_for_kbit_training(model)

    # Expert-specific configurations
    expert_configs = {
        "REASON": LoraConfig(
            r=16,                     # Higher rank for complex reasoning
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "EXPLAIN": LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
            lora_dropout=0.10,        # Higher dropout for better generalization in explanations
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "TEACH": LoraConfig(
            r=8,                      # Lower rank sufficient for teaching patterns
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "PREDICT": LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        ),
        "RETROSPECT": LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    }

    # Apply LoRA adapters for each expert type
    expert_models = {}
    for expert_name, config in expert_configs.items():
        expert_models[expert_name] = get_peft_model(model, config)

    return expert_models
```

Each expert uses custom LoRA hyperparameters tailored to its function, with higher ranks for complex reasoning tasks.

### 3.2 Training Configuration

The following training setup optimizes for memory efficiency while maintaining learning quality:

```python
def create_training_args(expert_name, output_dir="./results"):
    from transformers import TrainingArguments

    # Base configuration
    base_args = {
        "output_dir": f"{output_dir}/{expert_name}",
        "per_device_train_batch_size": 1,      # Small batch size due to memory constraints
        "gradient_accumulation_steps": 16,     # Accumulate gradients to simulate larger batch
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "fp16": True,                          # Use mixed precision
        "do_eval": True,
        "evaluation_strategy": "steps",
        "eval_steps": 200,
        "save_strategy": "steps",
        "save_steps": 200,
        "save_total_limit": 3,                 # Keep only the 3 best checkpoints
        "logging_steps": 10,
        "max_steps": 10000,                    # Adjust based on dataset size
        "max_grad_norm": 0.3,                  # Prevent gradient explosions
        "dataloader_num_workers": 4,
        "report_to": "wandb",                  # Use Weights & Biases for tracking
        "ddp_find_unused_parameters": False,   # Optimize DDP
        "gradient_checkpointing": True,        # Enable gradient checkpointing
        "optim": "paged_adamw_8bit",           # 8-bit optimizer saves memory
    }

    # Expert-specific modifications
    expert_adjustments = {
        "REASON": {"learning_rate": 2e-4, "max_steps": 12000},
        "EXPLAIN": {"learning_rate": 3e-4, "max_steps": 10000},
        "TEACH": {"learning_rate": 3e-4, "max_steps": 8000},
        "PREDICT": {"learning_rate": 2e-4, "max_steps": 10000},
        "RETROSPECT": {"learning_rate": 2.5e-4, "max_steps": 6000}
    }

    # Update base args with expert-specific ones
    base_args.update(expert_adjustments.get(expert_name, {}))

    return TrainingArguments(**base_args)
```

Expert-specific learning rates and training durations optimize each for its particular task.

### 3.3 Memory-Efficient Data Pipeline

Implementing an efficient data loading pipeline reduces memory pressure during training:

```python
def create_efficient_data_pipeline(dataset, tokenizer, max_length=2048):
    from torch.utils.data import DataLoader

    # Tokenize dataset efficiently
    def tokenize_function(examples):
        # Process in batches
        inputs = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        return inputs

    # Apply tokenization with smart batching
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,  # Process 100 examples at once
        num_proc=4,      # Parallelize across 4 CPU cores
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Tokenizing dataset",
    )

    # Create memory-efficient data loader
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=1,  # Keep batch size small
        shuffle=True,
        pin_memory=True,  # Speeds up data transfer to GPU
        drop_last=True,
    )

    return data_loader
```

This implementation uses parallel preprocessing, caching, and pinned memory to optimize the data pipeline.

### 3.4 Distributed Training Setup

For distributed training across both GPUs:

```python
def setup_distributed_training(expert_models, training_args_dict, data_loaders_dict):
    import os
    from transformers import Trainer

    # Set environment variables for distributed training
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"  # Use loopback interface for faster communication

    # Initialize distributed training
    torch.distributed.init_process_group(backend="nccl")

    # Train each expert
    trained_experts = {}
    for expert_name, model in expert_models.items():
        trainer = Trainer(
            model=model,
            args=training_args_dict[expert_name],
            train_dataset=data_loaders_dict[expert_name]["train"],
            eval_dataset=data_loaders_dict[expert_name]["eval"]
        )
        trainer.train()
        # Save model
        model.save_pretrained(f"./models/{expert_name}")
        trained_experts[expert_name] = model

    # Clean up
    torch.distributed.destroy_process_group()

    return trained_experts
```

This configuration allows expert training to be distributed across GPUs while maintaining memory efficiency.

## 4. Expert Disagreement Resolution Algorithm

### 4.1 Weighted Confidence Scoring

When multiple experts are active, we implement a weighted confidence scoring mechanism to resolve disagreements:

```python
def expert_disagreement_resolution(expert_outputs, expert_confidence, temperature=0.5):
    """
    Resolve disagreements between experts based on confidence scores.

    Args:
        expert_outputs: List of tensors containing logits from each expert
        expert_confidence: List of confidence scores for each expert
        temperature: Controls the sharpness of the weighting (lower = more decisive)

    Returns:
        Combined output logits with resolved disagreements
    """
    # Normalize confidence scores using softmax
    weights = F.softmax(torch.tensor(expert_confidence) / temperature, dim=0)

    # Initialize combined output with zeros
    combined_output = torch.zeros_like(expert_outputs[0])

    # Weighted combination of expert outputs
    for i, output in enumerate(expert_outputs):
        combined_output += weights[i] * output

    return combined_output
```

This algorithm creates a weighted average of expert outputs, prioritizing experts with higher confidence.

### 4.2 Confidence Estimation

We determine expert confidence through multiple signals:

```python
def calculate_expert_confidence(expert_outputs, input_text, expert_types):
    """
    Calculate confidence scores for each expert based on multiple signals.

    Args:
        expert_outputs: List of tensors containing logits from each expert
        input_text: The query text
        expert_types: List of expert types (REASON, EXPLAIN, etc.)

    Returns:
        List of confidence scores
    """
    confidence_scores = []

    # Get router's confidence in each expert for this input
    router_scores = router_classifier.get_scores(input_text)

    for i, expert_type in enumerate(expert_types):
        # Start with router confidence
        base_confidence = router_scores[expert_type]

        # Adjust based on output entropy (lower entropy = higher confidence)
        probs = F.softmax(expert_outputs[i], dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        entropy_factor = math.exp(-entropy)  # Lower entropy gives higher factor

        # Calculate self-consistency score
        self_consistency = calculate_self_consistency(expert_outputs[i])

        # Combine factors with appropriate weighting
        confidence = base_confidence * 0.4 + entropy_factor * 0.3 + self_consistency * 0.3

        confidence_scores.append(confidence)

    return confidence_scores
```

This multi-factor approach combines router confidence, output entropy, and self-consistency to determine which expert to trust.

### 4.3 Disagreement Detection

We detect significant disagreements between experts using cosine similarity of output distributions:

```python
def detect_expert_disagreement(expert_outputs, threshold=0.85):
    """
    Detect if experts significantly disagree.

    Args:
        expert_outputs: List of tensors containing logits from each expert
        threshold: Similarity threshold below which experts are considered to disagree

    Returns:
        Boolean indicating whether significant disagreement exists
    """
    # Convert logits to probability distributions
    expert_probs = [F.softmax(output, dim=-1) for output in expert_outputs]

    # Calculate pairwise cosine similarities
    num_experts = len(expert_probs)
    for i in range(num_experts):
        for j in range(i+1, num_experts):
            similarity = F.cosine_similarity(
                expert_probs[i].view(1, -1),
                expert_probs[j].view(1, -1)
            ).item()

            if similarity < threshold:
                return True  # Significant disagreement found

    return False  # No significant disagreement
```

When a significant disagreement is detected, the system can trigger additional resolution mechanisms.

### 4.4 Explicit Reasoning Reconciliation

For critical disagreements, we implement explicit reasoning reconciliation:

```python
def reconcile_with_explicit_reasoning(expert_outputs, expert_types, input_text, tokenizer, model):
    """
    Reconcile expert disagreements through explicit reasoning.

    Args:
        expert_outputs: List of tensors containing logits from each expert
        expert_types: List of expert types (REASON, EXPLAIN, etc.)
        input_text: Original query text
        tokenizer: Tokenizer for text processing
        model: Base model for generation

    Returns:
        Resolved output after explicit reasoning
    """
    # Create prompt for reconciliation
    reconciliation_prompt = f"""
    I need to reconcile different expert opinions on the following question:

    Question: {input_text}

    Expert opinions:
    """

    # Decode each expert's preferred output
    for i, expert_type in enumerate(expert_types):
        top_tokens = torch.argmax(expert_outputs[i], dim=-1)
        expert_output = tokenizer.decode(top_tokens)
        reconciliation_prompt += f"\n{expert_type} Expert: {expert_output}"

    reconciliation_prompt += "\n\nReconciled answer with explanation:"

    # Generate reconciled answer
    inputs = tokenizer(reconciliation_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=2048,
            temperature=0.3,  # Low temperature for more deterministic output
            do_sample=True
        )

    reconciled_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the reconciled part
    reconciled_part = reconciled_answer.split("Reconciled answer with explanation:")[1].strip()

    return reconciled_part
```

This approach generates an explicit reconciliation when experts have fundamentally different opinions, rather than simply averaging their outputs.

## 5. Transaction-Based MoE Routing Implementation

### 5.1 TinyLLM Transaction Classifier

The transaction classifier routes queries to the appropriate experts:

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransactionClassifier:
    """
    A lightweight classifier that routes queries to appropriate experts.
    """
    def __init__(self, model_path="path/to/transaction_classifier", device="cuda:0"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=5,  # REASON, EXPLAIN, TEACH, PREDICT, RETROSPECT
            torch_dtype=torch.float16  # Use FP16 for efficiency
        ).to(device)
        self.model.eval()
        self.id2label = {
            0: "REASON",
            1: "EXPLAIN",
            2: "TEACH",
            3: "PREDICT",
            4: "RETROSPECT"
        }
        self.threshold = 0.5  # Confidence threshold for selecting experts

    def classify(self, query):
        """
        Classify a query and return the appropriate expert types with confidence scores.

        Args:
            query: The input text query

        Returns:
            Dictionary mapping expert types to confidence scores
        """
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=0)

        # Get expert types and confidence scores
        expert_confidence = {
            self.id2label[i]: float(prob)
            for i, prob in enumerate(probs)
            if prob >= self.threshold
        }

        # If no expert meets threshold, use the best one
        if not expert_confidence:
            top_idx = torch.argmax(probs).item()
            expert_confidence = {self.id2label[top_idx]: float(probs[top_idx])}

        return expert_confidence

    def get_top_k_experts(self, query, k=2):
        """Get the top k experts for a query."""
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=0)

        # Get top k experts
        top_k_probs, top_k_indices = torch.topk(probs, k=min(k, len(probs)))

        expert_confidence = {
            self.id2label[idx.item()]: float(prob)
            for idx, prob in zip(top_k_indices, top_k_probs)
        }

        return expert_confidence
```

This classifier uses a small, efficient model to determine transaction types with minimal latency (<10ms).

### 5.2 Training the Transaction Classifier

To create an accurate classifier, we implement the following training procedure:

```python
def train_transaction_classifier(base_model="distilbert-base-uncased", train_dataset=None):
    """
    Train the transaction classifier on labeled MTG queries.

    Args:
        base_model: Base model for the classifier
        train_dataset: Dataset with labeled examples

    Returns:
        Trained classifier model and tokenizer
    """
    from transformers import Trainer, TrainingArguments

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=5,
        problem_type="single_label_classification"
    )

    # Pre-process dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_dataset = train_dataset.map(preprocess_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./classifier_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    # Train model
    trainer.train()

    # Save model
    model.save_pretrained("./transaction_classifier")
    tokenizer.save_pretrained("./transaction_classifier")

    return model, tokenizer
```

The classifier training uses a small dataset of labeled MTG queries to learn the appropriate routing patterns.

### 5.3 Expert Gating Mechanism

We implement a dynamic expert gating mechanism that activates only the selected experts:

```python
class ExpertGate(nn.Module):
    """
    Implements a transaction-based gating mechanism for MoE experts.
    Unlike traditional MoE that routes each token individually, this
    gate activates specific experts for the entire sequence based on
    transaction type.
    """
    def __init__(self, num_experts=8, hidden_size=4096):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size

        # Map from transaction types to expert indices
        self.transaction_to_experts = {
            "REASON": [0, 1],        # Primary and secondary reasoning experts
            "EXPLAIN": [2, 3],       # Primary and secondary explanation experts
            "TEACH": [4],            # Teaching expert
            "PREDICT": [5],          # Prediction expert
            "RETROSPECT": [6],       # Retrospective analysis expert
            "DEFAULT": [7]           # General purpose expert
        }

        # Expert GPU mapping
        self.expert_to_device = {
            0: "cuda:0", 1: "cuda:0", 2: "cuda:0", 3: "cuda:0",  # First 4 on GPU 0
            4: "cuda:1", 5: "cuda:1", 6: "cuda:1", 7: "cuda:1"   # Last 4 on GPU 1
        }

    def forward(self, hidden_states, transaction_types):
        """
        Forward pass that activates only the relevant experts.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            transaction_types: List of transaction types for the batch

        Returns:
            Tensor with outputs only from activated experts and zeros elsewhere
        """
        batch_size = hidden_states.shape[0]

        # Initialize outputs tensor to zeros
        expert_outputs = torch.zeros(
            (batch_size, self.num_experts, hidden_states.shape[1], self.hidden_size),
            device=hidden_states.device
        )

        # Expert activation mask (1 for active, 0 for inactive)
        expert_mask = torch.zeros((batch_size, self.num_experts), device=hidden_states.device)

        # Activate experts for each transaction type in the batch
        for i, trans_type in enumerate(transaction_types):
            if trans_type in self.transaction_to_experts:
                expert_indices = self.transaction_to_experts[trans_type]
            else:
                expert_indices = self.transaction_to_experts["DEFAULT"]

            # Set mask to 1 for active experts
            for idx in expert_indices:
                expert_mask[i, idx] = 1.0

        # Apply gating weights - we'll fill expert_outputs later with actual expert computations
        # This is a placeholder for the gating logic

        return expert_mask

    def select_experts_for_batch(self, transaction_types):
        """
        Returns a list of expert indices to activate for a batch of transactions.
        Used to determine which experts to load/offload from GPU.

        Args:
            transaction_types: List of transaction types for the batch

        Returns:
            Set of expert indices that should be active
        """
        active_experts = set()

        for trans_type in transaction_types:
            if trans_type in self.transaction_to_experts:
                indices = self.transaction_to_experts[trans_type]
            else:
                indices = self.transaction_to_experts["DEFAULT"]

            active_experts.update(indices)

        return active_experts

    def preload_experts(self, active_experts, expert_modules):
        """
        Ensures the required experts are loaded on the correct GPUs.

        Args:
            active_experts: Set of expert indices to activate
            expert_modules: Dictionary mapping expert indices to their modules

        Returns:
            None (experts are loaded to appropriate devices as a side effect)
        """
        for expert_idx in range(self.num_experts):
            target_device = self.expert_to_device[expert_idx]

            if expert_idx in active_experts:
                # Ensure expert is on the right GPU
                if expert_modules[expert_idx].device != target_device:
                    expert_modules[expert_idx].to(target_device)
            else:
                # Offload inactive experts to CPU if not needed
                expert_modules[expert_idx].to("cpu")
```

This gating mechanism ensures only the needed experts are active, maintaining sequence-level consistency while optimizing memory usage.

### 5.4 Transaction Router Integration

We integrate the transaction classifier with the MoE model:

```python
class TransactionRoutedMoE(nn.Module):
    """
    Complete MoE module with transaction-based routing integration.
    This combines the classifier, gate, and experts into a single module.
    """
    def __init__(self, experts, classifier, hidden_size=4096):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.classifier = classifier
        self.gate = ExpertGate(num_experts=len(experts), hidden_size=hidden_size)
        self.cross_expert_attention = CrossExpertAttention(hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, hidden_states, transaction_types=None, queries=None):
        """
        Forward pass through the transaction-routed MoE.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            transaction_types: List of transaction types (if already classified)
            queries: List of text queries (to classify if transaction_types not provided)

        Returns:
            Output tensor after routing through appropriate experts
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Classify transactions if types not provided
        if transaction_types is None:
            if queries is None:
                raise ValueError("Either transaction_types or queries must be provided")

            transaction_types = []
            for query in queries:
                expert_confidence = self.classifier.classify(query)
                # Get the transaction type with highest confidence
                top_transaction = max(expert_confidence.items(), key=lambda x: x[1])[0]
                transaction_types.append(top_transaction)

        # Get expert activation mask
        expert_mask = self.gate(hidden_states, transaction_types)

        # Get active expert indices for this batch
        active_experts = self.gate.select_experts_for_batch(transaction_types)

        # Process inputs through active experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            if i in active_experts:
                # Move expert to correct device if needed
                target_device = self.gate.expert_to_device[i]
                if next(expert.parameters()).device != target_device:
                    expert.to(target_device)

                # Process through expert
                output = expert(hidden_states.to(target_device))
                expert_outputs.append(output)
            else:
                # Skip inactive experts
                expert_outputs.append(None)

        # Apply cross-expert attention for active experts
        if len(active_experts) > 1:
            valid_outputs = [out for out in expert_outputs if out is not None]
            combined_output = self.cross_expert_attention(valid_outputs)
        else:
            # If only one expert, use its output directly
            combined_output = next(out for out in expert_outputs if out is not None)

        return combined_output
```

This implementation classifies the transaction type once per query and routes the entire sequence through the selected experts, maintaining consistency in generated text.

### 5.5 Multi-Expert Activation Strategy

For scenarios requiring multiple experts, we implement a strategy for combined activation:

```python
def determine_active_experts(classifier_outputs, threshold=0.3, max_experts=2):
    """
    Determine which experts to activate based on classifier confidence scores.

    Args:
        classifier_outputs: Dictionary mapping expert types to confidence scores
        threshold: Minimum confidence score to activate an expert
        max_experts: Maximum number of experts to activate

    Returns:
        List of expert types to activate
    """
    # Sort experts by confidence score
    sorted_experts = sorted(
        classifier_outputs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Select experts above threshold, up to max_experts
    active_experts = []
    for expert_type, confidence in sorted_experts:
        if confidence >= threshold and len(active_experts) < max_experts:
            active_experts.append(expert_type)

    # Always select at least one expert
    if not active_experts and sorted_experts:
        active_experts.append(sorted_experts[0][0])

    return active_experts
```

This function determines which experts to activate based on confidence thresholds, ensuring efficient multi-expert collaboration.

## 6. Cross-Expert Attention Mechanism

### 6.1 Cross-Expert Attention Implementation

To facilitate information exchange between experts, we implement a custom cross-expert attention mechanism:

```python
class CrossExpertAttention(nn.Module):
    """
    Implements attention mechanism for combining outputs from multiple experts.
    This allows experts to attend to each other's intermediate representations.
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
        self.scaling = self.head_dim ** -0.5

    def forward(self, expert_outputs):
        """
        Apply cross-expert attention to combine outputs from multiple experts.

        Args:
            expert_outputs: List of tensors [batch_size, seq_len, hidden_size] from each expert

        Returns:
            Combined output tensor after cross-expert attention
        """
        batch_size, seq_len, _ = expert_outputs[0].shape
        num_experts = len(expert_outputs)

        # Stack expert outputs along a new dimension
        # Shape: [batch_size, num_experts, seq_len, hidden_size]
        stacked_outputs = torch.stack(expert_outputs, dim=1)

        # Project to queries, keys, and values
        # Each expert's output serves as both a query and a key/value
        # Shape: [batch_size, num_experts, seq_len, hidden_size]
        queries = self.q_proj(stacked_outputs)
        keys = self.k_proj(stacked_outputs)
        values = self.v_proj(stacked_outputs)

        # Reshape for multi-head attention
        # Shape: [batch_size, num_experts, seq_len, num_heads, head_dim]
        queries = queries.view(batch_size, num_experts, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_experts, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_experts, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, seq_len, num_experts, num_heads, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Reshape for attention computation
        # Shape: [batch_size * seq_len, num_experts, num_heads, head_dim]
        queries = queries.reshape(batch_size * seq_len, num_experts, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size * seq_len, num_experts, self.num_heads, self.head_dim)
        values = values.reshape(batch_size * seq_len, num_experts, self.num_heads, self.head_dim)

        # Transpose queries and keys for attention score computation
        # Shape: [batch_size * seq_len, num_heads, num_experts, head_dim]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2).transpose(2, 3)
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

        # Reshape and combine expert dimensions
        # Shape: [batch_size, seq_len, num_experts, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads, num_experts, self.head_dim)

        # Average over the expert dimension for final output
        # Shape: [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.mean(dim=3)

        # Reshape to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        # Final projection
        output = self.o_proj(attn_output)

        return output
```

This mechanism allows experts to attend to each other's intermediate representations, facilitating information exchange and opinion reconciliation.

### 6.2 Expert Information Exchange

To improve expert collaboration, we implement a dedicated information exchange layer:

```python
class ExpertInformationExchange(nn.Module):
    """
    Facilitates information exchange between experts to improve collaboration.
    """
    def __init__(self, hidden_size=4096, exchange_ratio=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.exchange_ratio = exchange_ratio

        # Projection for information to be shared
        self.exchange_proj = nn.Linear(hidden_size, hidden_size)

        # Gate to control information flow
        self.exchange_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, expert_representations):
        """
        Exchange information between experts.

        Args:
            expert_representations: List of tensors from each expert

        Returns:
            Updated expert representations after information exchange
        """
        num_experts = len(expert_representations)

        if num_experts <= 1:
            return expert_representations

        # Stack expert representations
        # Shape: [num_experts, batch_size, seq_len, hidden_size]
        stacked_reps = torch.stack(expert_representations, dim=0)

        # Calculate shared information (average across experts)
        # Shape: [batch_size, seq_len, hidden_size]
        shared_info = torch.mean(stacked_reps, dim=0)

        # Project shared information
        shared_info = self.exchange_proj(shared_info)

        # Updated representations
        updated_representations = []

        for i, rep in enumerate(expert_representations):
            # Calculate exchange gate for this expert
            exchange_weight = self.exchange_gate(rep)

            # Combine original representation with shared information
            updated_rep = (1 - exchange_weight * self.exchange_ratio) * rep + \
                          (exchange_weight * self.exchange_ratio) * shared_info

            updated_representations.append(updated_rep)

        return updated_representations
```

This module explicitly controls how much information each expert shares with others, allowing for targeted collaboration while maintaining expert specialization.

### 6.3 Expert-Specific Prompting

We implement a mechanism for expert-specific prompting to further enhance specialist capabilities:

```python
def create_expert_specific_prompt(query, active_experts):
    """
    Create expert-specific prompts to guide each expert's reasoning process.

    Args:
        query: The original query text
        active_experts: List of active expert types

    Returns:
        Dictionary mapping expert types to modified prompts
    """
    # Base prompts for each expert type
    expert_prompts = {
        "REASON": "Consider this as a step-by-step logical reasoning task about MTG rules and mechanics. Analyze the game state methodically.",
        "EXPLAIN": "Explain the following MTG concept clearly and accurately, making it easy to understand for players.",
        "TEACH": "Teach this MTG concept with educational examples and clear explanations. Break down complex ideas into simpler parts.",
        "PREDICT": "Forecast the likely game outcomes based on the current state. Consider probabilities and potential lines of play.",
        "RETROSPECT": "Analyze what happened in this MTG scenario. Identify key decision points and opportunities for improvement."
    }

    # Create expert-specific prompts
    expert_queries = {}
    for expert_type in active_experts:
        if expert_type in expert_prompts:
            # Prefix the query with expert-specific instructions
            modified_prompt = f"<{expert_type}>\n{expert_prompts[expert_type]}\n\nQuery: {query}"
            expert_queries[expert_type] = modified_prompt
        else:
            # Fallback for unknown expert types
            expert_queries[expert_type] = f"<{expert_type}>\n{query}"

    return expert_queries
```

These expert-specific prompts guide each expert's reasoning process, ensuring specialized outputs even when processing the same query.

## 7. Hybrid KG/RAG Integration Details

### 7.1 Knowledge Graph Implementation

We implement a comprehensive knowledge graph for MTG cards, rules, and relationships:

```python
import networkx as nx
from typing import List, Dict, Any

class MTGKnowledgeGraph:
    """
    Knowledge graph for MTG cards, rules, and relationships.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.card_index = {}  # Maps card names to node IDs
        self.rule_index = {}  # Maps rule IDs to node IDs

    def build_card_graph(self, card_data: List[Dict[str, Any]]):
        """
        Build the card-related portion of the knowledge graph.

        Args:
            card_data: List of card dictionaries containing card information
        """
        # Add card nodes
        for card in card_data:
            card_id = self._normalize_name(card["name"])

            # Add node for the card
            self.graph.add_node(
                card_id,
                type="card",
                name=card["name"],
                mana_cost=card.get("mana_cost", ""),
                cmc=card.get("cmc", 0),
                colors=card.get("colors", []),
                color_identity=card.get("color_identity", []),
                card_types=card.get("types", []),
                subtypes=card.get("subtypes", []),
                text=card.get("text", ""),
                power=card.get("power", ""),
                toughness=card.get("toughness", ""),
                loyalty=card.get("loyalty", ""),
                legalities=card.get("legalities", {})
            )

            self.card_index[card["name"]] = card_id

            # Add type edges
            for card_type in card.get("types", []):
                type_node = f"type_{card_type.lower()}"

                # Create type node if it doesn't exist
                if not self.graph.has_node(type_node):
                    self.graph.add_node(type_node, type="card_type", name=card_type)

                # Add edge from card to type
                self.graph.add_edge(card_id, type_node, relation="has_type")

            # Add subtype edges
            for subtype in card.get("subtypes", []):
                subtype_node = f"subtype_{subtype.lower()}"

                # Create subtype node if it doesn't exist
                if not self.graph.has_node(subtype_node):
                    self.graph.add_node(subtype_node, type="subtype", name=subtype)

                # Add edge from card to subtype
                self.graph.add_edge(card_id, subtype_node, relation="has_subtype")

            # Add keyword edges
            for keyword in self._extract_keywords(card.get("text", "")):
                keyword_node = f"keyword_{keyword.lower()}"

                # Create keyword node if it doesn't exist
                if not self.graph.has_node(keyword_node):
                    self.graph.add_node(keyword_node, type="keyword", name=keyword)

                # Add edge from card to keyword
                self.graph.add_edge(card_id, keyword_node, relation="has_keyword")

    def build_rule_graph(self, rules_data: Dict[str, str]):
        """
        Build the rules portion of the knowledge graph.

        Args:
            rules_data: Dictionary mapping rule IDs to rule text
        """
        # Add rule nodes
        for rule_id, rule_text in rules_data.items():
            self.graph.add_node(
                f"rule_{rule_id}",
                type="rule",
                rule_id=rule_id,
                text=rule_text
            )

            self.rule_index[rule_id] = f"rule_{rule_id}"

            # Link related rules (e.g., 601.2 is related to 601.2a)
            if '.' in rule_id:
                parent_id = rule_id.split('.')[0]
                parent_node = f"rule_{parent_id}"

                if parent_node in self.graph:
                    self.graph.add_edge(f"rule_{rule_id}", parent_node, relation="sub_rule_of")

    def build_relationships(self):
        """
        Build relationships between cards, rules, and other entities.
        This could be derived from card text analysis or explicitly provided.
        """
        # Example: Link cards to relevant rules based on keywords
        for card_name, card_id in self.card_index.items():
            card_data = self.graph.nodes[card_id]

            # Link cards to relevant rules based on keywords
            for keyword in self._extract_keywords(card_data.get("text", "")):
                # Find rules related to this keyword
                related_rules = self._find_rules_for_keyword(keyword)

                for rule_id in related_rules:
                    rule_node = self.rule_index.get(rule_id)
                    if rule_node:
                        self.graph.add_edge(
                            card_id,
                            rule_node,
                            relation="governed_by"
                        )

    def query(self, query_type, **params):
        """
        Query the knowledge graph.

        Args:
            query_type: Type of query to perform
            params: Parameters for the query

        Returns:
            Query results
        """
        if query_type == "card_by_name":
            return self._query_card_by_name(params.get("name", ""))
        elif query_type == "cards_by_type":
            return self._query_cards_by_type(params.get("card_type", ""))
        elif query_type == "rule_by_id":
            return self._query_rule_by_id(params.get("rule_id", ""))
        elif query_type == "rules_for_card":
            return self._query_rules_for_card(params.get("card_name", ""))
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def _query_card_by_name(self, card_name):
        """Get card data by name."""
        card_id = self.card_index.get(card_name)
        if not card_id:
            # Try normalized name
            card_id = self.card_index.get(self._normalize_name(card_name))

        if card_id and card_id in self.graph:
            return self.graph.nodes[card_id]

        return None

    def _query_cards_by_type(self, card_type):
        """Get all cards of a specific type."""
        type_node = f"type_{card_type.lower()}"

        if not self.graph.has_node(type_node):
            return []

        # Find all cards with an edge to this type
        cards = []
        for card_id, _, data in self.graph.in_edges(type_node, data=True):
            if data.get("relation") == "has_type" and self.graph.nodes[card_id]["type"] == "card":
                cards.append(self.graph.nodes[card_id])

        return cards

    def _query_rule_by_id(self, rule_id):
        """Get rule by ID."""
        rule_node = self.rule_index.get(rule_id)

        if rule_node and rule_node in self.graph:
            return self.graph.nodes[rule_node]

        return None

    def _query_rules_for_card(self, card_name):
        """Get rules relevant to a specific card."""
        card_id = self.card_index.get(card_name)
        if not card_id:
            # Try normalized name
            card_id = self.card_index.get(self._normalize_name(card_name))

        if not card_id:
            return []

        # Find all rule nodes connected to this card
        rules = []
        for _, rule_node, data in self.graph.out_edges(card_id, data=True):
            if data.get("relation") == "governed_by" and self.graph.nodes[rule_node]["type"] == "rule":
                rules.append(self.graph.nodes[rule_node])

        return rules

    def _normalize_name(self, name):
        """Normalize card name for consistent lookup."""
        return name.lower().replace(' ', '_').replace(',', '').replace("'", '')

    def _extract_keywords(self, text):
        """Extract MTG keywords from card text."""
        # List of common MTG keywords
        keywords = [
            "Flying", "First Strike", "Double Strike", "Deathtouch", "Haste",
            "Hexproof", "Indestructible", "Lifelink", "Menace", "Reach",
            "Trample", "Vigilance", "Flash", "Defender", "Prowess"
        ]

        found_keywords = []
        for keyword in keywords:
            if keyword.lower() in text.lower():
                found_keywords.append(keyword)

        return found_keywords

    def _find_rules_for_keyword(self, keyword):
        """Find rules related to a keyword."""
        # This is a simplified version - in practice would use more sophisticated matching
        keyword_to_rules = {
            "Flying": ["702.9"],
            "First Strike": ["702.7"],
            "Double Strike": ["702.4"],
            "Deathtouch": ["702.2"],
            "Haste": ["702.10"],
            "Hexproof": ["702.11"],
            "Indestructible": ["702.12"],
            "Lifelink": ["702.15"],
            "Menace": ["702.16"],
            "Reach": ["702.17"],
            "Trample": ["702.19"],
            "Vigilance": ["702.20"],
            "Flash": ["702.8"],
            "Defender": ["702.3"],
            "Prowess": ["702.107"]
        }

        return keyword_to_rules.get(keyword, [])
```

This knowledge graph implementation captures card properties, rule relationships, and semantic connections for high-precision data retrieval.

### 7.2 Retrieval-Augmented Generation (RAG) System

We implement a RAG system for efficient text retrieval:

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional

class MTGRetriever:
    """
    Retrieval-Augmented Generation system for MTG-related text.
    """
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.document_index = None
        self.documents = []
        self.document_types = []
        self.document_ids = []

    def index_documents(self, documents: List[Dict[str, str]]):
        """
        Index documents for retrieval.

        Args:
            documents: List of dictionaries containing document text and metadata
                Each document should have 'text', 'type', and 'id' fields
        """
        # Extract text, type, and ID from documents
        texts = [doc["text"] for doc in documents]
        self.document_types = [doc["type"] for doc in documents]
        self.document_ids = [doc["id"] for doc in documents]
        self.documents = texts

        # Create embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        vector_dimension = embeddings.shape[1]
        self.document_index = faiss.IndexFlatIP(vector_dimension)
        self.document_index.add(embeddings)

        print(f"Indexed {len(texts)} documents")

    def retrieve(self, query: str, top_k: int = 5, doc_type: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Retrieve relevant documents based on the query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve
            doc_type: Optional filter for document type

        Returns:
            List of retrieved documents with relevance scores
        """
        if self.document_index is None:
            raise ValueError("Document index not initialized. Call index_documents first.")

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar documents
        scores, indices = self.document_index.search(query_embedding, len(self.documents))

        # Filter by document type if specified
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.documents):
                continue

            doc = {
                "text": self.documents[idx],
                "type": self.document_types[idx],
                "id": self.document_ids[idx],
                "score": float(score)
            }

            if doc_type is None or doc["type"] == doc_type:
                results.append(doc)

            if len(results) >= top_k:
                break

        return results

    def retrieve_by_categories(self, query: str, top_k_per_type: int = 3) -> Dict[str, List[Dict[str, any]]]:
        """
        Retrieve documents by category, retrieving top_k for each document type.

        Args:
            query: Query text
            top_k_per_type: Number of documents to retrieve per type

        Returns:
            Dictionary mapping document types to retrieved documents
        """
        if self.document_index is None:
            raise ValueError("Document index not initialized. Call index_documents first.")

        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search for similar documents
        scores, indices = self.document_index.search(query_embedding, len(self.documents))

        # Group by document type
        results_by_type = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx >= len(self.documents):
                continue

            doc_type = self.document_types[idx]

            if doc_type not in results_by_type:
                results_by_type[doc_type] = []

            if len(results_by_type[doc_type]) < top_k_per_type:
                doc = {
                    "text": self.documents[idx],
                    "type": doc_type,
                    "id": self.document_ids[idx],
                    "score": float(score)
                }
                results_by_type[doc_type].append(doc)

        return results_by_type
```

This retrieval system enables efficient semantic search across card descriptions, rules text, and other MTG knowledge sources.

### 7.3 Dynamic Knowledge Selection Mechanism

We implement a mechanism to dynamically select between KG and RAG:

```python
class DynamicKnowledgeSelector:
    """
    Dynamically decides whether to use KG or RAG based on query type.
    """
    def __init__(self, kg, retriever):
        self.kg = kg
        self.retriever = retriever

        # Compiled patterns for query classification
        self.kg_patterns = {
            "card_lookup": r"\b(what|find|show|get|tell me about)\s+(.+?)\s+(card|cards)\b",
            "rule_lookup": r"\b(rule|section|paragraph)\s+(\d+\.\d+[a-z]?)\b",
            "format_legality": r"\b(legal|banned|restricted)\s+in\s+(standard|modern|commander|legacy|vintage)\b",
            "card_type_search": r"\ball\s+(creatures|instants|sorceries|artifacts|enchantments|planeswalkers)\s+with\s+(.+)\b"
        }

        self.rag_patterns = {
            "complex_rule": r"\b(how does|what happens|interact|stack|trigger|resolve)\b",
            "meta_question": r"\b(best|top|meta|current|popular|strongest)\s+(deck|strategy|card)\b",
            "strategy_advice": r"\b(strategy|advice|tips|how to play|how to use|counters|against)\b",
            "card_comparison": r"\b(better|worse|stronger|comparing|compare|versus|vs\.?)\b"
        }

    def select_knowledge_source(self, query):
        """
        Determine whether to use KG, RAG, or both based on the query.

        Args:
            query: User query text

        Returns:
            Tuple (use_kg, use_rag, query_type)
        """
        # Default to using both
        use_kg = True
        use_rag = True
        query_type = "general"

        # Check if query matches KG patterns
        for pattern_type, pattern in self.kg_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                use_kg = True
                use_rag = False
                query_type = pattern_type
                break

        # Check if query matches RAG patterns
        for pattern_type, pattern in self.rag_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                use_kg = False
                use_rag = True
                query_type = pattern_type
                break

        # For card name detection, always check KG
        card_names = self._extract_card_names(query)
        if card_names and query_type == "general":
            use_kg = True

        return use_kg, use_rag, query_type

    def get_knowledge(self, query):
        """
        Retrieve knowledge based on the query.

        Args:
            query: User query text

        Returns:
            Dictionary with retrieved knowledge
        """
        use_kg, use_rag, query_type = self.select_knowledge_source(query)

        result = {
            "knowledge_source": [],
            "kg_data": None,
            "rag_data": None,
            "query_type": query_type
        }

        # Get knowledge from KG if needed
        if use_kg:
            kg_data = self._get_kg_data(query, query_type)
            result["kg_data"] = kg_data
            result["knowledge_source"].append("kg")

        # Get knowledge from RAG if needed
        if use_rag:
            rag_data = self._get_rag_data(query, query_type)
            result["rag_data"] = rag_data
            result["knowledge_source"].append("rag")

        return result

    def _get_kg_data(self, query, query_type):
        """Get data from knowledge graph based on query type."""
        # Extract card names for card-related queries
        card_names = self._extract_card_names(query)

        # Extract rule numbers for rule-related queries
        rule_ids = self._extract_rule_ids(query)

        # Handle different query types
        if query_type == "card_lookup" or card_names:
            # Return card data for the first detected card
            card_data = []
            for card_name in card_names:
                card = self.kg.query("card_by_name", name=card_name)
                if card:
                    card_data.append(card)
            return {"type": "card_data", "data": card_data}

        elif query_type == "rule_lookup" or rule_ids:
            # Return rule data for the detected rule ID
            rule_data = []
            for rule_id in rule_ids:
                rule = self.kg.query("rule_by_id", rule_id=rule_id)
                if rule:
                    rule_data.append(rule)
            return {"type": "rule_data", "data": rule_data}

        elif query_type == "card_type_search":
            # Extract card type from query
            match = re.search(r"all\s+(creatures|instants|sorceries|artifacts|enchantments|planeswalkers)\s+with\s+(.+)", query, re.IGNORECASE)
            if match:
                card_type = match.group(1).rstrip('s')  # Remove trailing 's' if present
                cards = self.kg.query("cards_by_type", card_type=card_type)
                return {"type": "card_list", "data": cards}

        # Fallback: return empty data
        return {"type": "empty", "data": []}

    def _get_rag_data(self, query, query_type):
        """Get data from RAG based on query type."""
        # For complex rule questions, focus on rule documents
        if query_type == "complex_rule":
            return self.retriever.retrieve(query, top_k=5, doc_type="rule")

        # For meta questions, focus on deck guides and meta reports
        elif query_type == "meta_question":
            return self.retriever.retrieve(query, top_k=5, doc_type="meta")

        # For strategy advice, focus on strategy guides
        elif query_type == "strategy_advice":
            return self.retriever.retrieve(query, top_k=5, doc_type="strategy")

        # For card comparisons, retrieve from all sources
        elif query_type == "card_comparison":
            return self.retriever.retrieve_by_categories(query, top_k_per_type=3)

        # General case: retrieve from all sources
        else:
            return self.retriever.retrieve(query, top_k=5)

    def _extract_card_names(self, query):
        """Extract card names from the query using a named entity recognition approach."""
        # This is a simplified placeholder - in practice, would use a more sophisticated approach
        # such as a trained NER model or matching against a card name list
        card_names = []

        # Example implementation using a regex pattern for card names in double brackets
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, query)
        card_names.extend(matches)

        return card_names

    def _extract_rule_ids(self, query):
        """Extract rule IDs from the query."""
        pattern = r'\b(\d+\.\d+[a-z]?)\b'
        matches = re.findall(pattern, query)
        return matches
```

This selector intelligently determines which knowledge source to use based on query patterns, optimizing for both precision and retrieval speed.

### 7.4 Knowledge Integration Pipeline

We integrate the KG and RAG components with the main inference pipeline:

```python
class MTGKnowledgeIntegrationPipeline:
    """
    Pipeline for integrating knowledge from KG and RAG into LLM prompts.
    """
    def __init__(self, kg, retriever, tokenizer, max_tokens=1536):
        self.knowledge_selector = DynamicKnowledgeSelector(kg, retriever)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def enhance_prompt_with_knowledge(self, query, active_experts):
        """
        Enhance the prompt with relevant knowledge from KG and RAG.

        Args:
            query: User query text
            active_experts: List of active expert types

        Returns:
            Enhanced prompt with integrated knowledge
        """
        # Get knowledge from selector
        knowledge = self.knowledge_selector.get_knowledge(query)

        # Create base prompt with expert-specific instructions
        expert_prompts = create_expert_specific_prompt(query, active_experts)

        # Enhance each expert prompt with knowledge
        enhanced_prompts = {}
        for expert_type, prompt in expert_prompts.items():
            # Create knowledge section based on expert type
            knowledge_section = self._format_knowledge_for_expert(
                knowledge, expert_type, query
            )

            # Combine prompt with knowledge
            if knowledge_section:
                enhanced_prompt = f"{prompt}\n\n[KNOWLEDGE]\n{knowledge_section}\n[/KNOWLEDGE]"
            else:
                enhanced_prompt = prompt

            # Truncate if necessary to fit into context window
            enhanced_prompt = self._truncate_prompt(enhanced_prompt)

            enhanced_prompts[expert_type] = enhanced_prompt

        return enhanced_prompts

    def _format_knowledge_for_expert(self, knowledge, expert_type, query):
        """Format knowledge based on expert type."""
        formatted_knowledge = ""

        # Handle KG data if present
        if "kg" in knowledge["knowledge_source"] and knowledge["kg_data"]:
            kg_section = self._format_kg_data(knowledge["kg_data"], expert_type)
            formatted_knowledge += kg_section

        # Handle RAG data if present
        if "rag" in knowledge["knowledge_source"] and knowledge["rag_data"]:
            rag_section = self._format_rag_data(knowledge["rag_data"], expert_type)

            # Add separator if there's already KG content
            if formatted_knowledge:
                formatted_knowledge += "\n\n"

            formatted_knowledge += rag_section

        return formatted_knowledge

    def _format_kg_data(self, kg_data, expert_type):
        """Format knowledge graph data based on expert type."""
        data_type = kg_data["type"]
        data = kg_data["data"]

        if not data:
            return ""

        formatted_text = ""

        if data_type == "card_data":
            # Format card data
            formatted_text = "Card Information:\n\n"

            for card in data:
                formatted_text += f"Name: {card.get('name', 'Unknown')}\n"
                formatted_text += f"Types: {', '.join(card.get('card_types', []))}\n"

                if card.get('subtypes'):
                    formatted_text += f"Subtypes: {', '.join(card.get('subtypes', []))}\n"

                if card.get('mana_cost'):
                    formatted_text += f"Mana Cost: {card.get('mana_cost')}\n"

                if card.get('text'):
                    formatted_text += f"Text: {card.get('text')}\n"

                if card.get('power') and card.get('toughness'):
                    formatted_text += f"Power/Toughness: {card.get('power')}/{card.get('toughness')}\n"

                if card.get('loyalty'):
                    formatted_text += f"Loyalty: {card.get('loyalty')}\n"

                formatted_text += "\n"

        elif data_type == "rule_data":
            # Format rule data
            formatted_text = "Rules Information:\n\n"

            for rule in data:
                formatted_text += f"Rule {rule.get('rule_id', 'Unknown')}: {rule.get('text', '')}\n\n"

        elif data_type == "card_list":
            # Format card list (simplified for space)
            formatted_text = "Relevant Cards:\n\n"

            for i, card in enumerate(data[:10]):  # Limit to 10 cards
                formatted_text += f"{i+1}. {card.get('name', 'Unknown')} - {card.get('text', '')}\n"

        return formatted_text

    def _format_rag_data(self, rag_data, expert_type):
        """Format RAG data based on expert type."""
        if isinstance(rag_data, dict):
            # Handle retrieval by categories result
            formatted_text = "Retrieved Information:\n\n"

            for doc_type, docs in rag_data.items():
                formatted_text += f"{doc_type.capitalize()} Documents:\n"

                for doc in docs:
                    formatted_text += f"- {doc['text'][:300]}...\n\n"
        else:
            # Handle regular retrieval result
            formatted_text = "Retrieved Information:\n\n"

            for doc in rag_data:
                formatted_text += f"- {doc['text'][:300]}...\n\n"

        return formatted_text

    def _truncate_prompt(self, prompt):
        """Truncate prompt to fit within token limit."""
        tokens = self.tokenizer.encode(prompt)

        if len(tokens) <= self.max_tokens:
            return prompt

        # Truncate tokens and decode
        truncated_tokens = tokens[:self.max_tokens]
        truncated_prompt = self.tokenizer.decode(truncated_tokens)

        return truncated_prompt
```

This pipeline intelligently combines knowledge from both sources, formats it according to expert requirements, and ensures it fits within token limits.

## 8. Code Implementation Examples (Key Components)

### 8.1 Complete MTG Assistant Implementation

Here is a complete implementation of the MTG AI Assistant class integrating all components:

```python
class MTGAIAssistant:
    """
    Complete MTG AI Assistant that integrates all components:
    - Transaction classification
    - Expert activation and routing
    - Knowledge retrieval and integration
    - Cross-expert collaboration
    - Response generation
    """
    def __init__(self, model_path="mistralai/Mixtral-8x7B-v0.1", device_map="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load base model with quantization
        self.model = load_quantized_model(model_path)

        # Initialize transaction classifier
        self.classifier = TransactionClassifier()

        # Initialize knowledge components
        self.kg = MTGKnowledgeGraph()
        self.retriever = MTGRetriever()
        self.knowledge_pipeline = MTGKnowledgeIntegrationPipeline(
            self.kg, self.retriever, self.tokenizer
        )

        # Initialize MoE components
        self.expert_gate = ExpertGate()
        self.cross_expert_attention = CrossExpertAttention()
        self.expert_info_exchange = ExpertInformationExchange()

        # Load expert adapters (LoRA)
        self.expert_adapters = self._load_expert_adapters()

        # Initialize metrics tracking
        self.metrics = {
            "queries_processed": 0,
            "expert_usage": {
                "REASON": 0,
                "EXPLAIN": 0,
                "TEACH": 0,
                "PREDICT": 0,
                "RETROSPECT": 0
            },
            "knowledge_source_usage": {
                "kg": 0,
                "rag": 0,
                "both": 0,
                "none": 0
            },
            "multi_expert_activations": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }

    def _load_expert_adapters(self):
        """Load LoRA adapters for each expert."""
        expert_adapters = {}

        expert_types = ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"]

        for expert_type in expert_types:
            try:
                adapter_path = f"./models/{expert_type.lower()}_adapter"
                adapter = PeftModel.from_pretrained(self.model, adapter_path)
                expert_adapters[expert_type] = adapter
                print(f"Loaded adapter for {expert_type}")
            except Exception as e:
                print(f"Failed to load adapter for {expert_type}: {e}")
                # Create placeholder for missing adapter
                expert_adapters[expert_type] = None

        return expert_adapters

    def answer_query(self, query):
        """
        Process a query and generate a response.

        Args:
            query: User query text

        Returns:
            Generated response
        """
        start_time = time.time()

        # Update metrics
        self.metrics["queries_processed"] += 1

        # Step 1: Classify transaction and select experts
        expert_confidence = self.classifier.classify(query)
        active_experts = list(expert_confidence.keys())

        # Update metrics for expert usage
        for expert_type in active_experts:
            self.metrics["expert_usage"][expert_type] += 1

        if len(active_experts) > 1:
            self.metrics["multi_expert_activations"] += 1

        # Step 2: Retrieve relevant knowledge
        enhanced_prompts = self.knowledge_pipeline.enhance_prompt_with_knowledge(
            query, active_experts
        )

        # Track which knowledge sources were used
        knowledge = self.knowledge_pipeline.knowledge_selector.select_knowledge_source(query)[0:2]
        if knowledge[0] and knowledge[1]:
            self.metrics["knowledge_source_usage"]["both"] += 1
        elif knowledge[0]:
            self.metrics["knowledge_source_usage"]["kg"] += 1
        elif knowledge[1]:
            self.metrics["knowledge_source_usage"]["rag"] += 1
        else:
            self.metrics["knowledge_source_usage"]["none"] += 1

        # Step 3: Generate expert-specific outputs
        expert_outputs = {}
        for expert_type in active_experts:
            # Skip if adapter not available
            if not self.expert_adapters[expert_type]:
                continue

            # Apply expert adapter
            self._apply_expert_adapter(expert_type)

            # Tokenize the enhanced prompt
            inputs = self.tokenizer(
                enhanced_prompts[expert_type],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # Generate output with the expert
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=4096,
                    temperature=0.7 if expert_type in ["EXPLAIN", "TEACH"] else 0.3,
                    do_sample=True
                )

            # Decode and store the expert's output
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            expert_outputs[expert_type] = output_text

        # Step 4: Resolve disagreements or combine expert outputs
        if len(expert_outputs) > 1:
            final_response = self._combine_expert_outputs(
                expert_outputs, expert_confidence, query
            )
        elif len(expert_outputs) == 1:
            # Use the only available expert's output
            final_response = list(expert_outputs.values())[0]
        else:
            # Fallback if no experts generated output
            final_response = "I don't have enough information to provide a specific answer about Magic: The Gathering for that question."

        # Step 5: Format final response and update metrics
        end_time = time.time()
        response_time = end_time - start_time

        self.metrics["total_response_time"] += response_time
        self.metrics["avg_response_time"] = (
            self.metrics["total_response_time"] / self.metrics["queries_processed"]
        )

        return final_response

    def _apply_expert_adapter(self, expert_type):
        """Apply a specific expert's LoRA adapter to the model."""
        # Deactivate all adapters first
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()

        # Activate the requested expert's adapter
        adapter = self.expert_adapters[expert_type]
        if adapter and hasattr(adapter, "enable_adapter"):
            adapter.enable_adapter()

    def _combine_expert_outputs(self, expert_outputs, expert_confidence, query):
        """
        Combine outputs from multiple experts into a cohesive response.

        Args:
            expert_outputs: Dictionary mapping expert types to their outputs
            expert_confidence: Dictionary mapping expert types to confidence scores
            query: Original query text

        Returns:
            Combined response
        """
        # Check if experts significantly disagree
        has_disagreement = self._detect_expert_disagreement(expert_outputs)

        if has_disagreement:
            # Use explicit reconciliation
            combined_response = self._reconcile_with_explanation(
                expert_outputs, expert_confidence, query
            )
        else:
            # Simple combination
            if "REASON" in expert_outputs and "EXPLAIN" in expert_outputs:
                # Combine reasoning with explanation
                combined_response = f"{expert_outputs['EXPLAIN']}\n\nAdditional technical details:\n{self._extract_key_insights(expert_outputs['REASON'])}"

            elif "TEACH" in expert_outputs:
                # Teaching response takes precedence for educational content
                combined_response = expert_outputs["TEACH"]

                # Add prediction if available
                if "PREDICT" in expert_outputs:
                    prediction_insight = self._extract_key_insights(expert_outputs["PREDICT"])
                    combined_response += f"\n\nPrediction for gameplay outcomes:\n{prediction_insight}"

            elif "PREDICT" in expert_outputs and "REASON" in expert_outputs:
                # Combine prediction with reasoning
                combined_response = f"{expert_outputs['PREDICT']}\n\nThis prediction is based on the following analysis:\n{self._extract_key_insights(expert_outputs['REASON'])}"

            else:
                # Default fallback: combine all outputs with headers
                combined_response = ""
                for expert_type, output in expert_outputs.items():
                    combined_response += f"## {expert_type} Perspective\n{output}\n\n"

        return combined_response

    def _detect_expert_disagreement(self, expert_outputs):
        """
        Detect if experts significantly disagree in their outputs.

        Args:
            expert_outputs: Dictionary mapping expert types to their outputs

        Returns:
            Boolean indicating whether significant disagreement exists
        """
        # This is a simplified implementation - in practice, would use more
        # sophisticated NLP techniques to detect semantic disagreements

        # If only one expert, no disagreement
        if len(expert_outputs) <= 1:
            return False

        # Look for contradictory keywords
        contradiction_indicators = [
            ("can", "cannot"),
            ("legal", "illegal"),
            ("allowed", "not allowed"),
            ("works", "doesn't work"),
            ("will", "will not"),
            ("yes", "no")
        ]

        # Check for contradictions between any pair of experts
        expert_types = list(expert_outputs.keys())
        for i in range(len(expert_types)):
            for j in range(i+1, len(expert_types)):
                type_i = expert_types[i]
                type_j = expert_types[j]

                # Compare outputs for contradictions
                for pos, neg in contradiction_indicators:
                    if (pos in expert_outputs[type_i].lower() and neg in expert_outputs[type_j].lower()) or \
                       (neg in expert_outputs[type_i].lower() and pos in expert_outputs[type_j].lower()):
                        return True

        return False

    def _reconcile_with_explanation(self, expert_outputs, expert_confidence, query):
        """
        Reconcile conflicting expert outputs with an explanation.

        Args:
            expert_outputs: Dictionary mapping expert types to their outputs
            expert_confidence: Dictionary mapping expert types to confidence scores
            query: Original query text

        Returns:
            Reconciled response with explanation
        """
        reconciliation_prompt = f"""
        I need to reconcile different expert perspectives on this Magic: The Gathering question:

        Question: {query}

        Expert perspectives:
        """

        for expert_type, output in expert_outputs.items():
            reconciliation_prompt += f"\n\n## {expert_type} Expert (Confidence: {expert_confidence.get(expert_type, 0):.2f}):\n{output}"

        reconciliation_prompt += "\n\nPlease provide a unified answer that addresses these different perspectives and explains any apparent contradictions:"

        # Generate reconciliation using the base model without specific expert
        inputs = self.tokenizer(
            reconciliation_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3072
        ).to(self.model.device)

        # Temporarily disable all adapters for neutral reconciliation
        if hasattr(self.model, "disable_adapter"):
            self.model.disable_adapter()

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=4096,
                temperature=0.3,
                do_sample=True
            )

        # Extract reconciliation from after the prompt
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        reconciliation = output_text.split("Please provide a unified answer that addresses these different perspectives and explains any apparent contradictions:")[-1].strip()

        return reconciliation

    def _extract_key_insights(self, text, max_length=500):
        """Extract key insights from a longer text."""
        # This is a simplified implementation - in practice, would use more
        # sophisticated summarization techniques

        # Simple truncation for demonstration
        if len(text) <= max_length:
            return text

        # Try to find paragraph breaks for cleaner truncation
        paragraphs = text.split('\n\n')
        result = ""

        for paragraph in paragraphs:
            if len(result + paragraph) < max_length:
                result += paragraph + "\n\n"
            else:
                break

        if not result:  # If no paragraph breaks, just truncate
            result = text[:max_length] + "..."

        return result

    def get_metrics(self):
        """Return current usage metrics."""
        return self.metrics
```

This comprehensive implementation encapsulates all the components we've developed into a single cohesive system.

### 8.2 Dual-GPU Memory Management

Here's a detailed implementation of the memory management system for dual GPUs:

```python
def optimize_memory_usage(model, expert_indices, current_device_map):
    """
    Optimize memory usage by moving experts between devices as needed.

    Args:
        model: The MoE model
        expert_indices: List of expert indices to activate
        current_device_map: Current mapping of experts to devices

    Returns:
        Updated device map
    """
    import gc
    import torch

    # Ensure GPU memory is cleared
    gc.collect()
    torch.cuda.empty_cache()

    # Define target device map based on active experts
    target_device_map = current_device_map.copy()

    # Count experts per device for load balancing
    device_counts = {"cuda:0": 0, "cuda:1": 0}
    for idx in expert_indices:
        if idx in current_device_map:
            device = current_device_map[idx]
            device_counts[device] += 1

    # Balance experts across devices
    if len(expert_indices) > 1:
        # Sort active experts to prioritize movement
        sorted_experts = sorted(
            expert_indices,
            key=lambda idx: device_counts[current_device_map.get(idx, "cuda:0")]
        )

        # Assign experts to balance load
        for idx in sorted_experts:
            if device_counts["cuda:0"] <= device_counts["cuda:1"]:
                target_device_map[idx] = "cuda:0"
                device_counts["cuda:0"] += 1
            else:
                target_device_map[idx] = "cuda:1"
                device_counts["cuda:1"] += 1

    # Calculate memory needed for each device after rebalancing
    memory_needed = {
        "cuda:0": sum(8 if target_device_map.get(i, "cuda:0") == "cuda:0" else 0 for i in range(8)),
        "cuda:1": sum(8 if target_device_map.get(i, "cuda:1") == "cuda:1" else 0 for i in range(8))
    }

    # Check if memory allocation is within limits
    gpu_memory = {
        "cuda:0": torch.cuda.get_device_properties(0).total_memory // (1024**2),
        "cuda:1": torch.cuda.get_device_properties(1).total_memory // (1024**2)
    }

    # Adjust if memory would be exceeded
    if memory_needed["cuda:0"] > gpu_memory["cuda:0"] * 0.9:
        # Move some experts to cuda:1
        overflow_experts = [
            idx for idx in range(8)
            if target_device_map.get(idx, "cuda:0") == "cuda:0"
            and idx not in expert_indices
        ]

        # Move non-active experts to the other device
        for idx in overflow_experts:
            target_device_map[idx] = "cuda:1"

            # Update memory tracking
            memory_needed["cuda:0"] -= 8
            memory_needed["cuda:1"] += 8

            # Stop if we've freed enough memory
            if memory_needed["cuda:0"] <= gpu_memory["cuda:0"] * 0.9:
                break

    # Do the same for cuda:1 if needed
    if memory_needed["cuda:1"] > gpu_memory["cuda:1"] * 0.9:
        overflow_experts = [
            idx for idx in range(8)
            if target_device_map.get(idx, "cuda:1") == "cuda:1"
            and idx not in expert_indices
        ]

        for idx in overflow_experts:
            target_device_map[idx] = "cuda:0"

            memory_needed["cuda:1"] -= 8
            memory_needed["cuda:0"] += 8

            if memory_needed["cuda:1"] <= gpu_memory["cuda:1"] * 0.9:
                break

    # Actually move experts between devices
    for idx, device in target_device_map.items():
        if idx < len(model.experts) and device != current_device_map.get(idx):
            print(f"Moving expert {idx} to {device}")
            model.experts[idx].to(device)

    # Additional optimization: Offload inactive experts to CPU if we're really tight on memory
    if max(memory_needed.values()) > max(gpu_memory.values()) * 0.95:
        print("Memory pressure high, offloading inactive experts to CPU")
        for idx in range(len(model.experts)):
            if idx not in expert_indices:
                model.experts[idx].to("cpu")
                target_device_map[idx] = "cpu"

    return target_device_map
```

This implementation intelligently redistributes experts across GPUs to maximize utilization while preventing out-of-memory errors. It handles dynamic resource allocation and can even offload inactive experts to CPU when memory pressure is high.

### 8.3 Efficient Inference Pipeline

Here's an implementation of the complete inference pipeline optimized for latency:

```python
class EfficientInferencePipeline:
    """
    Optimized inference pipeline that minimizes latency while maximizing quality.
    """
    def __init__(self, model, tokenizer, experts, classifier, knowledge_pipeline):
        self.model = model
        self.tokenizer = tokenizer
        self.experts = experts
        self.classifier = classifier
        self.knowledge_pipeline = knowledge_pipeline

        # Caching for efficient retrieval
        self.kv_cache = {}
        self.knowledge_cache = {}

        # Device mapping for experts
        self.device_map = {i: "cuda:0" if i < 4 else "cuda:1" for i in range(len(experts))}

        # Performance tracking
        self.inference_times = []

    def generate_response(self, query, stream=True, max_length=2048):
        """
        Generate a response for a query with optimized inference.

        Args:
            query: User query text
            stream: Whether to stream the response token by token
            max_length: Maximum response length

        Returns:
            Generated response
        """
        start_time = time.time()

        # Step 1: Classify transaction type
        expert_confidence = self.classifier.classify(query)
        active_experts = list(expert_confidence.keys())

        # Step 2: Retrieve and format knowledge
        query_hash = hashlib.md5(query.encode()).hexdigest()

        if query_hash in self.knowledge_cache:
            enhanced_prompts = self.knowledge_cache[query_hash]
        else:
            enhanced_prompts = self.knowledge_pipeline.enhance_prompt_with_knowledge(
                query, active_experts
            )
            self.knowledge_cache[query_hash] = enhanced_prompts

        # Step 3: Optimize expert device placement
        active_expert_indices = [self._get_expert_index(expert_type) for expert_type in active_experts]
        self.device_map = optimize_memory_usage(self.model, active_expert_indices, self.device_map)

        # Step 4: Generate with optimized pipeline
        inputs = {}
        for expert_type in active_experts:
            inputs[expert_type] = self.tokenizer(
                enhanced_prompts[expert_type],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )

        # If streaming, use an asynchronous approach
        if stream:
            return self._generate_streaming(inputs, active_experts, max_length)

        # Non-streaming generation
        expert_outputs = {}
        for expert_type in active_experts:
            expert_idx = self._get_expert_index(expert_type)
            expert_device = self.device_map[expert_idx]

            # Move inputs to correct device
            device_inputs = {
                key: val.to(expert_device) for key, val in inputs[expert_type].items()
            }

            # Activate the correct expert
            self._activate_expert(expert_idx)

            # Generate with this expert
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=device_inputs["input_ids"],
                    attention_mask=device_inputs["attention_mask"],
                    max_length=max_length,
                    temperature=0.7 if expert_type in ["EXPLAIN", "TEACH"] else 0.3,
                    do_sample=True,
                    use_cache=True
                )

            # Decode and store output
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            expert_outputs[expert_type] = output_text

        # Combine expert outputs
        final_response = self._combine_expert_outputs(expert_outputs, expert_confidence)

        # Record inference time
        end_time = time.time()
        inference_time = end_time - start_time
        self.inference_times.append(inference_time)

        return final_response

    def _generate_streaming(self, inputs, active_experts, max_length):
        """Generate response with token-by-token streaming."""
        import asyncio

        # Initialize generators for each expert
        expert_generators = {}
        for expert_type in active_experts:
            expert_idx = self._get_expert_index(expert_type)
            expert_device = self.device_map[expert_idx]

            # Move inputs to correct device
            device_inputs = {
                key: val.to(expert_device) for key, val in inputs[expert_type].items()
            }

            # Activate the correct expert
            self._activate_expert(expert_idx)

            # Create generator
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_special_tokens=True, timeout=10.0
            )

            # Start generation in separate thread
            threading.Thread(
                target=self._generate_thread,
                args=(
                    device_inputs["input_ids"],
                    device_inputs["attention_mask"],
                    max_length,
                    0.7 if expert_type in ["EXPLAIN", "TEACH"] else 0.3,
                    streamer
                )
            ).start()

            expert_generators[expert_type] = streamer

        # Return streamer iterators for frontend to consume
        return expert_generators

    def _generate_thread(self, input_ids, attention_mask, max_length, temperature, streamer):
        """Thread function for generation without blocking."""
        try:
            with torch.no_grad():
                self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    streamer=streamer,
                    use_cache=True
                )
        except Exception as e:
            print(f"Generation error: {e}")
            streamer.stop()

    def _get_expert_index(self, expert_type):
        """Map expert type to expert index."""
        expert_mapping = {
            "REASON": 0,
            "EXPLAIN": 2,
            "TEACH": 4,
            "PREDICT": 5,
            "RETROSPECT": 6
        }
        return expert_mapping.get(expert_type, 7)  # Default to general expert (7)

    def _activate_expert(self, expert_idx):
        """Activate a specific expert and deactivate others."""
        # Create expert mask with 1.0 for active expert, 0.0 for others
        expert_mask = torch.zeros(len(self.experts))
        expert_mask[expert_idx] = 1.0

        # Apply mask to model's gating mechanism
        for layer in self.model.layers:
            if hasattr(layer, "block_sparse_moe"):
                layer.block_sparse_moe.gate.set_expert_mask(expert_mask)

    def _combine_expert_outputs(self, expert_outputs, expert_confidence):
        """Combine outputs from multiple experts."""
        # This is a simplified version - refer to the full implementation
        # in the MTGAIAssistant class for a more comprehensive approach

        if len(expert_outputs) == 1:
            return list(expert_outputs.values())[0]

        combined_output = "I've analyzed this from multiple perspectives:\n\n"

        for expert_type, output in expert_outputs.items():
            combined_output += f"## {expert_type} Analysis (Confidence: {expert_confidence[expert_type]:.2f})\n"
            combined_output += output + "\n\n"

        return combined_output

    def clear_caches(self):
        """Clear memory caches to prevent memory buildup."""
        self.kv_cache.clear()
        self.knowledge_cache.clear()
        torch.cuda.empty_cache()

    def get_average_inference_time(self):
        """Get average inference time in seconds."""
        if not self.inference_times:
            return 0
        return sum(self.inference_times) / len(self.inference_times)
```

This efficient pipeline implements memory optimization, knowledge caching, and streaming capabilities to provide responsive user interactions even on limited GPU hardware.

## 9. A/B Testing Methodology

### 9.1 Test Configuration Framework

Here's an implementation of the A/B testing framework for comparing system configurations:

```python
class ABTestingFramework:
    """
    Framework for A/B testing different model configurations and parameters.
    """
    def __init__(self, base_config, variant_configs):
        self.base_config = base_config
        self.variant_configs = variant_configs
        self.results = {}

        # Initialize base model
        self.base_model = self._initialize_model(base_config)

        # Initialize variant models
        self.variant_models = {}
        for variant_name, config in variant_configs.items():
            self.variant_models[variant_name] = self._initialize_model(config)

    def _initialize_model(self, config):
        """Initialize a model with the given configuration."""
        # Create model instance based on config
        model = MTGAIAssistant(
            model_path=config.get("model_path", "mistralai/Mixtral-8x7B-v0.1"),
            device_map=config.get("device_map", "auto")
        )

        # Set configuration-specific parameters
        model.classifier.threshold = config.get("classifier_threshold", 0.5)

        if "active_experts" in config:
            # Enable only specified experts
            for expert_type in model.expert_adapters:
                if expert_type not in config["active_experts"]:
                    model.expert_adapters[expert_type] = None

        return model

    def run_test(self, test_queries, metrics=None):
        """
        Run A/B test on a set of test queries.

        Args:
            test_queries: List of query texts to test
            metrics: List of metrics to evaluate (default: ["correctness", "helpfulness", "time"])

        Returns:
            Test results
        """
        if metrics is None:
            metrics = ["correctness", "helpfulness", "time"]

        # Initialize results dictionary
        self.results = {
            "base": {metric: [] for metric in metrics},
        }

        for variant in self.variant_models:
            self.results[variant] = {metric: [] for metric in metrics}

        # Run test for each query
        for query in test_queries:
            # Test base model
            base_results = self._evaluate_model(self.base_model, query, metrics)

            for metric in metrics:
                self.results["base"][metric].append(base_results[metric])

            # Test variant models
            for variant, model in self.variant_models.items():
                variant_results = self._evaluate_model(model, query, metrics)

                for metric in metrics:
                    self.results[variant][metric].append(variant_results[metric])

        # Calculate aggregate statistics
        summary = self._summarize_results(metrics)

        return summary

    def _evaluate_model(self, model, query, metrics):
        """
        Evaluate a model on a single query.

        Args:
            model: Model to evaluate
            query: Query text
            metrics: List of metrics to evaluate

        Returns:
            Dictionary of metric values
        """
        results = {}

        # Measure response time
        start_time = time.time()
        response = model.answer_query(query)
        end_time = time.time()

        # Record time metric
        if "time" in metrics:
            results["time"] = end_time - start_time

        # For correctness and helpfulness, we would need human evaluation
        # Here we simulate with a basic evaluation function
        if "correctness" in metrics:
            results["correctness"] = self._evaluate_correctness(query, response)

        if "helpfulness" in metrics:
            results["helpfulness"] = self._evaluate_helpfulness(query, response)

        return results

    def _evaluate_correctness(self, query, response):
        """
        Evaluate correctness of a response.
        In a real system, this would be done through human evaluation
        or comparison with known ground truth.
        """
        # For demonstration purposes, this is a placeholder
        # In practice, use rule-based validation or human judges
        return random.uniform(0.7, 1.0)

    def _evaluate_helpfulness(self, query, response):
        """
        Evaluate helpfulness of a response.
        In a real system, this would be done through human evaluation.
        """
        # For demonstration purposes, this is a placeholder
        # In practice, use human judges or user feedback
        return random.uniform(0.6, 1.0)

    def _summarize_results(self, metrics):
        """
        Summarize test results.

        Args:
            metrics: List of metrics to summarize

        Returns:
            Summary dictionary
        """
        summary = {}

        for model_name, model_results in self.results.items():
            summary[model_name] = {}

            for metric in metrics:
                values = model_results[metric]

                summary[model_name][metric] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values)
                }

        return summary

    def generate_report(self):
        """Generate a detailed report of test results."""
        if not self.results:
            return "No test results available. Run test first."

        report = "# A/B Testing Results\n\n"

        # Variant summaries
        report += "## Model Variants\n\n"

        report += "### Base Configuration\n"
        report += self._format_config(self.base_config)
        report += "\n\n"

        for variant, config in self.variant_configs.items():
            report += f"### Variant: {variant}\n"
            report += self._format_config(config)
            report += "\n\n"

        # Performance comparison
        report += "## Performance Comparison\n\n"

        # Time comparison
        if "time" in next(iter(self.results.values())):
            report += "### Response Time (seconds)\n\n"

            time_data = {
                model: results["time"]["mean"]
                for model, results in self.results.items()
            }

            report += "| Model | Mean Time | Speedup vs Base |\n"
            report += "|-------|----------:|----------------:|\n"

            base_time = time_data["base"]

            for model, time in sorted(time_data.items(), key=lambda x: x[1]):
                speedup = base_time / time if model != "base" else 1.0
                report += f"| {model} | {time:.3f}s | {speedup:.2f}x |\n"

            report += "\n\n"

        # Correctness comparison
        if "correctness" in next(iter(self.results.values())):
            report += "### Correctness Score\n\n"

            correctness_data = {
                model: results["correctness"]["mean"]
                for model, results in self.results.items()
            }

            report += "| Model | Correctness | % Change vs Base |\n"
            report += "|-------|------------:|------------------:|\n"

            base_correctness = correctness_data["base"]

            for model, score in sorted(correctness_data.items(), key=lambda x: x[1], reverse=True):
                change = ((score / base_correctness) - 1) * 100 if model != "base" else 0.0
                report += f"| {model} | {score:.2f} | {change:+.2f}% |\n"

            report += "\n\n"

        # Helpfulness comparison
        if "helpfulness" in next(iter(self.results.values())):
            report += "### Helpfulness Score\n\n"

            helpfulness_data = {
                model: results["helpfulness"]["mean"]
                for model, results in self.results.items()
            }

            report += "| Model | Helpfulness | % Change vs Base |\n"
            report += "|-------|------------:|------------------:|\n"

            base_helpfulness = helpfulness_data["base"]

            for model, score in sorted(helpfulness_data.items(), key=lambda x: x[1], reverse=True):
                change = ((score / base_helpfulness) - 1) * 100 if model != "base" else 0.0
                report += f"| {model} | {score:.2f} | {change:+.2f}% |\n"

        return report

    def _format_config(self, config):
        """Format configuration details for reporting."""
        result = ""

        for key, value in config.items():
            if isinstance(value, list):
                result += f"- **{key}**: {', '.join(value)}\n"
            else:
                result += f"- **{key}**: {value}\n"

        return result
```

This framework allows testing different configurations to identify optimal model parameters through systematic comparison.

### 9.2 Test Case Generation

Here's a system for generating comprehensive test cases:

```python
class MTGTestCaseGenerator:
    """
    Generates diverse test cases for evaluating the MTG AI Assistant.
    """
    def __init__(self):
        self.test_categories = {
            "rules": self._generate_rules_questions,
            "gameplay": self._generate_gameplay_questions,
            "deck_building": self._generate_deck_questions,
            "strategy": self._generate_strategy_questions,
            "meta": self._generate_meta_questions
        }

    def generate_test_suite(self, num_per_category=10):
        """
        Generate a complete test suite with questions from all categories.

        Args:
            num_per_category: Number of questions to generate per category

        Returns:
            List of test cases
        """
        test_suite = []

        for category, generator in self.test_categories.items():
            category_tests = generator(num_per_category)

            # Add category label to each test
            for test in category_tests:
                test["category"] = category

            test_suite.extend(category_tests)

        return test_suite

    def _generate_rules_questions(self, count):
        """Generate rules-related test questions."""
        templates = [
            "What happens when {card1} and {card2} interact?",
            "If I have {card} on the battlefield, can I {action}?",
            "How does {mechanic} work with {card}?",
            "Can I cast {card} during {phase}?",
            "If my opponent controls {card}, what happens when I {action}?",
            "Does {card1} trigger when {card2} enters the battlefield?",
            "What's the correct timing for activating {card}'s ability?",
            "Is {action} legal according to the rules?",
            "Can {card} target {permanent_type} with {characteristic}?",
            "How does damage assignment work when {card} blocks multiple creatures?"
        ]

        # Sample cards, mechanics, and actions
        cards = [
            "Lightning Bolt", "Counterspell", "Wrath of God", "Thoughtseize",
            "Birds of Paradise", "Dark Confidant", "Cryptic Command", "Tarmogoyf",
            "Snapcaster Mage", "Path to Exile", "Force of Will", "Brainstorm"
        ]

        mechanics = [
            "cascade", "delve", "flashback", "landfall", "prowess", "storm",
            "devotion", "convoke", "morph", "persist", "undying", "infect"
        ]

        actions = [
            "sacrifice a creature", "counter a spell", "exile a card from my graveyard",
            "cast an instant", "activate an ability", "declare attackers",
            "respond to a trigger", "play a land", "tutor for a card"
        ]

        phases = [
            "my upkeep", "my draw step", "my first main phase", "combat",
            "my second main phase", "my end step", "my opponent's turn"
        ]

        permanent_types = ["creature", "artifact", "enchantment", "planeswalker", "land"]

        characteristics = [
            "flying", "hexproof", "deathtouch", "indestructible",
            "protection from white", "defender", "haste", "vigilance"
        ]

        test_cases = []

        for _ in range(count):
            template = random.choice(templates)

            # Fill in template placeholders
            question = template

            if "{card}" in question:
                question = question.replace("{card}", random.choice(cards))

            if "{card1}" in question:
                question = question.replace("{card1}", random.choice(cards))

            if "{card2}" in question:
                question = question.replace("{card2}", random.choice(cards))

            if "{mechanic}" in question:
                question = question.replace("{mechanic}", random.choice(mechanics))

            if "{action}" in question:
                question = question.replace("{action}", random.choice(actions))

            if "{phase}" in question:
                question = question.replace("{phase}", random.choice(phases))

            if "{permanent_type}" in question:
                question = question.replace("{permanent_type}", random.choice(permanent_types))

            if "{characteristic}" in question:
                question = question.replace("{characteristic}", random.choice(characteristics))

            test_cases.append({
                "query": question,
                "type": "rules"
            })

        return test_cases

    def _generate_gameplay_questions(self, count):
        """Generate gameplay-related test questions."""
        templates = [
            "I have {card1} and {card2} in hand with {mana} available. What's the best play?",
            "My opponent has {opponent_board}. Should I attack with {my_creatures}?",
            "I'm at {life} life facing {opponent_board}. What's my best line of play?",
            "My opponent played {card}. How should I respond?",
            "I have {cards_in_hand} in hand and {mana} available. Should I mulligan?",
            "My opponent attacked with {attacking_creatures}. How should I block?",
            "I'm playing against {deck_archetype}. What should I be careful about?",
            "Should I use {removal} on {target} now or wait?",
            "I have {permanent} on the battlefield. When is the best time to sacrifice it?",
            "Is it worth using {card} to protect my {creature} from {opponent_action}?"
        ]

        # Generate test cases similar to rules questions
        # Simplified implementation for brevity
        return [{"query": f"Gameplay question {i}", "type": "gameplay"} for i in range(count)]

    def _generate_deck_questions(self, count):
        """Generate deck building-related test questions."""
        # Simplified implementation for brevity
        return [{"query": f"Deck building question {i}", "type": "deck_building"} for i in range(count)]

    def _generate_strategy_questions(self, count):
        """Generate strategy-related test questions."""
        # Simplified implementation for brevity
        return [{"query": f"Strategy question {i}", "type": "strategy"} for i in range(count)]

    def _generate_meta_questions(self, count):
        """Generate meta-related test questions."""
        # Simplified implementation for brevity
        return [{"query": f"Meta question {i}", "type": "meta"} for i in range(count)]
```

This generator creates diverse test cases across multiple MTG knowledge domains to enable comprehensive evaluation.

### 9.3 Experiment Configuration Example

Here's an example of how to set up A/B testing experiments:

```python
def configure_ab_testing_experiments():
    """Set up A/B testing experiments with various configurations."""
    # Base configuration
    base_config = {
        "model_path": "mistralai/Mixtral-8x7B-v0.1",
        "device_map": "auto",
        "classifier_threshold": 0.5,
        "active_experts": ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
        "cross_expert_attention": True,
        "knowledge_retrieval": "hybrid"
    }

    # Variant configurations
    variant_configs = {
        # Test with different expert subsets
        "minimal_experts": {
            "model_path": "mistralai/Mixtral-8x7B-v0.1",
            "device_map": "auto",
            "classifier_threshold": 0.5,
            "active_experts": ["REASON", "EXPLAIN"],  # Only 2 core experts
            "cross_expert_attention": True,
            "knowledge_retrieval": "hybrid"
        },

        # Test with different classifier thresholds
        "high_confidence": {
            "model_path": "mistralai/Mixtral-8x7B-v0.1",
            "device_map": "auto",
            "classifier_threshold": 0.8,  # Higher threshold for expert activation
            "active_experts": ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
            "cross_expert_attention": True,
            "knowledge_retrieval": "hybrid"
        },

        # Test without cross-expert attention
        "no_cross_attention": {
            "model_path": "mistralai/Mixtral-8x7B-v0.1",
            "device_map": "auto",
            "classifier_threshold": 0.5,
            "active_experts": ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
            "cross_expert_attention": False,  # Disable cross-expert attention
            "knowledge_retrieval": "hybrid"
        },

        # Test with KG-only knowledge retrieval
        "kg_only": {
            "model_path": "mistralai/Mixtral-8x7B-v0.1",
            "device_map": "auto",
            "classifier_threshold": 0.5,
            "active_experts": ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
            "cross_expert_attention": True,
            "knowledge_retrieval": "kg"  # KG only
        },

        # Test with RAG-only knowledge retrieval
        "rag_only": {
            "model_path": "mistralai/Mixtral-8x7B-v0.1",
            "device_map": "auto",
            "classifier_threshold": 0.5,
            "active_experts": ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"],
            "cross_expert_attention": True,
            "knowledge_retrieval": "rag"  # RAG only
        }
    }

    # Set up test framework
    test_framework = ABTestingFramework(base_config, variant_configs)

    # Generate test cases
    test_generator = MTGTestCaseGenerator()
    test_cases = test_generator.generate_test_suite(num_per_category=5)

    # Extract just the queries
    test_queries = [case["query"] for case in test_cases]

    return test_framework, test_queries
```

This configuration enables systematic comparison of different system aspects to identify optimal settings.

## Conclusion

This technical implementation guide provides ML engineers with a comprehensive blueprint for building the MTG AI Reasoning Assistant on dual 16GB GPUs. By following the detailed memory allocation strategy, QLoRA fine-tuning approach, expert specialization techniques, and efficient inference pipeline, engineers can implement a high-performance system within the specified hardware constraints.

The transaction-based MoE routing system with specialized expert personas provides a novel approach to complex reasoning tasks while maintaining memory efficiency. The hybrid KG/RAG integration ensures accurate knowledge retrieval, and the cross-expert attention mechanism facilitates collaboration between specialized experts.

Key technical innovations include:

- Byte-level memory optimization that precisely distributes model components across dual GPUs
- Expert disagreement resolution through weighted confidence scoring and explicit reconciliation
- Dynamic knowledge selection that intelligently chooses between KG and RAG based on query patterns
- Efficient inference pipeline with caching and streaming capabilities for low-latency responses

By implementing the provided code components and following the A/B testing methodology, engineering teams can deploy, evaluate, and continuously improve the MTG AI Assistant to provide users with accurate, insightful, and helpful responses across all aspects of Magic: The Gathering.
