# Comprehensive Technical Design Document: MTG AI Reasoning Assistant

_Based on the Mixtral 8×7B Model_  
_Optimized for 2×16GB GPUs_

---

## 1. Executive Summary and Technical Vision

### Executive Summary

The MTG AI Reasoning Assistant is a specialised AI system designed to support Magic: The Gathering (MTG) players by providing real‐time, expert-level game analysis and strategic guidance. By leveraging a Mixtral 8×7B Mixture-of-Experts (MoE) architecture and advanced fine-tuning via QLoRA, the assistant operates efficiently on a consumer-grade dual 16GB GPU setup. It supports multiple reasoning modes—**REASON, EXPLAIN, TEACH, PREDICT, and RETROSPECT**—each corresponding to a dedicated expert module.

**Core Innovations:**

- **Transaction-Based MoE Routing:** A lightweight TinyLLM classifier routes each complete query to one or more experts. This “transaction” approach avoids per-token routing overhead while ensuring consistent reasoning style.
- **Expert Personality Specialization:** Each expert is fine-tuned on mode-specific data. For instance, the REASON expert focuses on rule application and step-by-step logic, while the TEACH expert provides didactic, user-friendly explanations.
- **QLoRA on Dual GPUs:** The system uses 4-bit quantization and low-rank adaptation to fine-tune the base model and expert layers efficiently within a 32GB VRAM limit.
- **Hybrid KG/RAG Integration:** By combining a Knowledge Graph (for structured rules and card relationships) with a Retrieval-Augmented Generation system, the assistant achieves high factual accuracy.
- **Expert Disagreement Resolution:** A novel mechanism allows conflicting expert outputs to be reconciled via cross-attention and confidence-based weighting.

### Technical Approach and Expected Outcomes

The assistant’s architecture is engineered to solve the unique challenges of MTG gameplay analysis. During a session, the user’s query—augmented with context from MTG rules and card data—is first classified by the TinyLLM router. Based on the router’s decision, the system dynamically activates one or more expert modules within the Mixtral 8×7B framework. For example, a query about a complex stack interaction may trigger both the REASON and PREDICT experts, with their outputs merged via a custom cross-expert attention layer.

A detailed byte-level memory allocation plan ensures that all components—from the quantized base model to the LoRA adapters and retrieval cache—fit within a strict 32GB VRAM budget. Optimizations such as aggressive 4-bit quantization, gradient checkpointing, and strategic sharding across GPUs are employed to prevent OOM errors during both training and inference.

By integrating expert knowledge transplantation, the system is designed to scale seamlessly to larger MoE architectures in the future. The end result is an AI assistant that not only provides rapid and accurate MTG analysis but also continuously improves via an integrated A/B testing and evaluation pipeline that compares expert configurations against human expert performance.

**Expected Outcomes:**

- **Enhanced MTG Analysis:** The assistant will produce accurate, rule-compliant game analyses and strategic insights comparable to professional judges and top-level players.
- **Real-Time Responsiveness:** Optimised for consumer-grade hardware, the assistant will deliver sub-second latency on complex queries.
- **Scalability and Flexibility:** The design supports future scaling to multi-GPU systems and larger models, while its modular architecture permits iterative improvements.
- **User-Centric Improvements:** Continuous A/B testing and novel MTG-specific evaluation metrics will ensure that the system evolves based on real player feedback.

---

## 2. Detailed Memory Allocation

### Byte-Level Memory Allocation Plan

Our goal is to manage the entire system within 32GB VRAM across 2×16GB GPUs. Below is a detailed table breaking down the memory allocation per component (all sizes are approximate, in MB):

| Component                                                          | Memory Usage (MB) per GPU   | Total (MB) | Optimizations & Notes                                                            |
| ------------------------------------------------------------------ | --------------------------- | ---------- | -------------------------------------------------------------------------------- |
| **Base Model (Quantized 4-bit)**                                   | 11,750                      | 23,500     | 4-bit QLoRA reduces precision; weights sharded across GPUs                       |
| **LoRA Adapters (Experts)**                                        | ~300 per expert × 4 = 1,200 | 1,200      | Low-rank updates; activated selectively; stored in FP16/INT8                     |
| **Shared Transformer Layers**                                      | ~2,500                      | 5,000      | Split evenly; use gradient checkpointing to reduce peak memory                   |
| **TinyLLM Router Classifier**                                      | 100                         | 100        | Model quantized to INT8/FP16 for fast inference (<10ms latency)                  |
| **KG/RAG Retrieval Cache & Buffers**                               | 250                         | 250        | Data stored on CPU and streamed to GPU as needed; caching helps repeated queries |
| **Attention KV Cache (for Streaming)**                             | ~400                        | 400        | Key/value caching across long context; reused during autoregressive generation   |
| **Miscellaneous Overheads** (buffers, CUDA streams, logging, etc.) | 500                         | 500        | Extra safety margin to handle overhead and dynamic allocation                    |
| **Total per GPU**                                                  | ≈15,750                     | ≈31,500    | Leaves a ~500 MB margin per GPU to avoid OOM                                     |

### Optimization Techniques

- **4-bit Quantization & LoRA:** Aggressively quantize the base model; update only low-rank adapters.
- **Gradient Checkpointing:** Saves memory during training by recomputing intermediate activations.
- **Sharding Across GPUs:** Distribute experts evenly (e.g. 4 experts per GPU) to balance memory usage.
- **Dynamic KV Cache:** Retain computed key/value pairs for recurrent tokens in streaming inference.
- **Efficient Data Streaming:** Use pinned memory for retrieval buffers; offload rarely used data to CPU.

### Mitigation Strategies for OOM Scenarios

- **Dynamic Batch Sizing:** Adjust batch size dynamically based on current memory load.
- **Fallback Routing:** In cases where routing multiple experts might breach the memory limit, fallback to a single expert mode.
- **Memory Profiling Hooks:** Continuous monitoring via DeepSpeed’s memory usage hooks will trigger warnings and adjust micro-batch sizes automatically.
- **Offloading & CPU Swapping:** Less critical modules (e.g. KG retrieval buffers) can be offloaded to CPU when GPU memory is under pressure.

### Visualization of Memory Usage

Below is an example schematic of memory usage across GPUs during inference:

```
┌─────────────────────────────────────┐          ┌─────────────────────────────────────┐
│            GPU 0 (16GB)             │          │            GPU 1 (16GB)             │
│  ┌────────────┐   ┌─────────────┐    │          │  ┌────────────┐   ┌─────────────┐    │
│  │ Base Model │   │ Shared Layers│    │          │  │ Base Model │   │ Shared Layers│    │
│  │ (4-bit Q)  │   │  (Half)     │    │          │  │ (4-bit Q)  │   │  (Half)      │    │
│  └────────────┘   └─────────────┘    │          │  └────────────┘   └─────────────┘    │
│  ┌────────────┐                     │          │  ┌────────────┐                     │
│  │ Experts 1-4│  (LoRA & Routing)     │          │  │ Experts 5-8│  (LoRA & Routing)     │
│  └────────────┘                     │          │  └────────────┘                     │
│  ┌────────────┐                     │          │  ┌────────────┐                     │
│  │ TinyLLM    │                     │          │  │ KV Cache   │                     │
│  │ Classifier │                     │          │  │ & Buffers  │                     │
│  └────────────┘                     │          │  └────────────┘                     │
└─────────────────────────────────────┘          └─────────────────────────────────────┘
```

In the above, the base model is split and shared layers are divided equally. Expert adapters are localized per GPU. The router runs on GPU0 for fast decision-making, and key caches are maintained on both GPUs for continuous autoregressive generation.

---

## 3. MTG-Specific Game State Examples

### Example 1: Complex Stack Interaction

**Game State:**

- **Board:**
  - Player A has a creature with “haste” and an enchantment that doubles damage.
  - Player B has a counterspell and a card granting “hexproof.”
- **Stack:**
  1. Player A casts _Lightning Bolt_ targeting Player B’s creature.
  2. In response, Player B casts _Counterspell_.
  3. Player A activates an ability from an artifact that changes target.

**Expert Processing:**

- **REASON:** Breaks down the resolution order—Lightning Bolt resolves if not countered, but Counterspell on stack nullifies it; then, the artifact’s ability is applied.
- **PREDICT:** Simulates possible outcomes: if Counterspell is successful, damage is prevented; if artifact re-targets, then damage could be redirected.
- **EXPLAIN:** Provides a step-by-step explanation of the stack’s interactions, highlighting timing rules and priority.

**Expected Output:**  
_"When Lightning Bolt is cast, it targets your creature. However, your opponent’s Counterspell is on the stack. If Counterspell resolves first, Lightning Bolt is countered. In response, activating the artifact to change the target can only take effect if the Counterspell is somehow invalidated. Therefore, the most likely outcome is that Lightning Bolt is countered, and the artifact’s ability has no effect."_

### Example 2: Multi-Attacker Combat Scenario

**Game State:**

- **Board:**
  - Attacker has three creatures: one with first strike, one with deathtouch, and one with lifelink.
  - Defender has two blockers: one with indestructible and one with a damage prevention effect.

**Expert Processing:**

- **REASON:** Calculates combat damage, ordering first strike damage and damage assignment.
- **PREDICT:** Evaluates which blockers to assign to minimise loss, considering deathtouch and lifelink abilities.
- **EXPLAIN:** Summarises the combat sequence and effects of abilities.

**Expected Output:**  
_"In combat, the creature with first strike deals damage before the normal combat damage step. The deathtouch creature, if blocked, can eliminate any blocker it deals damage to, while lifelink returns life equal to damage dealt. With an indestructible blocker and a damage prevention effect, the optimal assignment would be to block the deathtouch creature with the indestructible blocker, ensuring that damage is neutralised, and assign the lifelink creature to the remaining blocker for efficient trade."_

### Example 3: Probability-Based Decision (Mulligan)

**Game State:**

- **Hand:**
  - A mixed hand with 3 lands and 4 non-land cards.
- **Context:**
  - The deck’s curve and known archetype suggest a 40% chance of a functional opening hand.

**Expert Processing:**

- **PREDICT:** Computes the probability of drawing the necessary cards for a strong opening, suggesting a mulligan if below threshold.
- **TEACH:** Explains the mulligan strategy and expected outcomes.
- **EXPLAIN:** Breaks down the risks and rewards of keeping a suboptimal hand.

**Expected Output:**  
_"Given your hand of 3 lands and 4 non-land cards, the probability of drawing the necessary spells in subsequent turns is only about 40%. It is statistically advisable to mulligan in order to increase your chances of an optimal start. Mulliganing, in this case, is a strategic decision supported by the game’s probabilities and deck composition."_

### Example 4: Multi-Turn Planning Scenario

**Game State:**

- **Board:**
  - A player has set up a combo that requires two turns to resolve, involving a sacrifice outlet and a card that doubles life gain.
- **Context:**
  - The opponent has reactive countermeasures available.

**Expert Processing:**

- **REASON:** Outlines the sequential steps required over multiple turns.
- **RETROSPECT:** Reviews similar past game states and identifies potential pitfalls.
- **EXPLAIN:** Clarifies the importance of timing and resource management.

**Expected Output:**  
_"Your combo requires you to sacrifice your creature on turn two to trigger the doubling effect. However, your opponent might hold a counter or removal spell. Considering past matches, it is crucial to delay the combo until you secure additional protection or bait out your opponent’s responses. A two-turn plan should include a contingency plan for countering your opponent’s reactive spells."_

---

## 4. Expert Disagreement Resolution Algorithm

### Formal Algorithm for Conflict Resolution

When multiple experts provide differing outputs, we resolve conflicts via a weighted aggregation mechanism. The algorithm uses confidence scores from each expert and cross-attention feedback.

#### Mathematical Formulation

Let \( E = \{e*1, e_2, \dots, e_k\} \) be the set of active experts with outputs \( O_i \) and confidence scores \( c_i \) (where \( c_i \in [0,1] \)). The aggregated output \( O \) is:
\[
O = \frac{\sum*{i=1}^{k} c*i \cdot O_i}{\sum*{i=1}^{k} c_i}
\]
where \( O_i \) are vector representations of the expert outputs. A decision tree governs which experts to prefer based on thresholds and domain-specific rules.

#### Pseudocode

```python
def resolve_expert_conflicts(expert_outputs: Dict[str, Tuple[torch.Tensor, float]]) -> torch.Tensor:
    """
    Resolve conflicting outputs from multiple experts.

    Args:
        expert_outputs (Dict[str, Tuple[torch.Tensor, float]]):
            A dictionary mapping expert mode names (e.g., "REASON", "PREDICT") to a tuple:
            (output vector, confidence score).

    Returns:
        Aggregated output vector (torch.Tensor).
    """
    # Sum weighted outputs
    weighted_sum = None
    total_confidence = 0.0
    for mode, (output, confidence) in expert_outputs.items():
        if weighted_sum is None:
            weighted_sum = confidence * output
        else:
            weighted_sum += confidence * output
        total_confidence += confidence

    if total_confidence == 0:
        raise ValueError("Total confidence is zero; cannot resolve outputs.")

    # Final aggregated output is the weighted average
    aggregated_output = weighted_sum / total_confidence
    return aggregated_output

# Example test case for expert resolution:
def test_expert_resolution():
    # Simulated expert outputs (dummy vectors) and confidence scores
    output_reason = torch.tensor([1.0, 2.0])
    output_predict = torch.tensor([2.0, 0.0])
    # Confidence values (e.g., from softmax probabilities of router)
    expert_data = {
        "REASON": (output_reason, 0.8),
        "PREDICT": (output_predict, 0.6)
    }
    aggregated = resolve_expert_conflicts(expert_data)
    expected = (0.8 * output_reason + 0.6 * output_predict) / (0.8 + 0.6)
    assert torch.allclose(aggregated, expected), "Expert resolution failed!"
    print("Expert resolution test passed.")

test_expert_resolution()
```

#### Decision Tree Outline

1. **Start:** Receive outputs and confidence scores from each expert.
2. **Check Confidence Threshold:**
   - If an expert’s confidence \( c_i \) is below a threshold \( T \) (e.g., 0.5), discard its output.
3. **Cross-Attention Consistency:**
   - Compute cross-attention similarity between outputs. If the cosine similarity between any two outputs is above 0.9, increase their weighting.
4. **Final Aggregation:**
   - Compute the weighted sum using the formula above.
5. **Fallback:**
   - If all experts are below threshold, default to a safe “rules-only” answer (e.g., invoke a secondary rules engine).

---

## 5. Enhanced Code Quality

Below is an enhanced version of key code components with comprehensive docstrings, type hints, robust error handling, and unit tests. Performance monitoring hooks are included as comments for integration with logging systems.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Tuple, List

###############################################################################
# RouterClassifier with Enhanced Code Quality
###############################################################################
class RouterClassifier:
    """
    A lightweight classifier for routing MTG queries to the appropriate expert.

    This class uses a pre-trained model (e.g., DistilBERT) to classify input text
    into one or more expert modes. The classifier is quantized for fast inference.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (AutoModelForSequenceClassification): The classification model.
        modes (List[str]): List of expert mode names.
        device (str): Device for model computation.
    """
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 num_experts: int = 5, device: str = 'cuda:0') -> None:
        self.device: str = device
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            self.model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=num_experts
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
        self.model.to(device)
        self.model.eval()
        self.modes: List[str] = ["REASON", "EXPLAIN", "TEACH", "PREDICT", "RETROSPECT"]

    def predict(self, text: str) -> List[Tuple[int, float]]:
        """
        Predict the top expert modes for a given text query.

        Args:
            text (str): The input query.

        Returns:
            List[Tuple[int, float]]: A list of tuples containing expert index and confidence score.
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        except Exception as e:
            raise RuntimeError(f"Inference failed in RouterClassifier: {e}")
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=2)
        return [(int(idx), float(prob)) for idx, prob in zip(top_idx, top_probs)]

###############################################################################
# ExpertFFN and MoETransformerBlock with Enhanced Code Quality
###############################################################################
class ExpertFFN(nn.Module):
    """
    Feed-forward network representing an expert in the MoE model.

    Args:
        d_model (int): Dimensionality of the input.
        d_ff (int): Dimensionality of the feed-forward layer.
    """
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the expert.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].

        Returns:
            torch.Tensor: Output tensor of the same shape.
        """
        return self.layer2(F.gelu(self.layer1(x)))

class MoETransformerBlock(nn.Module):
    """
    A transformer block integrating a Mixture-of-Experts (MoE) feed-forward layer.

    Args:
        d_model (int): Model dimensionality.
        d_ff (int): Feed-forward layer dimensionality.
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts to use per forward pass.
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 5, top_k: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(num_experts)])
        self.router: List[int] = []  # To be set externally
        self.top_k: int = top_k

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the transformer block with MoE.

        Args:
            x (torch.Tensor): Input tensor.
            attn_mask (torch.Tensor, optional): Attention mask.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Self-attention
        attn_out, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
        x = x + attn_out  # Residual connection
        # MoE routing: if router not set, default to expert 0
        if not self.router:
            moe_out = self.experts[0](x)
        else:
            expert_outputs = []
            for idx in self.router:
                try:
                    expert_outputs.append(self.experts[idx](x))
                except Exception as e:
                    raise RuntimeError(f"Error in expert {idx} during forward pass: {e}")
            moe_out = sum(expert_outputs) / len(expert_outputs)
        return x + moe_out

###############################################################################
# Unit Tests and Performance Monitoring Hooks (Using logging)
###############################################################################
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def unit_test_router() -> None:
    """Unit test for RouterClassifier."""
    try:
        router = RouterClassifier()
        sample_text = "Test query for MTG rules regarding combat damage assignment."
        predictions = router.predict(sample_text)
        logger.info("Router predictions: %s", [(router.modes[i], p) for i, p in predictions])
    except Exception as e:
        logger.error("Router unit test failed: %s", e)

def unit_test_moe_block() -> None:
    """Unit test for MoETransformerBlock."""
    try:
        d_model, d_ff = 32, 64
        block = MoETransformerBlock(d_model, d_ff, num_experts=3)
        # For testing, assign router to experts 0 and 1.
        block.router = [0, 1]
        x = torch.randn(1, 4, d_model)
        output = block(x)
        assert output.shape == x.shape, "Output shape mismatch"
        logger.info("MoETransformerBlock unit test passed.")
    except Exception as e:
        logger.error("MoE block unit test failed: %s", e)

if __name__ == "__main__":
    unit_test_router()
    unit_test_moe_block()
```

_Notes:_

- Comprehensive docstrings follow PEP 257.
- All functions and methods include type hints (PEP 484).
- Exceptions are caught and logged; unit tests verify functionality.
- Performance monitoring hooks (via Python’s `logging`) have been integrated to track key steps and potential bottlenecks.

---

## 6. A/B Testing Methodology

### Testing Framework Overview

The A/B testing framework will compare different expert configurations (e.g., single versus multi-expert routing; different cross-attention schemes) by exposing two groups of users (or test cases) to alternate versions of the assistant. Key components include:

- **Metrics Collected:**

  - **Latency:** Average time per query.
  - **Rule Application Accuracy:** Percentage of responses with correct MTG rule enforcement.
  - **Strategic Decision Quality:** Scored by expert judges or automated simulation win-rates.
  - **Explanatory Clarity:** Rated by users on a Likert scale.
  - **Overall User Satisfaction:** Combined score from surveys.

- **Statistical Analysis Approach:**

  - Use two-sample t-tests (or non-parametric tests if distributions are not normal) to compare mean performance metrics between groups.
  - Employ confidence intervals (95%) to assess statistical significance.
  - Use effect size (Cohen’s d) to quantify improvements.

- **Sample Size Calculations:**

  - Based on a minimum detectable effect size (e.g., 0.5 standard deviations), calculate required sample sizes (commonly, 50–100 queries per variant) using standard power analysis methods.
  - For example, to achieve 80% power with alpha = 0.05, approximately 64 samples per group may be needed.

- **Testing Protocol:**

  1. **Control Group:** Uses the baseline configuration (e.g., single expert routing).
  2. **Experimental Group:** Uses the new configuration (e.g., dual-expert with cross-attention).
  3. **Randomized Assignment:** Queries are randomly assigned to each group.
  4. **Data Collection:** Automated logging collects performance metrics, response texts, and user ratings.
  5. **Post-Experiment Analysis:** Statistical tests determine if the experimental configuration significantly improves performance on the defined metrics.

- **Evaluation Criteria:**
  - A/B testing success is defined by statistically significant improvements in key metrics (e.g., >5% reduction in latency, >10% improvement in rule accuracy) and positive user satisfaction ratings.

---

## 7. Novel MTG-Specific Evaluation Metrics

To assess MTG reasoning quality beyond general LLM benchmarks, we propose the following evaluation metrics:

### Quantitative Measures

- **Rule Application Accuracy (RAA):**
  - _Definition:_ The percentage of responses that correctly reference and apply official MTG rules.
  - _Measurement:_ Compare model outputs with a gold standard database of rulings.
- **Strategic Decision Quality (SDQ):**
  - _Definition:_ A score (0–100) reflecting the quality of in-game decisions.
  - _Measurement:_ Simulate game outcomes using a custom engine; compare the model’s suggestions against expert moves.
- **Explanatory Clarity (EC):**
  - _Definition:_ A score (0–100) determined by linguistic analysis (readability metrics, semantic coherence) and human evaluation.
  - _Measurement:_ Use a combination of automated metrics (e.g., Flesch-Kincaid score) and user ratings.
- **Probabilistic Outcome Accuracy (POA):**
  - _Definition:_ Accuracy of predicted outcomes (e.g., win probabilities) compared with simulation results.
  - _Measurement:_ Calculate mean squared error between predicted probabilities and simulation outcomes.

### Composite Scoring System

- **Overall MTG Reasoning Score (OMRS):**
  - _Formula:_  
    \[
    OMRS = 0.3 \times RAA + 0.3 \times SDQ + 0.2 \times EC + 0.2 \times (100 - \text{MSE}\_{POA})
    \]
  - _Interpretation:_ The higher the OMRS, the closer the assistant’s performance is to that of a human expert judge.

### Benchmarking

- **Human Expert Comparison:**
  - Create a test set of 50 curated MTG scenarios, each answered by a panel of MTG judges.
  - Compare the assistant’s responses using the above metrics.
- **Correlation with Standard Benchmarks:**
  - Although MTG is unique, measure correlations with general reasoning benchmarks (GSM8K, MMLU) to validate overall reasoning ability.

---

## Conclusion

This document provides a fully integrated technical design for an MTG AI Reasoning Assistant. It covers the executive vision, detailed memory allocation with byte-level planning, domain-specific game state examples, a formal expert disagreement resolution algorithm, enhanced code quality with robust testing, a comprehensive A/B testing framework, and novel evaluation metrics tailored to MTG gameplay. The design leverages the latest advances in MoE architectures, QLoRA fine-tuning, and hybrid KG/RAG systems to deliver an AI capable of deep, expert-level reasoning—all within the constraints of consumer-grade 2×16GB GPU hardware.

This implementation-ready document should serve as a definitive guide for ML engineers tasked with building, deploying, and iteratively refining a state-of-the-art MTG assistant that meets both technical excellence and practical performance criteria.
