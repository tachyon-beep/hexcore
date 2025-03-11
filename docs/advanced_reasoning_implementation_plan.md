# Advanced Reasoning Implementation

## Overview

This document describes the implementation of three reasoning methodologies for MTG AI Assistant:

- Chain-of-Thought (CoT) reasoning
- Monte Carlo Tree Search (MCTS) reasoning
- R1-style reasoning

## Component Architecture

```
src/inference/
├── base_reasoning.py         # Abstract base class defining the reasoning interface
├── reasoning_selector.py     # Selects optimal reasoning strategy based on query
├── chain_of_thought.py       # Step-by-step reasoning for rules explanations
├── mcts_reasoning.py         # Probabilistic reasoning for game scenarios
├── r1_reasoning.py           # Comprehensive reasoning for complex edge cases
└── reasoning_factory.py      # Factory pattern for creating reasoning instances
```

## Class Relationships

- `BaseReasoning` (ABC): Defines the interface all reasoning implementations must implement
- `ReasoningModeSelector`: Analyzes queries to determine optimal reasoning method
- `ChainOfThoughtReasoning`: Implements step-by-step reasoning (extends BaseReasoning)
- `MCTSReasoning`: Implements probabilistic reasoning (extends BaseReasoning)
- `R1StyleReasoning`: Implements comprehensive reasoning (extends BaseReasoning)
- `ReasoningFactory`: Creates instances of reasoning implementations

## Implementation Details

### BaseReasoning (Abstract Base Class)

- Defines `apply()` method that all implementations must override
- Provides helper methods for prompt enhancement and knowledge integration
- Handles formatting of different knowledge types
- Integrates with knowledge context from KG/RAG

### ReasoningModeSelector

- Uses pattern matching and heuristics to analyze query content
- Considers assigned expert type when selecting reasoning mode
- Generates appropriate configuration for selected reasoning mode
- Dynamically adjusts complexity parameters based on query features

### ChainOfThoughtReasoning

- Implements step-by-step reasoning approach
- Breaks down rule explanation into 5 structured steps
- Supports verification of reasoning steps
- Grounds explanations in specific rule citations
- Template-based approach for consistent reasoning structure

### MCTSReasoning

- Implements probabilistic game state analysis
- Simulates game states and action sequences
- Evaluates probability of different outcomes
- Optimizes decision making based on expected value
- Provides detailed probability analysis

### R1StyleReasoning

- Implements comprehensive, deliberative reasoning
- Supports multiple interpretations of rules scenarios
- Includes explicit self-critique step
- Separates internal reasoning from final solution
- Maximizes reliability for complex edge cases

### ReasoningFactory

- Implements factory pattern for reasoning instantiation
- Maintains registry of available reasoning implementations
- Provides simple interface for creating reasoning instances
- Supports extension with new reasoning implementations

## Integration Points

1. **Knowledge System Integration**:

   - All reasoning implementations accept knowledge context
   - Knowledge formatting methods handle different types (KG, RAG)
   - Reasoning steps incorporate retrieved knowledge

2. **Input/Output Flow**:

   - Input: query, tokenized inputs, knowledge context, configuration
   - Output: enhanced inputs with structured reasoning prompt
   - No direct model calls (decoupled design)

3. **Pipeline Integration**:
   - Reasoning selector can be called from enhanced pipeline
   - Transaction classifier provides expert type input
   - Configuration parameters can be adjusted per query

## Usage Example

```python
# 1. Select reasoning mode
selector = ReasoningModeSelector()
mode, config = selector.select_reasoning_mode(
    query="How does the stack work?",
    expert_type="REASON",
    confidence_score=0.85
)

# 2. Create reasoning implementation
reasoning = ReasoningFactory.create_reasoning(mode)

# 3. Apply reasoning to enhance prompt
enhanced_inputs = reasoning.apply(
    query=query,
    inputs={"prompt": original_prompt},
    knowledge_context=knowledge,
    reasoning_config=config
)

# 4. Use enhanced prompt for generation
enhanced_prompt = enhanced_inputs["prompt"]
```

## Performance Considerations

1. **Memory Optimization**:

   - Configurable reasoning depths to control token usage
   - Lazy loading of knowledge resources
   - Efficient knowledge filtering based on relevance

2. **Extensibility**:
   - New reasoning methods can be added by extending BaseReasoning
   - Reasoning selector patterns can be expanded without changing architecture
   - Configuration parameters allow fine-tuning without code changes

## Testing Strategy

1. Unit tests for each reasoning implementation
2. Integration tests for the reasoning selection process
3. End-to-end tests with sample queries across different categories
4. Memory performance tests for large knowledge contexts
