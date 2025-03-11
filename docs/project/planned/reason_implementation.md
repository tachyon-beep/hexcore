## Implementation Steps

1. **You already have the core components:**

   - `BaseReasoning` abstract class defining the interface
   - `ChainOfThoughtReasoning` implementation of the CoT approach
   - `ReasoningModeSelector` to determine when to use CoT
   - `ReasoningFactory` to instantiate the reasoning implementations

2. **Integration with the Pipeline:**

To integrate CoT reasoning into the pipeline, you'll need to add code that:

1. Selects the appropriate reasoning method using `ReasoningModeSelector`
2. Creates the reasoning implementation using `ReasoningFactory`
3. Applies the reasoning to enhance the prompt

Here's a complete implementation example:

```python
def enhance_prompt_with_reasoning(
    query: str,
    original_prompt: str,
    knowledge_context: Dict[str, Any],
    expert_type: str,
    confidence_score: float = 0.8
) -> str:
    """
    Enhance a prompt with appropriate reasoning structure.

    Args:
        query: User query
        original_prompt: Original model prompt
        knowledge_context: Retrieved knowledge context
        expert_type: Expert type (REASON, EXPLAIN, etc.)
        confidence_score: Confidence score from transaction classifier

    Returns:
        Enhanced prompt with reasoning structure
    """
    # 1. Select reasoning mode
    selector = ReasoningModeSelector()
    mode, config = selector.select_reasoning_mode(
        query=query,
        expert_type=expert_type,
        confidence_score=confidence_score
    )

    # 2. Create reasoning implementation
    reasoning = ReasoningFactory.create_reasoning(mode)

    # 3. Apply reasoning to enhance prompt
    enhanced_inputs = reasoning.apply(
        query=query,
        inputs={"prompt": original_prompt},
        knowledge_context=knowledge_context,
        reasoning_config=config
    )

    # Return enhanced prompt
    return enhanced_inputs["prompt"]
```

## Integration with Your Pipeline

To integrate this with your `EnhancedMTGInferencePipeline`, you would modify the `_create_expert_prompt` method to include reasoning enhancement:

```python
def _create_expert_prompt(
    self,
    query: str,
    expert_type: str,
    knowledge: Dict[str, Any]
) -> str:
    """Create expert-specific prompt with reasoning enhancement."""
    # Get base prompt for this expert type
    base_prompt = self._get_expert_prompt_template(expert_type)

    # Get confidence score (use 0.85 as default if not available)
    confidence_score = getattr(self, "_expert_confidence", {}).get(expert_type, 0.85)

    # Enhance the prompt with reasoning
    enhanced_prompt = enhance_prompt_with_reasoning(
        query=query,
        original_prompt=base_prompt,
        knowledge_context=knowledge,
        expert_type=expert_type,
        confidence_score=confidence_score
    )

    # Format the enhanced prompt with query and knowledge
    formatted_prompt = self._format_prompt(enhanced_prompt, query, knowledge)

    return formatted_prompt
```

## Example Usage

With these changes in place, your system would automatically:

1. Analyze each query using `ReasoningModeSelector`
2. Determine if CoT is the best reasoning approach for the query
3. Apply CoT structure to the prompt before sending to the model
4. The model would then respond with a structured, step-by-step chain of thought reasoning

For MTG rules questions (which is the core of your data), it would typically use the CoT approach since the `ReasoningModeSelector` heavily weights rules-based questions toward CoT reasoning.

Your training data with 37 examples focusing on MTG rule interactions is well-suited for teaching the model to reason in a CoT manner, as these examples naturally follow a step-by-step logical reasoning process.

The beauty of your architecture is that different expert types (REASON, EXPLAIN, TEACH, etc.) can all benefit from CoT reasoning, but each applies it according to their specialty - focusing on different aspects of the reasoning process and formatting the output appropriately for their role.

## When to Apply CoT Reasoning

### During Training

Having the Chain of Thought reasoning structure applied during training would provide several benefits:

1. **Consistent Format**: The model would learn to generate responses in the same structured format it will use during inference
2. **Guided Learning**: The step-by-step structure could help the model learn logical reasoning patterns more efficiently
3. **Reduced Inference Customization**: Less prompt engineering would be needed at inference time since the model already knows the expected format

However, if applied during training, you'd need to:

- Ensure your training data already includes CoT-structured reasoning
- Or pre-process your training data to add CoT structure before fine-tuning

### During Inference Only

Applying CoT only at inference time also has several advantages:

1. **Flexibility**: You can adjust the reasoning approach without retraining the model
2. **Separation of Concerns**: The model focuses on learning MTG rules and reasoning, while the structure is imposed externally
3. **System Evolution**: You can improve the reasoning structure over time without retraining

## Recommendation for Your System

For your implementation, I recommend the following staged approach:

1. **Start with Inference-Only Implementation**:

   - Implement the CoT reasoning as an inference-time enhancement initially
   - This gives you flexibility to tune and adjust the CoT structure
   - You can evaluate different CoT formats without retraining

2. **Analyze Performance**:

   - Evaluate how well the model adapts to the reasoning structure
   - Identify any misalignments between how the model was trained and how you're prompting it

3. **Consider Hybrid Approach for Future Training**:
   - If you do additional fine-tuning, consider including some CoT-structured examples
   - This would reduce the "prompt tax" at inference time
   - The model would learn to automatically produce step-by-step reasoning

This approach gives you the best of both worlds - immediate implementation with flexibility to adjust, and the option to internalize the reasoning structure in future training rounds.

Since your current implementation with `BaseReasoning` and `ChainOfThoughtReasoning` is designed as an inference-time enhancement (it manipulates prompts rather than training data), I recommend starting there. You can always incorporate CoT into your training data format for future fine-tuning iterations if the results are promising.

Include reasoning in the example data:

{
"input": "If I cast Mycosynth Lattice and March of the Machines in the same turn, what happens to the board state?",
"output": "When Mycosynth Lattice and March of the Machines are both on the battlefield, the following occurs:\n\n1. Mycosynth Lattice makes all permanents (including lands, enchantments, planeswalkers, etc.) artifacts in addition to their other types.\n\n2. March of the Machines turns all noncreature artifacts into artifact creatures with power and toughness equal to their mana value (converted mana cost).\n\nKey interactions:\n- Lands: Since lands have a mana value of 0, they become 0/0 creatures. State-based actions immediately send them to the graveyard due to 0 toughness. This destroys all lands on the battlefield.\n- Other noncreature permanents: Enchantments, planeswalkers, and other noncreature artifacts become creatures with power and toughness equal to their mana value.\n - Mycosynth Lattice (mana value 6) becomes a 6/6 artifact creature.\n - March of the Machines (mana value 3) becomes a 3/3 artifact enchantment creature.\n - Planeswalkers retain their loyalty abilities but also gain power/toughness equal to their mana value.\n- Existing creatures remain unaffected by March of the Machines (they're still creatures and artifacts, but March only affects noncreature artifacts).\n\nThe end result is a board state with no lands (as they've all been destroyed) and all other noncreature permanents turned into artifact creatures with their power and toughness equal to their mana value.",
"reasoning": "Okay, so the question is about what happens when you cast Mycosynth Lattice and March of the Machines in the same turn. Let me try to work through this step by step.\n\nFirst, I need to remember what each of these cards does. Mycosynth Lattice is an artifact that makes all permanents on the battlefield and all cards in all zones become artifacts in addition to their other types. So everything is an artifact now. Then March of the Machines is an enchantment that says each noncreature artifact becomes an artifact creature with power and toughness equal to its mana value. But wait, March of the Machines affects noncreature artifacts. So normally, artifacts that aren't creatures become creatures. But with Mycosynth Lattice, all permanents are artifacts. So lands, enchantments, planeswalkers, etc., are all artifacts now. So under March of the Machines, those noncreature artifacts would become creatures. But wait, creatures are already creatures. So if a permanent is already a creature, it's unaffected, right? Because March only affects noncreature artifacts. But with Mycosynth Lattice, even creatures are artifacts, but they're still creatures. So March would turn all noncreature artifacts (which are now everything else) into creatures.\n\nBut here's the thing: when you cast both in the same turn, the order in which they resolve matters. Let's say you cast Mycosynth Lattice first. Then when it resolves, all permanents become artifacts. Then you cast March of the Machines. When March resolves, it sees all noncreature artifacts, which are all the permanents that weren't already creatures. But wait, creatures are still creatures, but they are also artifacts. So March would turn all noncreature artifacts into creatures. But since Mycosynth Lattice is already making everything artifacts, any permanent that isn't a creature would become a creature. But if a permanent was already a creature (like a creature card), it would remain a creature. However, if a permanent was a land, enchantment, planeswalker, etc., they are now artifacts and noncreature, so March turns them into creatures.\n\nBut here's a potential issue: when Mycosynth Lattice is on the battlefield, and March of the Machines is cast, when March enters, it's also an artifact because of Mycosynth. So March itself is an enchantment artifact. Then, March's effect applies to itself. Wait, March is a noncreature artifact, so it would become a creature. But March is an enchantment, so under Mycosynth, it's also an artifact. So March would become a creature with power and toughness equal to its mana value. March of the Machines has a mana value of 3, so it would be a 3/3. But when a permanent becomes a creature, if it's a noncreature permanent, like an enchantment, it becomes a creature in addition to its other types. So March is now an enchantment artifact creature. But creatures can't be enchantments, right? Wait, no, in Magic, a permanent can have multiple types. So March would be an enchantment, artifact, and creature. But creatures can have other types. So that's okay.\n\nHowever, when March becomes a creature, does that affect its ability? No, because its ability is static and already in effect. So March's ability is active as long as it's on the battlefield, regardless of its own status. So even though March is now a creature, its effect is still applying.\n\nNow, the critical point here is that when both are on the battlefield, all noncreature artifacts are creatures. But because Mycosynth Lattice makes everything an artifact, that includes lands, which are now artifacts. So lands would become creatures. But lands are noncreature artifacts, so March turns them into 0/0 creatures (since lands have a mana value of 0). Wait, mana value is the converted mana cost. Lands typically have a mana value of 0 because they don't have a mana cost. So any land would become a 0/0 creature, which would immediately die due to having 0 toughness. But if you have a way to boost their toughness, they can survive. But in this case, when Mycosynth and March are both out, all lands are artifacts, so March makes them 0/0 creatures, which die. So all lands would be put into the graveyard as a state-based action.\n\nBut wait, Mycosynth Lattice is also an artifact. So when March enters, Mycosynth Lattice is an artifact. Is Mycosynth a noncreature artifact? Yes, because it's an artifact but not a creature. So March would turn Mycosynth into a creature. Mycosynth's mana value is 6, so it becomes a 6/6 artifact creature. Similarly, any other noncreature artifacts would become creatures. But if you have other permanents, like enchantments or planeswalkers, they are also artifacts now, so they'd become creatures with power and toughness equal to their mana values.\n\nSo the key takeaway is that all lands die, and everything else becomes a creature. This can lead to a lot of stuff dying, especially lands, which would cripple both players' mana bases."
}
