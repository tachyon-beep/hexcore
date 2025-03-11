# Technical Specification: OODA Analysis & Quality Assurance Experts for MTG AI Assistant

## 1. Overview

This high-level design specifies two specialized experts for our MTG AI Assistant's Mixture-of-Experts architecture: the OODA Analysis Expert and the Quality Assurance Expert. These experts will enhance system performance through structured analysis and quality control without directly interfacing with users.

## 2. OODA Analysis Expert

### 2.1 Purpose

The OODA Analysis Expert transforms complex Magic: The Gathering game states into systematic analytical assessments following the Observe-Orient-Decide-Act framework. This expert provides structured evaluation that informs the reasoning of user-facing experts.

### 2.2 Functional Requirements

- Process complete game state data and associated queries
- Analyze game situations through the OODA framework lens
- Generate clearly structured analyses with distinct sections for each OODA component
- Format output consistently for downstream consumption by other experts
- Focus exclusively on analysis without generating user-directed explanations

### 2.3 Training Approach

The OODA Expert requires specialized training on:

- Diverse MTG game states spanning different formats, phases, and complexity levels
- Corresponding OODA analyses demonstrating systematic evaluation
- Examples ranging from basic board states to complex stack interactions and timing issues
- Approximately 100-150 carefully crafted training examples

### 2.4 Integration Points

The OODA Expert will:

- Be activated first in the processing pipeline for any query involving game state analysis
- Pass its structured output to subsequent experts as contextual information
- Never directly interface with users or generate final responses
- Operate as a "thinking engine" for the entire system

### 2.5 Example Input/Output

**Input:**

```json
{
  "query": "What's my best play?",
  "game_state": {
    "active_player": "Player1",
    "stack": [
      {
        "name": "Lightning Bolt",
        "controller": "Player2",
        "targets": [{ "name": "Llanowar Elves", "controller": "Player1" }]
      }
    ],
    "players": [
      {
        "name": "Player1",
        "hand": [
          "Giant Growth",
          "Grizzly Bears",
          "Forest",
          "Rampant Growth",
          "Naturalize"
        ],
        "battlefield": [
          { "name": "Forest", "status": { "tapped": false } },
          { "name": "Forest", "status": { "tapped": true } },
          { "name": "Llanowar Elves", "status": { "summoning_sick": true } }
        ]
      }
    ]
  }
}
```

**Output:**

```
### Observe: (Assess Current Game State)
- Stack: Lightning Bolt (Player2) targeting Llanowar Elves (Player1).
- Battlefield:
  - Player1 has:
    - Llanowar Elves (1/1, summoning sick)
    - 1 untapped Forest ({G} available)
    - 1 tapped Forest
  - Player2 has aggressive pressure.
- Hand (Player1): Giant Growth, Grizzly Bears, Rampant Growth, Naturalize, Forest.
- Life Totals: Player1 (20), Player2 (17).

### Orient: (Contextualize and Understand Implications)
- Llanowar Elves represents critical mana acceleration. Losing it significantly slows Player1's strategy.
- Lightning Bolt (3 damage) is lethal to Elves (only 1 toughness currently).
- Giant Growth can save Elves if cast in response, boosting toughness temporarily to 4.
- Losing Llanowar Elves gives Player2 a tempo advantage; saving Elves mitigates early aggression.
- Naturalize has no valid targets currently.
- Rampant Growth requires {1}{G} which isn't available with only one untapped Forest.

### Decide: (Evaluate Options and Select Optimal Action)
Available Choices:
1. Do nothing: Llanowar Elves dies, slowing future mana availability.
2. Cast Giant Growth (optimal): Elves survive; retains mana ramp and prevents Player2 from gaining tempo advantage.

Decision Criteria:
- Preserve future board development (tempo advantage).
- Minimize resource loss.
- Maintain early board presence.

### Act: (Execute Decision)
- Tap untapped Forest for {G}.
- Cast Giant Growth targeting Llanowar Elves, responding directly to Lightning Bolt.

Expected Outcome:
- Llanowar Elves survives with 3 damage marked and remains on battlefield.
- Player1 retains mana acceleration, maintaining strategic parity or regaining tempo in future turns.
```

## 3. Quality Assurance Expert

### 3.1 Purpose

The Quality Assurance Expert evaluates generated responses for accuracy, completeness, consistency, clarity, and strategic soundness before they reach users. It operates both during response generation (in-band) and as a separate validation step (out-of-band).

### 3.2 Functional Requirements

- Evaluate responses against explicit quality criteria for MTG content
- Identify rules inaccuracies, strategic misunderstandings, and logical inconsistencies
- Provide both in-band quality signals during generation and out-of-band review
- Suggest improvements or corrections when quality issues are detected
- Optimize for both quality and response time

### 3.3 Quality Dimensions

The QA Expert will assess responses across multiple dimensions:

1. **Rules Accuracy**: Correctness of rule interpretations and card interactions
2. **Strategic Soundness**: Validity of strategic assessments and recommendations
3. **Completeness**: Coverage of all relevant aspects of the query
4. **Clarity**: Comprehensibility and appropriate explanation level
5. **Consistency**: Internal coherence without contradictions

### 3.4 Training Approach

The QA Expert requires specialized training on:

- Paired examples of MTG questions and responses with quality annotations
- Examples of both high-quality responses and flawed responses with corrections
- Explicit feedback on each quality dimension
- Approximately 200-250 training examples with comprehensive annotations

### 3.5 Example Input/Output

**Input:**

```
QUERY: What's my best play with Lightning Bolt targeting my Llanowar Elves?

RESPONSE: You should let your Llanowar Elves die to the Lightning Bolt. Then use your Forest to cast Rampant Growth to get another land. This maximizes your mana development.
```

**Output:**

```
### Quality Assessment

Evaluation of response to query: "What's my best play with Lightning Bolt targeting my Llanowar Elves?"

1. Rules Accuracy: ❌ Low
   - Incorrectly suggests casting Rampant Growth (cost {1}{G}) with only a single Forest ({G})
   - Fails to mention Giant Growth as a relevant instant-speed option

2. Strategic Soundness: ❌ Low
   - Recommendation to let Elves die contradicts standard MTG strategy of preserving mana acceleration
   - Incorrectly claims this play "maximizes mana development" when it actually decreases available mana

3. Completeness: ❌ Low
   - Doesn't consider all available options in hand
   - Fails to analyze timing considerations with Lightning Bolt on stack

4. Clarity: ✓ Medium
   - Explanation is concise but lacks justification
   - Missing context about card interactions

5. Consistency: ✓ Medium
   - No internal contradictions, but reasoning is flawed

Overall Assessment: Significant issues with rules accuracy and strategic assessment.

REVISED RESPONSE:
Your best play is to cast Giant Growth targeting your Llanowar Elves in response to the Lightning Bolt. This costs {G} from your untapped Forest and will give your Elves +3/+3 until end of turn, making them 4/4. When Lightning Bolt resolves, it will deal 3 damage, which isn't enough to kill your now 4/4 Elves.

This play preserves your mana acceleration, allowing you to tap Llanowar Elves for mana on your next turn once summoning sickness wears off. Keeping Llanowar Elves alive is valuable for faster development and maintaining tempo.

Rampant Growth costs {1}{G}, which you can't cast with only one untapped Forest available. Even if you could cast it, preserving your mana creature is generally more valuable than adding a tapped land to the battlefield.
```

## 4. System Integration

### 4.1 Pipeline Architecture

The following pipeline architecture illustrates how these experts integrate into the system:

```
User Query + Game State →
    Router →
        OODA Analysis (for game state queries) →
            Domain Experts (with in-band QA) →
                Initial Response →
                    Out-of-Band QA Review →
                        Final Response
```

### 4.2 Implementation Example

The following code provides a conceptual implementation of the enhanced pipeline. Note that this is for illustration only and would need to be adapted to the actual codebase.

```python
class EnhancedMTGInferencePipeline:
    """
    Enhanced inference pipeline incorporating OODA Analysis and Quality Assurance experts.

    This pipeline implements a multi-stage processing flow:
    1. OODA Analysis of game states (when applicable)
    2. Primary response generation with in-band quality assurance
    3. Out-of-band quality review and refinement

    Note: This implementation is conceptual and would need adaptation to the actual system.
    """

    def __init__(
        self,
        base_model,
        tokenizer,
        domain_experts,
        ooda_expert,
        qa_expert,
        router,
        cross_attention_mechanism
    ):
        """
        Initialize the enhanced inference pipeline.

        Args:
            base_model: The underlying language model
            tokenizer: Tokenizer for the model
            domain_experts: Dictionary mapping expert types to their adapters
            ooda_expert: The OODA analysis expert adapter
            qa_expert: The quality assurance expert adapter
            router: The expert selection router
            cross_attention_mechanism: Mechanism for expert collaboration
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.domain_experts = domain_experts
        self.ooda_expert = ooda_expert
        self.qa_expert = qa_expert
        self.router = router
        self.cross_attention_mechanism = cross_attention_mechanism

    def generate_response(self, query, game_state=None):
        """
        Generate a response to the user query with enhanced processing.

        Args:
            query: The user's query text
            game_state: Optional game state information (board state JSON)

        Returns:
            Final response after all processing stages
        """
        # Initialize processing state
        processing_state = {
            "query": query,
            "game_state": game_state,
            "ooda_analysis": None,
            "selected_experts": [],
            "expert_outputs": {},
            "initial_response": None,
            "final_response": None
        }

        # Stage 1: OODA Analysis (if applicable)
        if game_state is not None:
            processing_state["ooda_analysis"] = self._generate_ooda_analysis(
                query, game_state)

        # Stage 2: Expert Selection
        processing_state["selected_experts"] = self.router.select_experts(
            query, game_state=game_state)

        # Always include QA expert for in-band quality assurance
        if "QA" not in processing_state["selected_experts"]:
            processing_state["selected_experts"].append("QA")

        # Stage 3: Generate expert outputs with cross-attention
        expert_hidden_states = []
        expert_confidence_scores = {}

        for expert_type in processing_state["selected_experts"]:
            # Generate hidden states for each expert
            if expert_type == "QA":
                expert = self.qa_expert
            elif expert_type == "OODA":
                expert = self.ooda_expert
            else:
                expert = self.domain_experts[expert_type]

            # Apply the appropriate adapter
            self.base_model.set_adapter(expert)

            # Format prompt for this expert
            expert_prompt = self._format_expert_prompt(
                expert_type,
                query,
                game_state,
                processing_state["ooda_analysis"]
            )

            # Generate output
            inputs = self.tokenizer(expert_prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.base_model(
                    **inputs, output_hidden_states=True)

            # Extract hidden states
            expert_hidden_states.append(outputs.hidden_states[-1])

            # Set confidence score (from router or default)
            if expert_type == "QA":
                # Give QA a moderate weight in cross-attention
                expert_confidence_scores[expert_type] = 0.3
            else:
                expert_confidence_scores[expert_type] = (
                    self.router.get_confidence(expert_type) or 1.0)

        # Stage 4: Apply cross-expert attention mechanism
        combined_hidden_states = self.cross_attention_mechanism(
            expert_hidden_states, expert_confidence_scores)

        # Stage 5: Generate initial response from combined hidden states
        initial_response = self._generate_from_hidden_states(
            combined_hidden_states)
        processing_state["initial_response"] = initial_response

        # Stage 6: Out-of-band quality assurance
        final_response = self._apply_quality_assurance(
            initial_response, query, game_state)
        processing_state["final_response"] = final_response

        return final_response

    def _generate_ooda_analysis(self, query, game_state):
        """
        Generate OODA analysis for the given game state.

        Args:
            query: The user's query
            game_state: The game state JSON

        Returns:
            Structured OODA analysis
        """
        # Implementation details omitted for brevity
        pass

    def _apply_quality_assurance(self, response, query, game_state):
        """
        Apply out-of-band quality assurance to the response.

        Args:
            response: The initial response
            query: The user's query
            game_state: The game state JSON

        Returns:
            Final response after quality assurance
        """
        # Implementation details omitted for brevity
        pass
```

## 5. Expected Benefits

Implementing these specialized experts should yield several key benefits:

1. **Enhanced Analytical Depth**: The OODA expert will provide systematic analysis of game states, ensuring all relevant factors are considered before generating responses.

2. **Improved Response Quality**: The QA expert will catch rules errors, strategic misunderstandings, and explanatory gaps before they reach users.

3. **Consistent Reasoning Approach**: The OODA framework provides a consistent structure for analyzing game situations, promoting methodical decision-making.

4. **Reduced Error Rate**: The dual-layer quality assurance approach (in-band and out-of-band) should substantially reduce incorrect or misleading information in responses.

5. **Efficient Resource Utilization**: By separating analytical processing (OODA) from response generation, the system can optimize resources for different query types.

## 6. Example of Non-MTG Application

The OODA expert is designed to apply its structured analytical framework to any decision context, even those outside Magic: The Gathering. For example, if presented with a lunch decision question, the expert would automatically apply its OODA framework:

```reasoning
Okay, so I need to figure out where to go for lunch using the OODA Loop. Let me start by recalling what OODA stands for: Observe, Orient, Decide, Act. Right, it's a decision-making process that involves continuously cycling through these stages. Let me break down each step as it applies to choosing a lunch spot.

First, Observe. I guess this means gathering information about my current situation. What do I know? Well, it's lunchtime, I'm hungry, and I need to decide where to eat. Let me list out the factors: time constraints, budget, dietary preferences, proximity, maybe any cravings I have. Also, what options are around me? Maybe check if there are any new places nearby. Oh, and maybe check the weather—if it's raining, I might not want to walk far. Wait, what's the time? If I have a meeting soon, I might need something quick. Do I have any dietary restrictions today? Like, maybe I want something healthy or I'm avoiding gluten. Also, payment options—do I have cash, or do I need a place that takes cards?

Next, Orient. This is about understanding the context and analyzing the information. So, from the observation phase, I have data points: time, budget, dietary needs, location, weather, etc. I need to process this. Let's say I have 45 minutes for lunch. Budget is around $15. I prefer something healthy, maybe a salad or a protein bowl. I'm near downtown, so there are several options. It's sunny, so walking is okay. Payment by card. Also, maybe considering if any places are crowded during lunch hours—wait times could be a factor. Maybe check online reviews or see which places my friends recommend. Also, any current promotions or discounts? Maybe a loyalty program at a specific restaurant.

Then, Decide. Based on orientation, generate options and choose one. Let's say the options nearby are a salad bar, a sandwich shop, a sushi place, and a taco truck. Considering time, sushi might take longer if there's a line. Taco truck is quick but maybe less healthy. Salad bar fits the healthy preference but might be a bit more expensive. Sandwich shop is within budget and quick. Wait, but I had a sandwich yesterday. Maybe sushi is a good option if the wait isn't too long. Let me check the estimated time via an app. If the sushi place has a 10-minute wait, that's manageable. Otherwise, maybe the salad bar. Need to weigh health vs. time vs. budget.

Finally, Act. Go to the chosen place. But after acting, I should loop back to Observe again to see if the decision was good. If the sushi place was too crowded when I got there, maybe I need to reorient and choose another option quickly. So the OODA Loop is iterative; if the first decision doesn't work, I go through the loop again.

Wait, did I miss anything in the Orient phase? Maybe considering past experiences. If the sushi place has been inconsistent before, maybe that's a risk. Or if the salad bar gives generous portions, that's a plus. Also, transportation—do I need to drive, or is it walkable? If I'm on foot, proximity is key. Also, any dietary restrictions like allergies? If I'm avoiding something, that affects options. Let me make sure all factors are considered.

In the Decide phase, perhaps ranking the options based on the criteria: health, speed, cost, taste. Assign weights to each factor. If health is most important, salad bar is top. If speed, taco truck. If cost, maybe the sandwich shop. So after evaluating, pick the one that scores highest.

Acting would be going there, but also being ready to adapt. If upon arrival the salad bar is closed, then I need to redo the loop quickly: Observe (closed), Orient (what's next best), Decide (next option), Act (go there).

So putting it all together, the OODA Loop helps in making a dynamic decision, considering real-time factors and being flexible to change if needed.
```

```output
### Observe: Gather Real-Time Information
   - Time Constraints: 45 minutes for lunch.
   - Budget: ~$15.
   - Dietary Preferences: Healthy options preferred.
   - Location: Downtown area with walkable options.
   - Weather: Sunny, walking feasible.
   - Nearby Options: Salad bar, sandwich shop, sushi restaurant, taco truck.

### Orient: Analyze Context and Prioritize
   - Health: Salad bar or sushi rank highest.
   - Speed: Taco truck or sandwich shop likely fastest.
   - Cost: All options fit within budget.
   - Past Experiences: Sushi place sometimes has wait times.

### Decide: Evaluate Options and Choose
   - Ranked Options:
     1. Salad Bar: Best for health, moderate speed, within budget.
     2. Sushi Restaurant: Healthy, but potential delay (10-minute wait).
     3. Taco Truck: Quick and affordable, less healthy.
     4. Sandwich Shop: Quick and budget-friendly.
   - Decision: Prioritize health and try the salad bar.

### Act: Execute and Adapt
   - Walk to the salad bar.
   - If closed/unavailable, proceed to sushi restaurant as backup.
```

This example demonstrates how the OODA expert will systematically apply its framework regardless of context, functioning effectively as a specialized "thinking module" within the overall system.

## 7. Conclusion

The integration of specialized OODA Analysis and Quality Assurance experts represents a significant architectural enhancement to our MTG AI Assistant. By implementing this design, we can expect improvements in analytical depth, response quality, and overall system reliability.

The OODA expert provides structured analysis that informs other experts' reasoning, while the QA expert ensures responses meet our quality standards before reaching users. This approach leverages the natural strengths of a Mixture-of-Experts architecture, allowing for specialization while maintaining a coherent user experience.
