# Architecture Decision Record (ADR)

## ADR 001: Expert Routing Robustness in MTG AI Reasoning Assistant

### Status

Accepted

### Context

The MTG AI Reasoning Assistant relies on a transaction classifier (a TinyLLM-based router) to direct user queries to the appropriate expert(s) among five specialized modules: REASON, EXPLAIN, TEACH, PREDICT, and RETROSPECT. This routing is a critical component, as it determines which expertise will be applied to the user's query.

The current design uses a single-pass classification approach where the classifier makes a definitive decision about which expert(s) should handle the query. If this classification is incorrect, the entire response may be inappropriate or suboptimal, regardless of the quality of the individual experts. For example, if a rules question is incorrectly routed to the TEACH expert instead of the REASON expert, the response may focus on educational aspects rather than precise rule interpretation.

Our initial testing with similar classification systems suggests we can achieve approximately 85-90% accuracy in routing. While this is reasonably high, it means that 10-15% of queries may be suboptimally routed, creating a notable user experience issue and potentially undermining trust in the system.

### Decision

We will implement a robust expert routing system with the following features:

1. **Confidence Thresholds and Multi-Expert Activation**:

   - The transaction classifier will return confidence scores for each expert type
   - When confidence for the top expert is below a threshold (initially set at 0.7), we will activate multiple experts
   - The number of activated experts will scale inversely with confidence (lower confidence = more experts)

2. **Ensemble Classification**:

   - For critical routing decisions, we will implement an ensemble of classification methods
   - This will include the primary TinyLLM classifier, a keyword-based heuristic classifier, and a rules-based classifier
   - The final routing decision will be made by weighing outputs from all classifiers

3. **Adaptive Routing with Feedback**:

   - We will implement a feedback mechanism to track routing success
   - The system will learn from successful and unsuccessful routings to adjust thresholds and weights
   - User feedback signals (explicit or implicit) will be incorporated into this learning process

4. **Expert Voting System**:

   - When multiple experts are activated due to uncertainty, each will provide a relevance score for their output
   - Experts can "vote" on whether another expert might be better suited to address the query
   - This creates a secondary, content-based routing layer after the initial classification

5. **Graceful Degradation**:
   - In cases of high routing uncertainty, the system will default to a conservative approach of using the REASON and EXPLAIN experts in combination
   - This combination provides both technical accuracy and clear explanation, covering most MTG query needs

### Consequences

#### Positive

- Reduced impact of classification errors on response quality
- More robust handling of edge cases and ambiguous queries
- Improved user experience through more consistent quality
- System that improves over time through feedback mechanisms
- Higher confidence in critical responses (rules interpretations, etc.)

#### Negative

- Increased computational cost from activating multiple experts
- Additional complexity in implementation and testing
- Potential latency increase when multiple experts are activated
- More complex debugging and error analysis
- Additional training data requirements for the feedback mechanisms

#### Neutral

- Changes to the memory allocation strategy may be needed to support efficient multi-expert activation
- Evaluation metrics will need to include routing accuracy and adaptation
- User interface may need to indicate when there is uncertainty in routing

### Implementation Notes

- Initial confidence thresholds will be determined through validation testing during Phase 2
- The ensemble classifier should be optimized for minimal latency impact
- We will need to create a dataset of "boundary cases" that are difficult to classify to test robustness

### Related Decisions

- Memory optimization strategy (ADR-002) will need to account for potential multi-expert activation
- Knowledge integration system (pending ADR) should support providing relevant information to multiple experts simultaneously
