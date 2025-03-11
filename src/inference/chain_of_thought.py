"""
Chain-of-Thought Reasoning Implementation for MTG AI Reasoning Assistant.

This module implements step-by-step reasoning for rule explanations and logical
analysis, particularly suited for breaking down complex game state scenarios and
rule interactions in Magic: The Gathering.
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple

from src.inference.base_reasoning import BaseReasoning

logger = logging.getLogger(__name__)


class ChainOfThoughtReasoning(BaseReasoning):
    """
    Implements Chain-of-Thought reasoning for rule explanations.

    This reasoning mode breaks down complex problems into intermediate steps,
    making it ideal for analyzing game states and rules interactions.
    """

    def __init__(self):
        """Initialize the Chain-of-Thought reasoning implementation."""
        super().__init__("chain_of_thought")

        # Template for step-by-step reasoning
        self.reasoning_template = """
To analyze this MTG rules question, I'll think through it step by step:

Step 1: Identify the relevant cards and abilities.
{step_1_content}

Step 2: Determine the applicable rules.
{step_2_content}

Step 3: Apply the rules to this specific situation.
{step_3_content}

Step 4: Consider any special cases or exceptions.
{step_4_content}

Step 5: Arrive at the correct ruling.
{step_5_content}

Therefore, the answer is: {conclusion}
"""

        # Template for rule verification
        self.verification_template = """
Let me verify my reasoning:

Does my explanation correctly identify all relevant cards and abilities? {verification_cards}

Are all applicable rules cited correctly? {verification_rules}

Have I properly applied the rules to this situation? {verification_application}

Have I considered all relevant special cases or exceptions? {verification_exceptions}

Is my conclusion consistent with my reasoning? {verification_conclusion}

Based on this verification: {verification_result}
"""

    def apply(
        self,
        query: str,
        inputs: Dict[str, Any],
        knowledge_context: Dict[str, Any],
        reasoning_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply CoT reasoning to the inputs.

        Args:
            query: User query
            inputs: Tokenized inputs for the model
            knowledge_context: Retrieved knowledge context
            reasoning_config: Optional configuration parameters

        Returns:
            Modified inputs with CoT reasoning structure applied
        """
        # Initialize configuration with defaults or overrides
        config = {"max_steps": 5, "verify_steps": True, "rule_grounding": True}

        if reasoning_config:
            config.update(reasoning_config)

        logger.debug(f"Applying CoT reasoning with config: {config}")

        # Extract any prompt from the inputs
        prompt = inputs.get("prompt", "")

        # Generate steps content
        steps_content = self._generate_steps_content(
            query,
            knowledge_context,
            max_steps=config["max_steps"],
            rule_grounding=config["rule_grounding"],
        )

        # Create verification if enabled
        verification_content = ""
        if config["verify_steps"]:
            verification_content = self._generate_verification(
                steps_content, knowledge_context
            )

        # Assemble full reasoning content
        reasoning_elements = steps_content.copy()
        if verification_content:
            reasoning_elements.update(verification_content)

        # Format the overall reasoning content
        reasoning_content = self._format_reasoning_content(
            reasoning_elements, include_verification=config["verify_steps"]
        )

        # Enhance the prompt with the reasoning content
        enhanced_prompt = self.enhance_prompt(
            prompt, {"reasoning_content": reasoning_content}
        )

        # Return updated inputs
        updated_inputs = inputs.copy()
        updated_inputs["prompt"] = enhanced_prompt

        return updated_inputs

    def _generate_steps_content(
        self,
        query: str,
        knowledge_context: Dict[str, Any],
        max_steps: int = 5,
        rule_grounding: bool = True,
    ) -> Dict[str, str]:
        """
        Generate content for each reasoning step.

        Args:
            query: User query
            knowledge_context: Retrieved knowledge context
            max_steps: Maximum number of steps
            rule_grounding: Whether to ground reasoning in specific rules

        Returns:
            Dictionary with content for each step
        """
        # This is a placeholder implementation.
        # In a real implementation, this would use language model completion
        # or retrieve template text based on query type.

        # Extract relevant knowledge for each step
        card_knowledge = self._extract_card_knowledge(knowledge_context)
        rule_knowledge = self._extract_rule_knowledge(knowledge_context)

        # Generate template placeholders
        step_contents = {}

        # Step 1: Identify relevant cards and abilities
        step_contents["step_1_content"] = self._generate_step_1(query, card_knowledge)

        # Step 2: Determine applicable rules
        step_contents["step_2_content"] = self._generate_step_2(
            query, rule_knowledge, rule_grounding
        )

        # Step 3: Apply rules to the situation
        step_contents["step_3_content"] = self._generate_step_3(
            query, card_knowledge, rule_knowledge
        )

        # Step 4: Consider special cases or exceptions
        step_contents["step_4_content"] = self._generate_step_4(query, rule_knowledge)

        # Step 5: Arrive at the correct ruling
        step_contents["step_5_content"] = self._generate_step_5(
            query, card_knowledge, rule_knowledge
        )

        # Generate conclusion
        step_contents["conclusion"] = self._generate_conclusion(query, step_contents)

        return step_contents

    def _generate_verification(
        self, step_contents: Dict[str, str], knowledge_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate verification of reasoning steps.

        Args:
            step_contents: Contents of reasoning steps
            knowledge_context: Retrieved knowledge context

        Returns:
            Dictionary with verification elements
        """
        # This is a placeholder implementation.
        # In a real implementation, this would use an LLM to
        # verify the reasoning in the step contents.

        rule_knowledge = self._extract_rule_knowledge(knowledge_context)

        verification = {}

        # Verify cards and abilities
        verification["verification_cards"] = self._verify_cards(
            step_contents["step_1_content"]
        )

        # Verify rules
        verification["verification_rules"] = self._verify_rules(
            step_contents["step_2_content"], rule_knowledge
        )

        # Verify rule application
        verification["verification_application"] = self._verify_application(
            step_contents["step_3_content"], step_contents["step_2_content"]
        )

        # Verify special cases
        verification["verification_exceptions"] = self._verify_exceptions(
            step_contents["step_4_content"]
        )

        # Verify conclusion
        verification["verification_conclusion"] = self._verify_conclusion(
            step_contents["step_5_content"], step_contents["conclusion"]
        )

        # Overall verification result
        verification["verification_result"] = self._determine_verification_result(
            verification
        )

        return verification

    def _format_reasoning_content(
        self, reasoning_elements: Dict[str, str], include_verification: bool = True
    ) -> str:
        """
        Format the reasoning content with the template.

        Args:
            reasoning_elements: Dictionary of reasoning elements
            include_verification: Whether to include verification

        Returns:
            Formatted reasoning content
        """
        # Format step-by-step reasoning
        steps_text = self.reasoning_template.format(**reasoning_elements)

        # Add verification if enabled
        if include_verification and "verification_result" in reasoning_elements:
            verification_text = self.verification_template.format(**reasoning_elements)
            return f"{steps_text}\n\n{verification_text}"

        return steps_text

    def verify_reasoning(
        self, reasoning_text: str, rules_knowledge: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Verify the reasoning for logical consistency and rules accuracy.

        Args:
            reasoning_text: Generated reasoning text
            rules_knowledge: Retrieved rules knowledge

        Returns:
            Dictionary with:
            - is_valid: Boolean indicating validity
            - issues: List of identified issues
            - suggestions: List of suggested corrections
        """
        # This is a placeholder implementation.
        # In a real implementation, this would use an LLM to
        # analyze the reasoning for consistency and accuracy.

        # Parse the reasoning text to extract steps
        steps = self._parse_reasoning_steps(reasoning_text)

        # Initialize verification result
        verification_result = {"is_valid": True, "issues": [], "suggestions": []}

        # Check for rule citations
        rule_citations = self._extract_rule_citations(reasoning_text)
        if not rule_citations:
            verification_result["is_valid"] = False
            verification_result["issues"].append("No specific rule citations found")
            verification_result["suggestions"].append(
                "Include specific rule numbers from the Comprehensive Rules"
            )

        # Check for logical flow between steps
        if not self._check_logical_flow(steps):
            verification_result["is_valid"] = False
            verification_result["issues"].append("Logical inconsistency between steps")
            verification_result["suggestions"].append(
                "Ensure conclusion follows from premises in each step"
            )

        # Check if rules are being applied correctly
        correct_application = self._check_rule_application(steps, rules_knowledge)
        if not correct_application["is_correct"]:
            verification_result["is_valid"] = False
            verification_result["issues"].append(
                f"Incorrect rule application: {correct_application['issue']}"
            )
            verification_result["suggestions"].append(correct_application["suggestion"])

        return verification_result

    def ground_in_rules(
        self, reasoning_step: str, rules_knowledge: List[Dict[str, Any]]
    ) -> str:
        """
        Ground reasoning in specific MTG rules.

        Args:
            reasoning_step: A reasoning step to ground in rules
            rules_knowledge: Retrieved rules knowledge

        Returns:
            Rule-grounded reasoning step
        """
        # This is a placeholder implementation.
        # In a real implementation, this would use an LLM or knowledge base
        # to enhance reasoning with specific rule references.

        # Check if step already contains rule references
        if self._has_rule_references(reasoning_step):
            return reasoning_step

        # Find relevant rules for this step
        relevant_rules = self._find_relevant_rules(reasoning_step, rules_knowledge)

        # If no relevant rules found, return original step
        if not relevant_rules:
            return reasoning_step

        # Format rule references
        rule_references = "\n\nRelevant rules:\n" + "\n".join(
            [
                f"- {rule['id']}: {rule['text']}"
                for rule in relevant_rules[:3]  # Limit to top 3
            ]
        )

        # Append rule references to the step
        grounded_step = reasoning_step + rule_references

        return grounded_step

    # Helper methods for step generation

    def _generate_step_1(self, query: str, card_knowledge: List[Dict[str, Any]]) -> str:
        """Generate step 1: Identify relevant cards and abilities."""
        # This would typically be done by an LLM
        # Placeholder implementation that extracts card names from the query
        card_names = self._extract_card_names(query)

        if not card_names and card_knowledge:
            # Fallback to cards in knowledge context
            card_names = [
                card.get("name", "Unknown Card") for card in card_knowledge[:3]
            ]

        if not card_names:
            return "No specific cards mentioned in the query. This appears to be a general rules question."

        # Format step content
        content = (
            f"The relevant cards in this scenario are: {', '.join(card_names)}.\n\n"
        )

        # Add abilities if available in card knowledge
        for card_name in card_names:
            matching_cards = [
                c
                for c in card_knowledge
                if c.get("name", "").lower() == card_name.lower()
            ]
            if matching_cards:
                card = matching_cards[0]
                if "text" in card:
                    content += f"{card_name}'s text: {card['text']}\n\n"
                if "abilities" in card:
                    content += (
                        f"{card_name}'s abilities: {', '.join(card['abilities'])}\n\n"
                    )

        return content

    def _generate_step_2(
        self, query: str, rule_knowledge: List[Dict[str, Any]], rule_grounding: bool
    ) -> str:
        """Generate step 2: Determine applicable rules."""
        # This would typically be done by an LLM
        # Placeholder implementation

        # Try to identify key rule concepts in the query
        key_concepts = self._extract_rule_concepts(query)

        if not key_concepts:
            return "Based on the query, we need to apply general game rules about card interactions."

        content = f"Based on the query, the applicable rules involve: {', '.join(key_concepts)}.\n\n"

        # Add specific rules if rule grounding is enabled
        if rule_grounding and rule_knowledge:
            content += "Specific rules that apply:\n\n"

            for concept in key_concepts:
                relevant_rules = [
                    r
                    for r in rule_knowledge
                    if concept.lower() in r.get("text", "").lower()
                ]
                if relevant_rules:
                    rule = relevant_rules[0]  # Take the first matching rule
                    content += f"- Rule {rule.get('id', 'Unknown')}: {rule.get('text', '')}\n\n"

        return content

    def _generate_step_3(
        self,
        query: str,
        card_knowledge: List[Dict[str, Any]],
        rule_knowledge: List[Dict[str, Any]],
    ) -> str:
        """Generate step 3: Apply rules to the specific situation."""
        # This would typically be done by an LLM
        # Placeholder implementation

        # This step combines information from cards and rules to apply to the scenario
        action_verbs = self._extract_action_verbs(query)

        if not action_verbs:
            return "To apply the rules to this situation, we need to consider how the cards interact according to their text and the game rules."

        content = "Applying the rules to this situation:\n\n"

        for verb in action_verbs:
            content += (
                f"When a player {verb}, the following happens according to the rules:\n"
            )

            # Find rules related to this action
            relevant_rules = [
                r for r in rule_knowledge if verb.lower() in r.get("text", "").lower()
            ]

            if relevant_rules:
                rule = relevant_rules[0]
                content += f"- {rule.get('text', '')}\n\n"
            else:
                content += f"- Standard game procedure applies.\n\n"

        return content

    def _generate_step_4(self, query: str, rule_knowledge: List[Dict[str, Any]]) -> str:
        """Generate step 4: Consider special cases or exceptions."""
        # This would typically be done by an LLM
        # Placeholder implementation

        # Look for keywords that might indicate special cases
        special_keywords = ["except", "unless", "however", "otherwise", "special"]
        has_special_case = any(keyword in query.lower() for keyword in special_keywords)

        if not has_special_case:
            # Check rule knowledge for exceptions
            exceptions = [
                r
                for r in rule_knowledge
                if any(kw in r.get("text", "").lower() for kw in special_keywords)
            ]

            if not exceptions:
                return "There don't appear to be any special cases or exceptions relevant to this scenario."

            # Use the first exception found
            exception = exceptions[0]
            return f"One special case to consider is: {exception.get('text', '')}"

        # If special case indicators are in the query
        return "This scenario involves a special case or exception to the normal rules. The specific exception needs to be determined based on the card interactions."

    def _generate_step_5(
        self,
        query: str,
        card_knowledge: List[Dict[str, Any]],
        rule_knowledge: List[Dict[str, Any]],
    ) -> str:
        """Generate step 5: Arrive at the correct ruling."""
        # This would typically be done by an LLM
        # Placeholder implementation

        # For the placeholder, we'll just prepare a template
        return "Based on the cards involved, the applicable rules, and consideration of any special cases, the correct ruling is that the interaction resolves according to the standard rules of Magic: The Gathering."

    def _generate_conclusion(self, query: str, step_contents: Dict[str, str]) -> str:
        """Generate the final conclusion."""
        # This would typically be done by an LLM
        # Placeholder implementation

        # For the placeholder, we'll just use a generic conclusion
        return "the interaction follows the rules as described above."

    # Helper methods for verification

    def _verify_cards(self, step_1_content: str) -> str:
        """Verify card identification is complete and accurate."""
        # Placeholder implementation
        if "no specific cards" in step_1_content.lower():
            return "Yes, this is a general rules question without specific cards."

        if (
            "relevant cards" in step_1_content.lower()
            and "text" in step_1_content.lower()
        ):
            return "Yes, all relevant cards and their abilities have been identified."

        return "Partially - the explanation identifies some cards but may be missing relevant abilities."

    def _verify_rules(
        self, step_2_content: str, rule_knowledge: List[Dict[str, Any]]
    ) -> str:
        """Verify rule citations are accurate."""
        # Placeholder implementation
        if (
            "specific rules" in step_2_content.lower()
            and "rule" in step_2_content.lower()
        ):
            return "Yes, specific rules are correctly cited."

        return "Partially - some rules are mentioned but more specific citations would be helpful."

    def _verify_application(self, step_3_content: str, step_2_content: str) -> str:
        """Verify rules are correctly applied."""
        # Placeholder implementation
        if (
            "applying the rules" in step_3_content.lower()
            and "according to the rules" in step_3_content.lower()
        ):
            return "Yes, the rules are properly applied to the situation."

        return "Partially - the application seems logical but could have more explicit connections to rule text."

    def _verify_exceptions(self, step_4_content: str) -> str:
        """Verify special cases and exceptions are considered."""
        # Placeholder implementation
        if (
            "special case" in step_4_content.lower()
            or "exception" in step_4_content.lower()
        ):
            return "Yes, relevant special cases or exceptions have been considered."

        if "don't appear to be any special cases" in step_4_content.lower():
            return "Yes, and correctly identifies that no special exceptions apply."

        return "Partially - some consideration is given but might be incomplete."

    def _verify_conclusion(self, step_5_content: str, conclusion: str) -> str:
        """Verify conclusion is consistent with reasoning."""
        # Placeholder implementation
        if "correct ruling" in step_5_content.lower() and len(conclusion) > 20:
            return "Yes, the conclusion follows logically from the reasoning."

        return "Partially - the conclusion is present but could be more clearly derived from the previous steps."

    def _determine_verification_result(self, verification: Dict[str, str]) -> str:
        """Determine overall verification result."""
        # Placeholder implementation
        if all(
            "yes" in v.lower()
            for v in verification.values()
            if v != "verification_result"
        ):
            return "My reasoning is correct and complete."

        if any(
            "no" in v.lower()
            for v in verification.values()
            if v != "verification_result"
        ):
            return "My reasoning has significant issues that need correction."

        return "My reasoning is generally correct but could be improved in some areas."

    # Utility methods

    def _extract_card_names(self, text: str) -> List[str]:
        """Extract potential card names from text."""
        # Very simplified implementation
        # In a real system, this would use NER or a card database

        # Look for card names between quotes or in capital case words
        quoted_names = re.findall(r'"([^"]+)"', text)
        capital_words = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)

        # Combine and deduplicate
        all_potential_names = quoted_names + capital_words
        return list(set(all_potential_names))

    def _extract_rule_concepts(self, text: str) -> List[str]:
        """Extract key rule concepts from text."""
        # Very simplified implementation
        # In a real system, this would use NER or a rules taxonomy

        # Common MTG rule concepts
        rule_keywords = [
            "priority",
            "stack",
            "combat",
            "targeting",
            "casting",
            "activated ability",
            "triggered ability",
            "state-based action",
            "replacement effect",
            "continuous effect",
            "layers",
            "mana ability",
            "tapping",
            "untapping",
            "damage",
            "counters",
            "sacrifice",
            "destroy",
            "exile",
            "discard",
            "draw",
            "shuffle",
            "scry",
            "protection",
            "trample",
            "first strike",
            "deathtouch",
            "lifelink",
            "hexproof",
            "indestructible",
            "flash",
            "haste",
        ]

        # Find matches in text
        matches = []
        for keyword in rule_keywords:
            if keyword.lower() in text.lower():
                matches.append(keyword)

        return matches

    def _extract_action_verbs(self, text: str) -> List[str]:
        """Extract MTG action verbs from text."""
        # Very simplified implementation
        # In a real system, this would use dependency parsing

        # Common MTG action verbs
        action_verbs = [
            "cast",
            "activate",
            "trigger",
            "target",
            "attack",
            "block",
            "sacrifice",
            "destroy",
            "exile",
            "discard",
            "draw",
            "shuffle",
            "tap",
            "untap",
            "pay",
            "gain",
            "lose",
            "add",
            "remove",
            "counter",
        ]

        # Find matches in text
        matches = []
        for verb in action_verbs:
            if verb.lower() in text.lower():
                matches.append(verb)

        return matches

    def _extract_card_knowledge(
        self, knowledge_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract card knowledge from context."""
        # Extract card data if available
        if not knowledge_context:
            return []

        if isinstance(knowledge_context, dict) and "kg_data" in knowledge_context:
            kg_data = knowledge_context["kg_data"]
            if isinstance(kg_data, dict) and kg_data.get("type") == "card_data":
                return kg_data.get("data", [])

        return []

    def _extract_rule_knowledge(
        self, knowledge_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract rule knowledge from context."""
        # Extract rule data if available
        if not knowledge_context:
            return []

        if isinstance(knowledge_context, dict) and "kg_data" in knowledge_context:
            kg_data = knowledge_context["kg_data"]
            if isinstance(kg_data, dict) and kg_data.get("type") == "rule_data":
                return kg_data.get("data", [])

        return []

    def _has_rule_references(self, text: str) -> bool:
        """Check if text already contains rule references."""
        # Look for patterns like "Rule 123.4" or "CR 123.4"
        return bool(re.search(r"(?:Rule|CR)\s+\d+\.\d+", text))

    def _find_relevant_rules(
        self, text: str, rules_knowledge: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find rules relevant to the text."""
        # Simplified implementation - would use semantic search or NLP in real system

        if not rules_knowledge:
            return []

        # Extract key terms from text
        # For simplicity, just using space-separated words
        key_terms = [
            word.strip().lower() for word in text.split() if len(word.strip()) > 3
        ]

        # Find rules that contain any of the key terms
        relevant_rules = []
        for rule in rules_knowledge:
            rule_text = rule.get("text", "").lower()
            if any(term in rule_text for term in key_terms):
                relevant_rules.append(rule)

        return relevant_rules

    def _parse_reasoning_steps(self, reasoning_text: str) -> Dict[str, str]:
        """Parse reasoning text into steps."""
        # Simplified implementation - would use more robust parsing in real system

        steps = {}

        # Look for step patterns
        step_matches = re.findall(
            r"Step (\d+):(.*?)(?=Step \d+:|Therefore|$)", reasoning_text, re.DOTALL
        )

        for step_num, content in step_matches:
            steps[f"step_{step_num}"] = content.strip()

        # Get conclusion
        conclusion_match = re.search(
            r"Therefore, the answer is:(.*?)(?=$)", reasoning_text, re.DOTALL
        )
        if conclusion_match:
            steps["conclusion"] = conclusion_match.group(1).strip()

        return steps

    def _extract_rule_citations(self, text: str) -> List[str]:
        """Extract rule citations from text."""
        # Look for patterns like "Rule 123.4" or "CR 123.4"
        return re.findall(r"(?:Rule|CR)\s+(\d+\.\d+[a-z]?)", text)

    def _check_logical_flow(self, steps: Dict[str, str]) -> bool:
        """Check if there is logical flow between steps."""
        # Simplified implementation - would use NLI in real system

        # Simply check if steps build on each other (very simplified)
        step_keys = [key for key in steps.keys() if key.startswith("step_")]
        step_keys.sort()

        if len(step_keys) < 2:
            return True  # Not enough steps to check flow

        # Check if later steps reference earlier ones
        for i in range(1, len(step_keys)):
            current_step = steps[step_keys[i]].lower()
            earlier_steps_text = []

            for j in range(i):
                earlier_steps_text.extend(
                    re.findall(r"\b\w{4,}\b", steps[step_keys[j]].lower())
                )

            # Check if any significant words from earlier steps appear in current step
            significant_words = [word for word in earlier_steps_text if len(word) > 4]
            if not any(word in current_step for word in significant_words):
                return False  # No reference to earlier content

        return True

    def _check_rule_application(
        self, steps: Dict[str, str], rules_knowledge: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check if rules are applied correctly."""
        # Simplified implementation - would use NLI in real system

        result = {"is_correct": True, "issue": "", "suggestion": ""}

        # This is just a placeholder - a real implementation would be much more complex
        # and would actually check rule semantics

        # For now, just check if rule citations exist
        rule_citations = []
        for step_content in steps.values():
            rule_citations.extend(self._extract_rule_citations(step_content))

        if not rule_citations:
            result["is_correct"] = False
            result["issue"] = "No rule citations found in reasoning"
            result["suggestion"] = (
                "Include specific rule citations from the Comprehensive Rules"
            )

        return result
