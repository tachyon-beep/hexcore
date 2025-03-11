"""
Reasoning Mode Selector for MTG AI Reasoning Assistant.

This module implements a selector that determines the most appropriate reasoning
methodology based on query content, assigned expert type, and complexity metrics.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class ReasoningModeSelector:
    """
    Selects appropriate reasoning methodology based on query type and expert.

    This component determines which reasoning method (CoT, MCTS, R1) should be
    applied to a given query based on its content, complexity, and the assigned
    expert type.
    """

    def __init__(self, default_mode: str = "chain_of_thought"):
        """
        Initialize the reasoning mode selector.

        Args:
            default_mode: Default reasoning mode if no specific match is found
        """
        self.default_mode = default_mode

        # Initialize pattern dictionaries for different reasoning modes
        self.cot_patterns = [
            r"how does (.*?) work",
            r"what happens when",
            r"explain the interaction",
            r"step[s]? by step",
            r"explain how",
            r"rule[s]? for (.*?)",
            r"mechanics of",
            r"card text mean[s]?",
            r"interpret (.*?)",
            r"explain the rule",
        ]

        self.mcts_patterns = [
            r"probability",
            r"chance[s]? of",
            r"odds",
            r"random",
            r"likely",
            r"best play",
            r"optimal (move|play|action)",
            r"win percentage",
            r"should i (attack|block|mulligan)",
            r"what are my chances",
        ]

        self.r1_patterns = [
            r"complex (interaction|scenario)",
            r"edge case",
            r"(unusual|rare) (interaction|case)",
            r"tournament rule[s]?",
            r"policy",
            r"corner case",
            r"advanced (rule|interaction)",
            r"comprehensive rule[s]?",
            r"detailed analysis",
            r"multiple (interpretations|viewpoints)",
        ]

        # Expert preference configuration
        self.expert_preferences = {
            "REASON": ["chain_of_thought", "r1_style", "mcts"],
            "EXPLAIN": ["chain_of_thought", "r1_style", "mcts"],
            "TEACH": ["chain_of_thought", "mcts", "r1_style"],
            "PREDICT": ["mcts", "chain_of_thought", "r1_style"],
            "RETROSPECT": ["r1_style", "chain_of_thought", "mcts"],
        }

        # Default configurations for each reasoning mode
        self.default_configs = {
            "chain_of_thought": {
                "max_steps": 5,
                "verify_steps": True,
                "rule_grounding": True,
            },
            "mcts": {
                "simulation_depth": 3,
                "max_sequences": 5,
                "probability_threshold": 0.1,
            },
            "r1_style": {
                "internal_deliberation_tokens": 1024,
                "self_critique": True,
                "alternative_interpretations": 2,
            },
        }

    def select_reasoning_mode(
        self, query: str, expert_type: str, confidence_score: float
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Analyze the query and select the optimal reasoning mode.

        Args:
            query: The user's query text
            expert_type: The selected expert type (REASON, EXPLAIN, etc.)
            confidence_score: Confidence score from transaction classifier

        Returns:
            reasoning_mode: String identifying the selected reasoning method
                ("chain_of_thought", "mcts", "r1_style")
            reasoning_config: Dictionary of configuration parameters for the
                selected reasoning mode
        """
        # Extract features from the query
        features = self.extract_query_features(query)
        logger.debug(f"Extracted features: {features}")

        # Calculate scores for each reasoning mode
        mode_scores = {
            "chain_of_thought": self._calculate_cot_score(
                features, expert_type, confidence_score
            ),
            "mcts": self._calculate_mcts_score(features, expert_type, confidence_score),
            "r1_style": self._calculate_r1_score(
                features, expert_type, confidence_score
            ),
        }
        logger.debug(f"Mode scores: {mode_scores}")

        # Select the mode with the highest score
        selected_mode = max(mode_scores.items(), key=lambda x: x[1])[0]

        # Adjust configuration based on features
        config = self._get_adjusted_config(selected_mode, features, expert_type)

        logger.info(f"Selected reasoning mode: {selected_mode} with config: {config}")
        return selected_mode, config

    def extract_query_features(self, query: str) -> Dict[str, Any]:
        """
        Extract features from a query to guide reasoning selection.

        Args:
            query: The user's query text

        Returns:
            Dictionary of features including:
            - contains_rules_keywords: Boolean
            - contains_probability_terms: Boolean
            - complexity_score: Float (0-1)
            - entity_count: Integer
            - requires_step_by_step: Boolean
        """
        # Normalize query for easier processing
        normalized_query = query.lower().strip()

        # Initialize feature dictionary
        features = {
            "contains_rules_keywords": False,
            "contains_probability_terms": False,
            "contains_complex_scenario_terms": False,
            "complexity_score": 0.0,
            "entity_count": 0,
            "requires_step_by_step": False,
            "cot_pattern_matches": [],
            "mcts_pattern_matches": [],
            "r1_pattern_matches": [],
        }

        # Check for pattern matches
        for pattern in self.cot_patterns:
            matches = re.findall(pattern, normalized_query, re.IGNORECASE)
            if matches:
                features["cot_pattern_matches"].extend(matches)
                features["contains_rules_keywords"] = True

        for pattern in self.mcts_patterns:
            matches = re.findall(pattern, normalized_query, re.IGNORECASE)
            if matches:
                features["mcts_pattern_matches"].extend(matches)
                features["contains_probability_terms"] = True

        for pattern in self.r1_patterns:
            matches = re.findall(pattern, normalized_query, re.IGNORECASE)
            if matches:
                features["r1_pattern_matches"].extend(matches)
                features["contains_complex_scenario_terms"] = True

        # Count MTG entities (cards, keywords, rules)
        # This is a simplistic implementation - would be enhanced in practice
        entity_count = self._count_mtg_entities(normalized_query)
        features["entity_count"] = entity_count

        # Calculate complexity score based on multiple factors
        features["complexity_score"] = self._calculate_complexity_score(
            normalized_query,
            features["entity_count"],
            features["contains_rules_keywords"],
            features["contains_complex_scenario_terms"],
        )

        # Determine if step-by-step reasoning is needed
        features["requires_step_by_step"] = (
            "step by step" in normalized_query
            or "explain how" in normalized_query
            or features["complexity_score"] > 0.6
        )

        return features

    def get_expert_reasoning_preferences(self, expert_type: str) -> List[str]:
        """
        Get ordered list of preferred reasoning modes for expert type.

        Args:
            expert_type: Expert type (REASON, EXPLAIN, etc.)

        Returns:
            List of reasoning modes in preference order
        """
        return self.expert_preferences.get(
            expert_type, ["chain_of_thought", "mcts", "r1_style"]
        )

    def _calculate_cot_score(
        self, features: Dict[str, Any], expert_type: str, confidence_score: float
    ) -> float:
        """Calculate score for Chain-of-Thought reasoning."""
        score = 0.0

        # Base score based on pattern matches
        if features["cot_pattern_matches"]:
            score += 0.5 + min(0.3, len(features["cot_pattern_matches"]) * 0.1)

        # Add score if step-by-step is explicitly needed
        if features["requires_step_by_step"]:
            score += 0.3

        # Add score based on expert type preference
        expert_preferences = self.get_expert_reasoning_preferences(expert_type)
        if "chain_of_thought" in expert_preferences:
            score += 0.3 * (
                1
                - expert_preferences.index("chain_of_thought") / len(expert_preferences)
            )

        # Adjust based on confidence score to weight expert preferences more when confidence is high
        score *= 0.7 + 0.3 * confidence_score

        # Rules keywords are a strong signal for CoT
        if features["contains_rules_keywords"]:
            score += 0.2

        return score

    def _calculate_mcts_score(
        self, features: Dict[str, Any], expert_type: str, confidence_score: float
    ) -> float:
        """Calculate score for MCTS reasoning."""
        score = 0.0

        # Base score based on pattern matches
        if features["mcts_pattern_matches"]:
            score += 0.5 + min(0.3, len(features["mcts_pattern_matches"]) * 0.1)

        # Add score based on expert type preference
        expert_preferences = self.get_expert_reasoning_preferences(expert_type)
        if "mcts" in expert_preferences:
            score += 0.3 * (
                1 - expert_preferences.index("mcts") / len(expert_preferences)
            )

        # Adjust based on confidence score to weight expert preferences more when confidence is high
        score *= 0.7 + 0.3 * confidence_score

        # Probability terms are a strong signal for MCTS
        if features["contains_probability_terms"]:
            score += 0.4

        return score

    def _calculate_r1_score(
        self, features: Dict[str, Any], expert_type: str, confidence_score: float
    ) -> float:
        """Calculate score for R1-style reasoning."""
        score = 0.0

        # Base score based on pattern matches
        if features["r1_pattern_matches"]:
            score += 0.5 + min(0.3, len(features["r1_pattern_matches"]) * 0.1)

        # Add score based on complexity
        score += features["complexity_score"] * 0.4

        # Add score based on expert type preference
        expert_preferences = self.get_expert_reasoning_preferences(expert_type)
        if "r1_style" in expert_preferences:
            score += 0.3 * (
                1 - expert_preferences.index("r1_style") / len(expert_preferences)
            )

        # Adjust based on confidence score to weight expert preferences more when confidence is high
        score *= 0.7 + 0.3 * confidence_score

        # Complex scenario terms are a strong signal for R1
        if features["contains_complex_scenario_terms"]:
            score += 0.4

        # High entity count indicates complex scenario suitable for R1
        if features["entity_count"] > 3:
            score += 0.2

        return score

    def _count_mtg_entities(self, query: str) -> int:
        """
        Count MTG-specific entities in the query text.

        This is a simplified implementation. In practice, this would use a more
        sophisticated approach like named entity recognition on MTG cards and terms.

        Args:
            query: The normalized query text

        Returns:
            Integer count of entities detected
        """
        # Example entity types to count (would be expanded in practice)
        entity_types = [
            # Card types
            r"\bcreature\b",
            r"\bartifact\b",
            r"\bplaneswalker\b",
            r"\binstant\b",
            r"\bsorcery\b",
            r"\benchantment\b",
            r"\bland\b",
            # Game actions
            r"\bcast\b",
            r"\battack\b",
            r"\bblock\b",
            r"\bsacrifice\b",
            r"\bdiscard\b",
            r"\bdraw\b",
            r"\bexile\b",
            r"\bcounter\b",
            # Keywords
            r"\bflying\b",
            r"\bdeathtouch\b",
            r"\btrample\b",
            r"\bhaste\b",
            r"\bdouble strike\b",
            r"\bfirst strike\b",
            r"\bvigilance\b",
            r"\breach\b",
        ]

        entity_count = 0
        for pattern in entity_types:
            entity_count += len(re.findall(pattern, query))

        return entity_count

    def _calculate_complexity_score(
        self,
        query: str,
        entity_count: int,
        has_rules_keywords: bool,
        has_complex_terms: bool,
    ) -> float:
        """
        Calculate complexity score from query features.

        Args:
            query: The normalized query text
            entity_count: Number of MTG entities detected
            has_rules_keywords: Whether query contains rules keywords
            has_complex_terms: Whether query contains complex scenario terms

        Returns:
            Float score between 0 and 1
        """
        # Initialize base score
        score = 0.0

        # Add score based on query length
        score += min(0.3, len(query) / 300)

        # Add score based on entity count
        score += min(0.3, entity_count / 10)

        # Add score for rules keywords
        if has_rules_keywords:
            score += 0.2

        # Add score for complex scenario terms
        if has_complex_terms:
            score += 0.3

        # Add score based on sentence count
        sentence_count = len(re.split(r"[.!?]", query))
        score += min(0.2, sentence_count / 10)

        # Add score if it contains conditional statements
        if re.search(r"\bif\b.*\bthen\b|when|while", query):
            score += 0.2

        # Cap at 1.0
        return min(1.0, score)

    def _get_adjusted_config(
        self, mode: str, features: Dict[str, Any], expert_type: str
    ) -> Dict[str, Any]:
        """
        Get mode configuration adjusted for query features.

        Args:
            mode: Selected reasoning mode
            features: Extracted query features
            expert_type: Expert type

        Returns:
            Configuration dictionary for the reasoning mode
        """
        # Start with default configuration
        config = self.default_configs[mode].copy()

        # Adjust based on expert type and features
        if mode == "chain_of_thought":
            # Adjust max steps based on complexity
            complexity = features["complexity_score"]
            config["max_steps"] = max(3, min(8, int(3 + 5 * complexity)))

            # Toggle verification based on expert type
            config["verify_steps"] = expert_type in ["REASON", "EXPLAIN", "TEACH"]

        elif mode == "mcts":
            # Adjust simulation depth based on complexity
            complexity = features["complexity_score"]
            config["simulation_depth"] = max(2, min(5, int(2 + 3 * complexity)))

            # Adjust probability threshold based on expert type
            if expert_type == "PREDICT":
                config["probability_threshold"] = (
                    0.05  # More precise for prediction expert
                )

        elif mode == "r1_style":
            # Adjust deliberation tokens based on complexity
            complexity = features["complexity_score"]
            config["internal_deliberation_tokens"] = int(768 + 1280 * complexity)

            # Adjust alternative interpretations count based on complexity
            config["alternative_interpretations"] = max(
                1, min(4, int(1 + 3 * complexity))
            )

        return config
