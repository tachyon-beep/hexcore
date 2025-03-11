"""
Base Reasoning Interface for MTG AI Reasoning Assistant.

This module defines the base interface for all reasoning implementations,
providing a consistent structure and shared utilities for different reasoning
approaches.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class BaseReasoning(ABC):
    """
    Base interface for all reasoning implementations.

    This abstract class defines the contract that all reasoning implementations
    must fulfill, ensuring consistent behavior across different reasoning methods.
    """

    def __init__(self, name: str):
        """
        Initialize the reasoning implementation.

        Args:
            name: Name identifier for this reasoning implementation
        """
        self.name = name
        logger.debug(f"Initialized {name} reasoning implementation")

    @abstractmethod
    def apply(
        self,
        query: str,
        inputs: Dict[str, Any],
        knowledge_context: Dict[str, Any],
        reasoning_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply reasoning method to the inputs.

        Args:
            query: User query
            inputs: Tokenized inputs for the model
            knowledge_context: Retrieved knowledge context
            reasoning_config: Optional configuration parameters

        Returns:
            Modified inputs with reasoning structure applied
        """
        pass

    def enhance_prompt(
        self, original_prompt: str, reasoning_elements: Dict[str, Any]
    ) -> str:
        """
        Enhance the prompt with reasoning structure.

        Args:
            original_prompt: Original model prompt
            reasoning_elements: Dictionary of elements to insert

        Returns:
            Enhanced prompt with reasoning structure
        """
        # Base implementation - can be overridden by subclasses
        # This provides a basic template-based enhancement

        # Create template markers if they don't exist in the original prompt
        if "<reasoning>" not in original_prompt:
            # Identify insertion point - typically after context and query
            # but before generation guidance

            # Simple heuristic: Look for lines with "Query:" or similar
            lines = original_prompt.split("\n")
            insertion_idx = len(lines)

            for i, line in enumerate(lines):
                if line.strip().lower().startswith(("query:", "question:", "prompt:")):
                    # Insert after the query line
                    insertion_idx = i + 1
                    break

            # Insert reasoning template
            enhanced_lines = lines[:insertion_idx]
            enhanced_lines.append("\n<reasoning>\n{reasoning_content}\n</reasoning>\n")
            enhanced_lines.extend(lines[insertion_idx:])

            enhanced_prompt = "\n".join(enhanced_lines)
        else:
            enhanced_prompt = original_prompt

        # Replace template markers with content
        for key, value in reasoning_elements.items():
            placeholder = "{" + key + "}"
            enhanced_prompt = enhanced_prompt.replace(placeholder, str(value))

        # Replace the main reasoning content placeholder
        reasoning_content = reasoning_elements.get("reasoning_content", "")
        enhanced_prompt = enhanced_prompt.replace(
            "{reasoning_content}", reasoning_content
        )

        return enhanced_prompt

    def integrate_knowledge(
        self, reasoning_text: str, knowledge_context: Dict[str, Any]
    ) -> str:
        """
        Integrate knowledge into reasoning text.

        Args:
            reasoning_text: Text with reasoning structure
            knowledge_context: Retrieved knowledge

        Returns:
            Reasoning text with integrated knowledge
        """
        # Base implementation - can be overridden by subclasses
        if not knowledge_context:
            return reasoning_text

        # Extract relevant knowledge sections
        knowledge_text = self._format_knowledge(knowledge_context)

        # Look for knowledge placeholders in the reasoning text
        if "{knowledge}" in reasoning_text:
            return reasoning_text.replace("{knowledge}", knowledge_text)

        # If no placeholder, insert knowledge at the beginning
        return (
            f"Based on the following knowledge:\n\n{knowledge_text}\n\n{reasoning_text}"
        )

    def _format_knowledge(self, knowledge_context: Dict[str, Any]) -> str:
        """
        Format knowledge context for integration.

        Args:
            knowledge_context: Retrieved knowledge

        Returns:
            Formatted knowledge text
        """
        if isinstance(knowledge_context, str):
            return knowledge_context

        if isinstance(knowledge_context, dict):
            # Different knowledge sources might be formatted differently
            if "text" in knowledge_context:
                return knowledge_context["text"]

            if "sections" in knowledge_context:
                sections = knowledge_context["sections"]
                if isinstance(sections, list):
                    return "\n\n".join(
                        section.get("content", "") for section in sections
                    )

            if "kg_data" in knowledge_context and knowledge_context["kg_data"]:
                return self._format_kg_data(knowledge_context["kg_data"])

            if "rag_data" in knowledge_context and knowledge_context["rag_data"]:
                return self._format_rag_data(knowledge_context["rag_data"])

            # If no recognized format, dump all text fields
            text_parts = []
            for key, value in knowledge_context.items():
                if isinstance(value, str):
                    text_parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    text_parts.append(f"{key}: {str(value)}")

            return "\n".join(text_parts)

        if isinstance(knowledge_context, list):
            return "\n\n".join(
                self._format_knowledge(item) for item in knowledge_context
            )

        # Default handling for unknown format
        return str(knowledge_context)

    def _format_kg_data(self, kg_data: Dict[str, Any]) -> str:
        """Format knowledge graph data."""
        if not kg_data:
            return ""

        if isinstance(kg_data, str):
            return kg_data

        data_type = kg_data.get("type", "")
        data = kg_data.get("data", [])

        if not data:
            return ""

        if data_type == "card_data":
            # Format card data
            parts = ["Card Information:"]

            for card in data:
                parts.append(f"Name: {card.get('name', 'Unknown')}")
                parts.append(f"Types: {', '.join(card.get('card_types', []))}")

                if card.get("subtypes"):
                    parts.append(f"Subtypes: {', '.join(card.get('subtypes', []))}")

                if card.get("mana_cost"):
                    parts.append(f"Mana Cost: {card.get('mana_cost')}")

                if card.get("text"):
                    parts.append(f"Text: {card.get('text')}")

                if card.get("power") and card.get("toughness"):
                    parts.append(
                        f"Power/Toughness: {card.get('power')}/{card.get('toughness')}"
                    )

                if card.get("loyalty"):
                    parts.append(f"Loyalty: {card.get('loyalty')}")

                parts.append("")  # Empty line between cards

            return "\n".join(parts)

        elif data_type == "rule_data":
            # Format rule data
            parts = ["Rules Information:"]

            for rule in data:
                parts.append(
                    f"Rule {rule.get('rule_id', 'Unknown')}: {rule.get('text', '')}"
                )
                parts.append("")  # Empty line between rules

            return "\n".join(parts)

        elif data_type == "card_list":
            # Format card list
            parts = ["Relevant Cards:"]

            for i, card in enumerate(data[:10]):  # Limit to 10 cards
                parts.append(
                    f"{i+1}. {card.get('name', 'Unknown')} - {card.get('text', '')}"
                )

            return "\n".join(parts)

        # Default handling
        return str(kg_data)

    def _format_rag_data(self, rag_data: Any) -> str:
        """Format RAG retrieval data."""
        if not rag_data:
            return ""

        if isinstance(rag_data, str):
            return rag_data

        if isinstance(rag_data, dict):
            # This might be a categorized retrieval result
            parts = ["Retrieved Information:"]

            for doc_type, docs in rag_data.items():
                if isinstance(docs, list):
                    parts.append(f"{doc_type.capitalize()} Documents:")

                    for doc in docs:
                        if isinstance(doc, dict) and "text" in doc:
                            parts.append(f"- {doc['text'][:300]}...")
                        else:
                            parts.append(f"- {str(doc)[:300]}...")

                        parts.append("")  # Empty line between documents

            return "\n".join(parts)

        if isinstance(rag_data, list):
            # This might be a list of retrieval results
            parts = ["Retrieved Information:"]

            for doc in rag_data:
                if isinstance(doc, dict) and "text" in doc:
                    parts.append(f"- {doc['text'][:300]}...")
                else:
                    parts.append(f"- {str(doc)[:300]}...")

                parts.append("")  # Empty line between documents

            return "\n".join(parts)

        # Default handling
        return str(rag_data)

    def _extract_relevant_knowledge(
        self, query: str, knowledge_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract knowledge relevant to the query.

        This method can be overridden by subclasses to implement
        more sophisticated knowledge filtering.

        Args:
            query: User query
            knowledge_context: Full knowledge context

        Returns:
            Filtered knowledge context with relevant information
        """
        # Base implementation - no filtering
        return knowledge_context
