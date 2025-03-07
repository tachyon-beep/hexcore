# src/inference/pipeline.py
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time

logger = logging.getLogger(__name__)


class MTGInferencePipeline:
    """
    Inference pipeline for the MTG AI assistant.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        classifier,
        retriever,
        data_loader,
        device: str = "cuda",
    ):
        """Initialize the inference pipeline."""
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.retriever = retriever
        self.data_loader = data_loader
        self.device = device

        # Performance metrics
        self.metrics = {
            "classification_time": [],
            "retrieval_time": [],
            "generation_time": [],
            "total_time": [],
        }

        logger.info("Initialized MTG Inference Pipeline")

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate a response to the query with timing metrics."""
        start_time = time.time()
        result = {
            "query": query,
            "response": "",
            "expert_type": "",
            "confidence": 0.0,
            "metrics": {},
        }

        # Step 1: Classify query
        classify_start = time.time()
        expert_confidence = self.classifier.classify(query)
        primary_expert = max(expert_confidence.items(), key=lambda x: x[1])
        expert_type, confidence = primary_expert
        classify_time = time.time() - classify_start
        self.metrics["classification_time"].append(classify_time)

        result["expert_type"] = expert_type
        result["confidence"] = confidence

        # Step 2: Retrieve knowledge
        retrieval_start = time.time()
        knowledge = self._retrieve_knowledge(query, expert_type)
        retrieval_time = time.time() - retrieval_start
        self.metrics["retrieval_time"].append(retrieval_time)

        # Step 3: Prepare prompt
        prompt = self._create_expert_prompt(query, expert_type, knowledge)

        # Step 4: Generate response
        generation_start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Dynamically adjust generation parameters based on expert type
        generation_params = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Adjust temperature based on expert type
        if expert_type in ["EXPLAIN", "TEACH"]:
            # More creative for explanations and teaching
            generation_params["temperature"] = 0.7
            generation_params["do_sample"] = True
            generation_params["top_p"] = 0.9
        else:
            # More precise for reasoning, prediction, and retrospection
            generation_params["temperature"] = 0.3
            generation_params["do_sample"] = True
            generation_params["top_p"] = 0.8

        with torch.no_grad():
            outputs = self.model.generate(**generation_params)

        # Extract only the generated part (after the prompt)
        prompt_tokens = inputs.input_ids.shape[1]
        response_tokens = outputs[0][prompt_tokens:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        generation_time = time.time() - generation_start
        self.metrics["generation_time"].append(generation_time)

        # Calculate total time
        total_time = time.time() - start_time
        self.metrics["total_time"].append(total_time)

        # Prepare result
        result["response"] = response
        result["metrics"] = {
            "classification_time": classify_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
        }

        return result

    def _retrieve_knowledge(self, query: str, expert_type: str) -> str:
        """
        Retrieve knowledge relevant to the query.

        Args:
            query: The user's query text
            expert_type: The selected expert type

        Returns:
            Formatted knowledge text for inclusion in the prompt
        """
        # Initialize empty knowledge string
        knowledge_text = ""

        # Extract potential card names from query
        card_names = []
        for card_name in self.data_loader.cards.keys():
            if card_name.lower() in query.lower():
                card_names.append(card_name)

        # Sort card names by length (prefer longer, more specific matches)
        card_names.sort(key=len, reverse=True)

        # Limit to top 3 cards to avoid information overload
        card_names = card_names[:3]

        # Retrieve card information if any cards were found
        if card_names:
            knowledge_text += "Card Information:\n\n"
            for card_name in card_names:
                card = self.data_loader.get_card(card_name)
                if card:
                    knowledge_text += f"Name: {card['name']}\n"
                    if "type_line" in card:
                        knowledge_text += f"Type: {card['type_line']}\n"
                    if "mana_cost" in card:
                        knowledge_text += f"Mana Cost: {card['mana_cost']}\n"
                    if "oracle_text" in card:
                        knowledge_text += f"Text: {card['oracle_text']}\n"
                    if "power" in card and "toughness" in card:
                        knowledge_text += (
                            f"Power/Toughness: {card['power']}/{card['toughness']}\n"
                        )
                    if "loyalty" in card:
                        knowledge_text += f"Loyalty: {card['loyalty']}\n"
                    knowledge_text += "\n"

        # Check for rules references (like "rule 101.2" or similar patterns)
        import re

        rule_pattern = r"\b(\d+\.\d+[a-z]?)\b"
        rule_matches = re.findall(rule_pattern, query)

        if rule_matches:
            knowledge_text += "Rules References:\n\n"
            for rule_id in rule_matches:
                rule_text = self.data_loader.get_rule(rule_id)
                if rule_text:
                    knowledge_text += f"Rule {rule_id}: {rule_text}\n\n"

        # Adapt retrieval strategy based on expert type
        doc_type = None
        if expert_type == "REASON":
            # Prioritize rules for reasoning expert
            doc_type = "rule"
        elif expert_type == "TEACH":
            # Prioritize educational content for teaching expert
            doc_type = "guide"
        elif expert_type == "PREDICT":
            # Prioritize strategic content for prediction expert
            doc_type = "strategy"

        # Use the retriever to get relevant documents
        if doc_type:
            # Get documents of specific type
            docs = self.retriever.retrieve(query, top_k=3, doc_type=doc_type)
        else:
            # For EXPLAIN and RETROSPECT, get balanced results across document types
            docs_by_type = self.retriever.retrieve_by_categories(
                query, top_k_per_type=2
            )
            docs = []
            for doc_list in docs_by_type.values():
                docs.extend(doc_list)
            # Sort by relevance score
            docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            # Limit to top 5
            docs = docs[:5]

        # Add retrieved documents to knowledge text
        if docs:
            knowledge_text += "Retrieved Information:\n\n"
            for doc in docs:
                doc_text = doc.get("text", "")
                doc_type = doc.get("type", "unknown")
                # Truncate long documents for prompt efficiency
                if len(doc_text) > 300:
                    doc_text = doc_text[:300] + "..."
                knowledge_text += f"[{doc_type.upper()}] {doc_text}\n\n"

        # If no knowledge was found, return empty string
        if not knowledge_text.strip():
            knowledge_text = "No specific MTG knowledge found for this query."

        return knowledge_text

    def _create_expert_prompt(
        self, query: str, expert_type: str, knowledge: str
    ) -> str:
        """
        Create an expert-specific prompt for the model.

        Args:
            query: The user's query text
            expert_type: The selected expert type
            knowledge: Retrieved knowledge text

        Returns:
            Formatted prompt for the model
        """
        # Expert-specific instructions and personas
        expert_instructions = {
            "REASON": (
                "You are an expert MTG rules advisor. Analyze this Magic: The Gathering question "
                "with careful step-by-step reasoning. Consider all relevant rules and card interactions. "
                "Provide a clear, accurate answer with references to specific rules when applicable."
            ),
            "EXPLAIN": (
                "You are a skilled MTG mentor. Explain this Magic: The Gathering concept clearly and "
                "concisely to make it easy to understand. Use examples where helpful, and break down "
                "complex ideas into simpler components."
            ),
            "TEACH": (
                "You are an experienced MTG instructor. Teach about this Magic: The Gathering topic "
                "in an educational manner. Assume the person is new to this aspect of the game. "
                "Provide a structured explanation with examples and gradually build up complexity."
            ),
            "PREDICT": (
                "You are a strategic MTG analyst. Predict the optimal play or outcome for this "
                "Magic: The Gathering scenario. Consider different lines of play, evaluate probabilities, "
                "and suggest the most advantageous approach."
            ),
            "RETROSPECT": (
                "You are an insightful MTG coach. Analyze this Magic: The Gathering game situation "
                "retrospectively. Identify key decision points, evaluate what could have been done "
                "differently, and suggest improvements for future games."
            ),
        }

        # Get the appropriate instruction for this expert type
        instruction = expert_instructions.get(
            expert_type,
            "You are a knowledgeable Magic: The Gathering assistant. Answer this question about MTG.",
        )

        # Create the full prompt with instruction, knowledge, and query
        prompt = f"{instruction}\n\n"

        # Add knowledge if available
        if knowledge and knowledge.strip():
            prompt += f"[KNOWLEDGE]\n{knowledge}\n[/KNOWLEDGE]\n\n"

        # Add the user's query
        prompt += f"Query: {query}\n\nResponse:"

        return prompt
