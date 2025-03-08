# src/inference/pipeline.py
import torch
from torch.nn import Module
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time

logger = logging.getLogger(__name__)


class MTGInferencePipeline:
    """
    Inference pipeline for the MTG AI assistant.
    """

    # Updated inference pipeline with expert adapters and cross-expert attention

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        classifier,
        retriever,
        data_loader,
        expert_manager,  # Add the expert manager
        cross_expert_attention=None,  # Add cross-expert attention
        device: str = "cuda",
    ):
        """Initialize the inference pipeline."""
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.retriever = retriever
        self.data_loader = data_loader
        self.expert_manager = expert_manager  # Store expert manager
        self.cross_expert_attention = (
            cross_expert_attention  # Store cross-expert attention
        )
        self.device = device

        # Initialize performance metrics
        self.metrics = {
            "classification_time": [],
            "retrieval_time": [],
            "generation_time": [],
            "total_time": [],
        }

        logger.info("Initialized MTG Inference Pipeline")

    def _generate_from_hidden_states(self, hidden_states):
        """
        Generate text from hidden states.

        Args:
            hidden_states: Hidden state tensor from the model

        Returns:
            Generated text as a string
        """
        # This is a simplified implementation
        # In practice, you would need to project back to vocabulary and decode

        # Assume we have a linear layer to project hidden states to logits
        # For now, we'll use the model's lm_head if available
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(hidden_states)

            # Get most likely tokens
            token_ids = torch.argmax(logits, dim=-1)

            # Decode the tokens
            return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        else:
            # Fallback if we can't generate directly from hidden states
            logger.warning(
                "Cannot generate directly from hidden states, using fallback method"
            )

            # For fallback, we'll generate from the base model without expert adapters
            # Get the base model without adapters
            base_model = self.expert_manager.base_model

            # Generate a short sequence to work with
            inputs = self.tokenizer("Continue the following:", return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                outputs = base_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                )

            # Extract only the generated part
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            return response

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        use_multiple_experts: bool = True,  # Add flag for using multiple experts
    ) -> Dict[str, Any]:
        """Generate a response using potentially multiple experts."""
        start_time = time.time()
        result = {
            "query": query,
            "response": "",
            "expert_types": [],
            "confidences": {},
            "metrics": {},
        }

        # Step 1: Classify query
        classify_start = time.time()
        if use_multiple_experts:
            # Get top 2 experts
            expert_confidence = self.classifier.get_top_k_experts(query, k=2)
        else:
            # Get single expert
            expert_confidence = self.classifier.classify(query)

        classify_time = time.time() - classify_start
        self.metrics["classification_time"].append(classify_time)

        result["expert_types"] = list(expert_confidence.keys())
        result["confidences"] = expert_confidence

        # Step 2: Retrieve knowledge
        retrieval_start = time.time()
        knowledge = self._retrieve_knowledge(query, result["expert_types"][0])
        retrieval_time = time.time() - retrieval_start
        self.metrics["retrieval_time"].append(retrieval_time)

        # Step 3: Generate with each selected expert
        generation_start = time.time()

        # If using multiple experts, generate from each and combine
        if use_multiple_experts and len(expert_confidence) > 1:
            expert_outputs = []

            for expert_type in expert_confidence.keys():
                # Apply this expert's adapter
                if self.expert_manager.apply_adapter(expert_type):
                    # Create expert-specific prompt
                    prompt = self._create_expert_prompt(query, expert_type, knowledge)

                    # Generate with this expert
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                    # Configure generation parameters based on expert type
                    generation_params = self._get_generation_params(
                        expert_type, inputs, max_new_tokens, temperature
                    )

                    # Generate from this expert
                    with torch.no_grad():
                        outputs = self.model.generate(**generation_params)

                    # Extract generated tokens
                    prompt_tokens = inputs.input_ids.shape[1]
                    response_tokens = outputs[0][prompt_tokens:]

                    # Convert to hidden states for cross-expert attention
                    # This is a simplified approach - in practice, you would probably
                    # extract hidden states from the model's forward pass
                    with torch.no_grad():
                        hidden_states = self.model(
                            input_ids=response_tokens.unsqueeze(0),
                            output_hidden_states=True,
                        ).hidden_states[
                            -1
                        ]  # Use the last layer's hidden states

                    expert_outputs.append(hidden_states)

            # Apply cross-expert attention if available
            if self.cross_expert_attention and len(expert_outputs) > 1:
                combined_hidden_states = self.cross_expert_attention(expert_outputs)

                # Generate final text from combined hidden states
                # This is simplified - you would typically need to project back to vocabulary
                # and generate text from the combined representation
                final_response = self._generate_from_hidden_states(
                    combined_hidden_states
                )
            else:
                # Fall back to selecting the response from the highest confidence expert
                primary_expert = max(expert_confidence.items(), key=lambda x: x[1])[0]
                final_response = self._generate_with_single_expert(
                    query, primary_expert, knowledge, max_new_tokens, temperature
                )
        else:
            # Use single expert (simpler path)
            primary_expert = result["expert_types"][0]
            final_response = self._generate_with_single_expert(
                query, primary_expert, knowledge, max_new_tokens, temperature
            )

        generation_time = time.time() - generation_start
        self.metrics["generation_time"].append(generation_time)

        # Calculate total time
        total_time = time.time() - start_time
        self.metrics["total_time"].append(total_time)

        # Prepare result
        result["response"] = final_response
        result["metrics"] = {
            "classification_time": classify_time,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": total_time,
        }

        return result

    def _generate_with_single_expert(
        self, query, expert_type, knowledge, max_new_tokens, temperature
    ):
        """Generate a response using a single expert."""
        # Apply this expert's adapter
        self.expert_manager.apply_adapter(expert_type)

        # Create prompt
        prompt = self._create_expert_prompt(query, expert_type, knowledge)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Configure generation parameters
        generation_params = self._get_generation_params(
            expert_type, inputs, max_new_tokens, temperature
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**generation_params)

        # Extract response
        prompt_tokens = inputs.input_ids.shape[1]
        response_tokens = outputs[0][prompt_tokens:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response

    def _get_generation_params(self, expert_type, inputs, max_new_tokens, temperature):
        """Get generation parameters based on expert type."""
        params = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Adjust parameters based on expert type
        if expert_type in ["EXPLAIN", "TEACH"]:
            params["temperature"] = 0.7  # More creative
            params["do_sample"] = True
            params["top_p"] = 0.9
        else:
            params["temperature"] = 0.3  # More precise
            params["do_sample"] = True
            params["top_p"] = 0.8

        return params

    def _retrieve_knowledge(self, query: str, expert_type: str) -> str:
        """
        Retrieve knowledge relevant to the query.

        Args:
            query: The user's query text.
            expert_type: The selected expert type.

        Returns:
            Formatted knowledge text for inclusion in the prompt.
        """
        parts = []

        card_info = self._get_card_information(query)
        if card_info:
            parts.append("Card Information:\n\n" + card_info)

        rule_info = self._get_rule_references(query)
        if rule_info:
            parts.append("Rules References:\n\n" + rule_info)

        docs_info = self._get_retrieved_documents(query, expert_type)
        if docs_info:
            parts.append("Retrieved Information:\n\n" + docs_info)

        if not parts:
            return "No specific MTG knowledge found for this query."

        return "\n\n".join(parts)

    def _get_card_information(self, query: str) -> str:
        """Extract and format card information based on card names found in the query."""
        # Find potential card names (case-insensitive search)
        card_names = [
            card
            for card in self.data_loader.cards.keys()
            if card.lower() in query.lower()
        ]
        # Prefer longer names for specificity and limit to top 3
        card_names.sort(key=len, reverse=True)
        card_names = card_names[:3]

        info = ""
        for card_name in card_names:
            card = self.data_loader.get_card(card_name)
            if not card:
                continue
            info += f"Name: {card['name']}\n"
            if "type_line" in card:
                info += f"Type: {card['type_line']}\n"
            if "mana_cost" in card:
                info += f"Mana Cost: {card['mana_cost']}\n"
            if "oracle_text" in card:
                info += f"Text: {card['oracle_text']}\n"
            if "power" in card and "toughness" in card:
                info += f"Power/Toughness: {card['power']}/{card['toughness']}\n"
            if "loyalty" in card:
                info += f"Loyalty: {card['loyalty']}\n"
            info += "\n"
        return info

    def _get_rule_references(self, query: str) -> str:
        """Extract and format rule references found in the query."""
        import re

        rule_pattern = r"\b(\d+\.\d+[a-z]?)\b"
        matches = re.findall(rule_pattern, query)
        if not matches:
            return ""

        info = ""
        for rule_id in matches:
            rule_text = self.data_loader.get_rule(rule_id)
            if rule_text:
                info += f"Rule {rule_id}: {rule_text}\n\n"
        return info

    def _get_retrieved_documents(self, query: str, expert_type: str) -> str:
        """Retrieve and format documents relevant to the query, adjusting strategy based on expert type."""
        # Determine document type based on expert type
        doc_type = None
        if expert_type == "REASON":
            doc_type = "rule"
        elif expert_type == "TEACH":
            doc_type = "guide"
        elif expert_type == "PREDICT":
            doc_type = "strategy"

        # Retrieve documents using the appropriate method
        if doc_type:
            docs = self.retriever.retrieve(query, top_k=3, doc_type=doc_type)
        else:
            docs_by_type = self.retriever.retrieve_by_categories(
                query, top_k_per_type=2
            )
            docs = []
            for doc_list in docs_by_type.values():
                docs.extend(doc_list)
            docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            docs = docs[:5]

        if not docs:
            return ""

        info = ""
        for doc in docs:
            doc_text = doc.get("text", "")
            dt = doc.get("type", "unknown")
            if len(doc_text) > 300:
                doc_text = doc_text[:300] + "..."
            info += f"[{dt.upper()}] {doc_text}\n\n"
        return info

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
