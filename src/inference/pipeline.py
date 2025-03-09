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
        expert_manager,
        cross_expert_attention=None,
        device: str = "cuda",
        kv_cache_manager=None,
    ):
        """
        Initialize the inference pipeline.

        Args:
            model: The base language model
            tokenizer: Tokenizer for the model
            classifier: Transaction classifier for routing queries
            retriever: Knowledge retriever
            data_loader: MTG data loader for cards and rules
            expert_manager: Expert adapter manager
            cross_expert_attention: Optional cross-expert attention module
            device: Base device to use (usually "cuda" or "cuda:0")
            kv_cache_manager: Optional KV cache manager for memory optimization
        """
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.retriever = retriever
        self.data_loader = data_loader
        self.expert_manager = expert_manager
        self.cross_expert_attention = cross_expert_attention
        self.device = device
        self.kv_cache_manager = kv_cache_manager

        # Get embedding device for inputs - crucial for MoE models
        self.embedding_device = self._determine_embedding_device()

        # Initialize performance metrics
        self.metrics = {
            "classification_time": [],
            "retrieval_time": [],
            "generation_time": [],
            "total_time": [],
        }

        logger.info(
            f"Initialized MTG Inference Pipeline (embedding device: {self.embedding_device})"
        )

    def _determine_embedding_device(self) -> str:
        """
        Determine the device where the embedding layer resides.
        This is crucial for ensuring inputs go to the right device.

        Returns:
            Device string for the embedding layer
        """
        try:
            # First try: check if model.model.embed_tokens exists and get its device
            if hasattr(self.model, "model") and hasattr(
                self.model.model, "embed_tokens"
            ):
                embed_device = next(self.model.model.embed_tokens.parameters()).device
                logger.info(f"Embedding layer found on {embed_device}")
                return str(embed_device)

            # Second try: for some model architectures
            if hasattr(self.model, "get_input_embeddings"):
                embeddings = self.model.get_input_embeddings()
                if embeddings is not None:
                    embed_device = next(embeddings.parameters()).device
                    logger.info(f"Input embeddings found on {embed_device}")
                    return str(embed_device)

            # Fallback: use the first parameter's device
            logger.warning(
                "Could not find embedding layer, using first parameter's device"
            )
            device = next(self.model.parameters()).device
            return str(device)
        except Exception as e:
            logger.warning(f"Error determining embedding device: {str(e)}")
            # Default fallback
            return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _get_tensor_length(self, tensor_obj: Any) -> int:
        """
        Safely get the length (sequence dimension) of a tensor-like object.
        Works with torch Tensors, lists, transformers' BatchEncoding, etc.

        Args:
            tensor_obj: The tensor-like object to get length from

        Returns:
            Integer length of the sequence dimension
        """
        # Handle various types
        try:
            # If it's a torch tensor
            if isinstance(tensor_obj, torch.Tensor):
                return tensor_obj.size(1) if len(tensor_obj.size()) > 1 else 1

            # If it has shape attribute (like numpy arrays)
            if hasattr(tensor_obj, "shape"):
                shape = tensor_obj.shape
                return shape[1] if len(shape) > 1 else 1

            # If it has size method (like some tokenizer outputs)
            if hasattr(tensor_obj, "size") and callable(getattr(tensor_obj, "size")):
                size = tensor_obj.size()
                return size[1] if len(size) > 1 else 1

            # If it's a list or list-like object
            if hasattr(tensor_obj, "__len__"):
                # If it's a nested structure, use the length of first item
                if len(tensor_obj) > 0 and hasattr(tensor_obj[0], "__len__"):
                    return len(tensor_obj[0])
                return len(tensor_obj)

            # If we got here, try to convert to int
            return int(tensor_obj)

        except Exception as e:
            logger.warning(f"Error determining tensor length: {str(e)}")
            # Default fallback - assume length 1
            return 1

    def _generate_from_hidden_states(self, hidden_states):
        """
        Generate text from hidden states.

        Args:
            hidden_states: Hidden state tensor from the model

        Returns:
            Generated text as a string
        """
        # This method handles generation from hidden states, with fallback mechanisms
        # for situations where direct generation isn't possible.

        # First attempt: direct projection using the model's language modeling head
        if hasattr(self.model, "lm_head"):
            try:
                # Project hidden states to vocabulary space
                logits = self.model.lm_head(hidden_states)

                # Get most likely tokens
                token_ids = torch.argmax(logits, dim=-1)

                # Decode the tokens
                return self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
            except Exception as e:
                logger.warning(
                    f"Error projecting hidden states to vocabulary: {str(e)}"
                )
                # Continue to second attempt

        # Second attempt: use the hidden states to initialize a new generation
        try:
            # Get the base model
            base_model = self.expert_manager.base_model

            # Create a dummy input to start generation
            seed_text = "Continue from context:"
            inputs = self.tokenizer(seed_text, return_tensors="pt").to(self.device)

            # Create a custom forward function that uses our hidden states
            # This is a simplified approach that injects the provided hidden states
            # at the beginning of generation
            original_forward = base_model.forward
            last_hidden_states = hidden_states

            def custom_forward(*args, **kwargs):
                # On first call, initialize with our hidden states
                nonlocal last_hidden_states
                if kwargs.get("use_cache", False) and last_hidden_states is not None:
                    # Inject our hidden states
                    result = original_forward(*args, **kwargs)

                    # Replace the last hidden state with our provided one
                    # This is a simplification; in practice, you'd need to match dimensions
                    if isinstance(result, tuple) and len(result) > 1:
                        result = (result[0], last_hidden_states) + result[2:]

                    # Only use custom hidden states on first call
                    last_hidden_states = None
                    return result
                return original_forward(*args, **kwargs)

            # Temporarily replace the forward function
            base_model.forward = custom_forward

            try:
                with torch.no_grad():
                    outputs = base_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        use_cache=True,
                    )

                # Extract only the generated part
                prompt_length = self._get_tensor_length(inputs.input_ids)
                response = self.tokenizer.decode(
                    outputs[0][prompt_length:], skip_special_tokens=True
                )

                return response
            finally:
                # Restore the original forward function
                base_model.forward = original_forward
        except Exception as e:
            logger.warning(f"Error using hidden states for generation: {str(e)}")

        # Final fallback: generate completely new text without using hidden states
        logger.warning(
            "Cannot use hidden states for generation, falling back to standard generation"
        )

        # Get the base model
        base_model = self.expert_manager.base_model

        # Generate a response
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
        prompt_length = self._get_tensor_length(inputs.input_ids)
        response = self.tokenizer.decode(
            outputs[0][prompt_length:], skip_special_tokens=True
        )

        return response

    def _ensure_device_consistency(self, inputs: Any) -> Any:
        """
        Ensure all input tensors are on the correct device (same as embedding layer).
        Works with both dictionaries of tensors and BatchEncoding objects from tokenizers.

        Args:
            inputs: Dictionary or BatchEncoding object with input tensors

        Returns:
            Inputs with tensors moved to the right device
        """
        # Handle case when inputs is None
        if inputs is None:
            return inputs

        # Determine target device
        target_device = self.embedding_device

        # For BatchEncoding from tokenizer
        if hasattr(inputs, "to") and callable(getattr(inputs, "to")):
            # Use BatchEncoding's native .to() method which handles the device conversion
            try:
                return inputs.to(target_device)
            except Exception as e:
                logger.warning(f"Error moving BatchEncoding to device: {str(e)}")
                # Fall back to manual conversion

        # Manual conversion for dictionary-like objects
        try:
            device_aligned_inputs = {}
            for name, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    if tensor.device != torch.device(target_device):
                        logger.debug(
                            f"Moving input tensor '{name}' from {tensor.device} to {target_device}"
                        )
                        device_aligned_inputs[name] = tensor.to(target_device)
                    else:
                        device_aligned_inputs[name] = tensor
                else:
                    device_aligned_inputs[name] = tensor
            return device_aligned_inputs
        except Exception as e:
            logger.warning(f"Error in device alignment: {str(e)}")
            # If all else fails, return the original inputs
            return inputs

    def generate_response(
        self,
        query: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        use_multiple_experts: bool = True,
        ensure_device_consistency: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a response using potentially multiple experts.

        Args:
            query: User query string
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            use_multiple_experts: Whether to use multiple experts
            ensure_device_consistency: Whether to enforce device consistency

        Returns:
            Dictionary with response and metadata
        """
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

        # Step 1b: Prefetch experts in background (if multiple experts needed)
        if use_multiple_experts and len(expert_confidence) > 1:
            primary_expert = max(expert_confidence.items(), key=lambda x: x[1])[0]
            # Prefetch all other experts with high confidence
            for expert_type in expert_confidence.keys():
                if expert_type != primary_expert:
                    try:
                        logger.debug(f"Prefetching expert {expert_type} in background")
                        # No need to wait for result - this just marks it for prefetching
                        self.expert_manager.prefetch_expert(expert_type)
                    except Exception as e:
                        logger.warning(f"Error prefetching expert {expert_type}: {e}")

        # Step 2: Retrieve knowledge
        retrieval_start = time.time()
        knowledge = self._retrieve_knowledge(query, result["expert_types"][0])
        retrieval_time = time.time() - retrieval_start
        self.metrics["retrieval_time"].append(retrieval_time)

        # Step 3: Generate with each selected expert
        generation_start = time.time()

        # Add expert memory usage statistics if available
        result["memory_stats"] = {}
        try:
            expert_memory_stats = self.expert_manager.get_memory_usage_stats()
            result["memory_stats"]["experts"] = expert_memory_stats
        except Exception as e:
            logger.warning(f"Error getting expert memory stats: {e}")

        # Clear KV cache before generation if cache manager exists
        if hasattr(self, "kv_cache_manager") and self.kv_cache_manager:
            self.kv_cache_manager.clear_cache()

        # If using multiple experts, generate from each and combine
        if use_multiple_experts and len(expert_confidence) > 1:
            expert_outputs = []

            for expert_type in expert_confidence.keys():
                # Apply this expert's adapter to the model and ensure it's on the right device
                # We pass the target device explicitly (the same as embedding layer)
                target_device = torch.device(self.embedding_device)
                if self.expert_manager.apply_adapter(expert_type, target_device):
                    # Create expert-specific prompt
                    prompt = self._create_expert_prompt(query, expert_type, knowledge)

                    # Tokenize prompt
                    inputs = self.tokenizer(prompt, return_tensors="pt")

                    # Make sure inputs go to the same device as the embedding layer
                    if ensure_device_consistency:
                        inputs = self._ensure_device_consistency(inputs)

                    # Configure generation parameters based on expert type
                    generation_params = self._get_generation_params(
                        expert_type, inputs, max_new_tokens, temperature
                    )

                    try:
                        # Generate from this expert
                        with torch.no_grad():
                            outputs = self.model.generate(**generation_params)

                        # Extract generated tokens safely using our helper method
                        prompt_tokens = self._get_tensor_length(inputs["input_ids"])
                        response_tokens = outputs[0][prompt_tokens:]

                        # Convert to hidden states for cross-expert attention
                        with torch.no_grad():
                            hidden_states = self.model(
                                input_ids=response_tokens.unsqueeze(0).to(
                                    target_device
                                ),
                                output_hidden_states=True,
                            ).hidden_states[
                                -1
                            ]  # Use the last layer's hidden states

                        expert_outputs.append(hidden_states)

                    except Exception as e:
                        logger.error(
                            f"Error generating output with expert {expert_type}: {str(e)}"
                        )
                        continue
                else:
                    logger.warning(f"Failed to apply adapter for expert {expert_type}")

            # Apply cross-expert attention if available
            if self.cross_expert_attention and len(expert_outputs) > 1:
                try:
                    combined_hidden_states = self.cross_expert_attention(expert_outputs)

                    # Generate final text from combined hidden states
                    final_response = self._generate_from_hidden_states(
                        combined_hidden_states
                    )
                except Exception as e:
                    logger.error(f"Error applying cross-expert attention: {str(e)}")
                    # Fall back to selecting the response from the highest confidence expert
                    primary_expert = max(expert_confidence.items(), key=lambda x: x[1])[
                        0
                    ]
                    final_response = self._generate_with_single_expert(
                        query,
                        primary_expert,
                        knowledge,
                        max_new_tokens,
                        temperature,
                        ensure_device_consistency,
                    )
            else:
                # Fall back to selecting the response from the highest confidence expert
                primary_expert = max(expert_confidence.items(), key=lambda x: x[1])[0]
                final_response = self._generate_with_single_expert(
                    query,
                    primary_expert,
                    knowledge,
                    max_new_tokens,
                    temperature,
                    ensure_device_consistency,
                )
        else:
            # Use single expert (simpler path)
            primary_expert = result["expert_types"][0]
            final_response = self._generate_with_single_expert(
                query,
                primary_expert,
                knowledge,
                max_new_tokens,
                temperature,
                ensure_device_consistency,
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
        self,
        query,
        expert_type,
        knowledge,
        max_new_tokens,
        temperature,
        ensure_device_consistency=True,
    ):
        """
        Generate a response using a single expert.

        Args:
            query: User query
            expert_type: Expert type to use
            knowledge: Retrieved knowledge
            max_new_tokens: Maximum new tokens to generate
            temperature: Temperature for sampling
            ensure_device_consistency: Whether to enforce device consistency

        Returns:
            Generated response text
        """
        # Determine target device (embedding device)
        target_device = torch.device(self.embedding_device)

        # Apply this expert's adapter with explicit device target
        self.expert_manager.apply_adapter(expert_type, target_device)

        # Create prompt
        prompt = self._create_expert_prompt(query, expert_type, knowledge)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Ensure inputs go to the same device as the embedding layer
        if ensure_device_consistency:
            inputs = self._ensure_device_consistency(inputs)

        # Configure generation parameters
        generation_params = self._get_generation_params(
            expert_type, inputs, max_new_tokens, temperature
        )

        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**generation_params)

            # Extract response using our safe helper method
            prompt_tokens = self._get_tensor_length(inputs["input_ids"])
            response_tokens = outputs[0][prompt_tokens:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

            return response
        except RuntimeError as e:
            if (
                "expected device" in str(e).lower()
                or "device mismatch" in str(e).lower()
            ):
                # Special handling for device mismatch errors
                logger.error(f"Device mismatch error: {str(e)}")
                logger.info("Attempting recovery with full device alignment...")

                # Diagnostic info for debugging
                model_devices = set()
                for name, param in self.model.named_parameters():
                    if param.device not in model_devices:
                        model_devices.add(param.device)
                logger.info(f"Model parameters are on devices: {model_devices}")

                # More aggressive device alignment
                for key, tensor in inputs.items():
                    if isinstance(tensor, torch.Tensor):
                        # Try all detected devices until one works
                        for device in model_devices:
                            try:
                                inputs[key] = tensor.to(device)
                                generation_params[key] = inputs[key]

                                # Attempt generation with this device
                                with torch.no_grad():
                                    outputs = self.model.generate(**generation_params)

                                # If we got here, it worked
                                prompt_tokens = self._get_tensor_length(
                                    inputs["input_ids"]
                                )
                                response_tokens = outputs[0][prompt_tokens:]
                                return self.tokenizer.decode(
                                    response_tokens, skip_special_tokens=True
                                )
                            except Exception:
                                # Try next device
                                continue

                # If all recovery attempts failed, return a helpful error message
                return f"I encountered a device mismatch error while processing your query with the {expert_type} expert. This issue has been logged for investigation."

            # For other runtime errors
            logger.error(f"Error during generation: {str(e)}")
            return f"I encountered an error while processing your query with the {expert_type} expert. This issue has been logged for investigation."
        except Exception as e:
            # For any other exceptions
            logger.error(f"Unexpected error during generation: {str(e)}")
            return f"I encountered an unexpected error while processing your query. This issue has been logged for investigation."

    def _get_generation_params(self, expert_type, inputs, max_new_tokens, temperature):
        """Get generation parameters based on expert type."""
        params = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
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
