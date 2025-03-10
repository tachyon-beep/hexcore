import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from src.utils.expert_config import get_expert_id_mappings, get_expert_types

logger = logging.getLogger(__name__)


class TransactionClassifier:
    """
    A lightweight classifier that routes queries to appropriate experts.
    """

    def __init__(
        self,
        model_path: str = "distilbert-base-uncased",
        num_labels: int = 5,
        threshold: float = 0.5,
        device: str = "cuda:0",
    ):
        """
        Initialize the transaction classifier.

        Args:
            model_path: Path to pretrained model or model identifier
            num_labels: Number of expert types to classify
            threshold: Confidence threshold for selecting experts
            device: Device to run the model on
        """
        self.device = device
        self.threshold = threshold

        # Initialize with a base model first (we'll fine-tune later)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            torch_dtype=torch.float16,  # Use float16 for efficiency
        ).to(device)

        # Get expert type mappings from centralized configuration
        expert_types = get_expert_types()

        # If num_labels doesn't match the number of expert types, adjust model to fit available types
        if num_labels != len(expert_types):
            logger.warning(
                f"Model has {num_labels} labels but {len(expert_types)} expert types defined. "
                f"Using all {len(expert_types)} expert types from configuration."
            )
            # Re-create model with correct number of labels if needed
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(expert_types),
                torch_dtype=torch.float16,
            ).to(device)

        # Get the ID mappings from the configuration
        self.id2label, self.label2id = get_expert_id_mappings()

    def classify(self, query: str) -> Dict[str, float]:
        """
        Classify a query and return the appropriate expert types with confidence scores.

        Args:
            query: The input text query

        Returns:
            Dictionary mapping expert types to confidence scores
        """
        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=0)

        # Get expert types and confidence scores
        expert_confidence = {
            self.id2label[i]: float(prob)
            for i, prob in enumerate(probs)
            if prob >= self.threshold
        }

        # If no expert meets threshold, use the best one
        if not expert_confidence:
            top_idx = torch.argmax(probs).item()
            # Ensure top_idx is an integer
            top_idx_int = int(top_idx)
            expert_confidence = {self.id2label[top_idx_int]: float(probs[top_idx_int])}

        return expert_confidence

    def get_top_k_experts(self, query: str, k: int = 2) -> Dict[str, float]:
        """Get the top k experts for a query."""
        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=0)

        # Get top k experts
        top_k_probs, top_k_indices = torch.topk(probs, k=min(k, len(probs)))

        expert_confidence = {
            self.id2label[int(idx.item())]: float(prob)
            for idx, prob in zip(top_k_indices, top_k_probs)
        }

        return expert_confidence
