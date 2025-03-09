# tests/models/test_transaction_classifier.py

import sys
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.transaction_classifier import TransactionClassifier


class TestTransactionClassifier:
    """Tests for the TransactionClassifier."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock = MagicMock()
        # Mock tokenizer to return a simple tensor dict
        tokenizer_output = {
            "input_ids": torch.tensor([[101, 2054, 2003, 1996, 2181, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]]),
        }

        # Add 'to' method to simulate device movement
        class DeviceMovable(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def to(self, device):
                return self

        mock_output = DeviceMovable(tokenizer_output)
        mock.return_value = mock_output
        return mock

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        mock = MagicMock()

        # Create mock output with logits
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.02]])
        mock.return_value = mock_output

        # Add 'to' method for device movement
        mock.to.return_value = mock
        return mock

    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.AutoModelForSequenceClassification.from_pretrained")
    def test_initialization(
        self, mock_model_init, mock_tokenizer_init, mock_tokenizer, mock_model
    ):
        """Test initialization of the transaction classifier."""
        # Set up mocks
        mock_tokenizer_init.return_value = mock_tokenizer
        mock_model_init.return_value = mock_model

        # Initialize classifier
        classifier = TransactionClassifier(
            model_path="test-model", num_labels=5, threshold=0.4, device="cpu"
        )

        # Verify initialization
        mock_tokenizer_init.assert_called_once_with("test-model")
        mock_model_init.assert_called_once()
        assert mock_model_init.call_args[0][0] == "test-model"
        assert mock_model_init.call_args[1]["num_labels"] == 5

        # Verify expert labels
        assert classifier.id2label == {
            0: "REASON",
            1: "EXPLAIN",
            2: "TEACH",
            3: "PREDICT",
            4: "RETROSPECT",
        }
        assert classifier.label2id == {
            "REASON": 0,
            "EXPLAIN": 1,
            "TEACH": 2,
            "PREDICT": 3,
            "RETROSPECT": 4,
        }
        assert classifier.threshold == pytest.approx(0.4)
        assert classifier.device == "cpu"

    @patch("torch.nn.functional.softmax")
    def test_classify_with_threshold(self, mock_softmax, mock_tokenizer, mock_model):
        """Test classification with threshold filtering."""
        # Set up mocks
        mock_probs = torch.tensor([0.6, 0.3, 0.05, 0.03, 0.02])
        mock_softmax.return_value = mock_probs

        # Create classifier with patched objects
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ), patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ):

            classifier = TransactionClassifier(threshold=0.2, device="cpu")

            # Test classification
            result = classifier.classify(
                "What happens when Lightning Bolt targets a creature with protection from red?"
            )

            # Verify thresholding worked correctly - should include REASON and EXPLAIN (>= 0.2)
            assert "REASON" in result
            assert "EXPLAIN" in result
            assert "TEACH" not in result
            assert "PREDICT" not in result
            assert "RETROSPECT" not in result

            # Verify probabilities (using approx for float precision)
            assert result["REASON"] == pytest.approx(0.6)
            assert result["EXPLAIN"] == pytest.approx(0.3)

    @patch("torch.nn.functional.softmax")
    def test_classify_no_threshold_match(
        self, mock_softmax, mock_tokenizer, mock_model
    ):
        """Test classification when no expert meets the threshold."""
        # Set up mocks - all below threshold
        mock_probs = torch.tensor([0.19, 0.18, 0.17, 0.16, 0.15])
        mock_softmax.return_value = mock_probs

        # Create classifier with patched objects
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ), patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ):

            classifier = TransactionClassifier(threshold=0.2, device="cpu")

            # Test classification
            result = classifier.classify("A very ambiguous question?")

            # Should return the highest probability expert despite being below threshold
            assert len(result) == 1
            assert "REASON" in result
            assert result["REASON"] == pytest.approx(0.19)

    @patch("torch.nn.functional.softmax")
    def test_get_top_k_experts(self, mock_softmax, mock_tokenizer, mock_model):
        """Test getting top k experts."""
        # Set up mocks
        mock_probs = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1])
        mock_softmax.return_value = mock_probs

        # Setup topk mock
        with patch("torch.topk") as mock_topk:
            mock_topk.return_value = (
                torch.tensor([0.3, 0.25]),
                torch.tensor([0, 1]),  # indices of top 2 values
            )

            # Create classifier with patched objects
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ), patch(
                "transformers.AutoModelForSequenceClassification.from_pretrained",
                return_value=mock_model,
            ):

                classifier = TransactionClassifier(device="cpu")

                # Test getting top 2 experts
                result = classifier.get_top_k_experts(
                    "How does Lightning Bolt interact with Spellskite?", k=2
                )

                # Should return top 2 experts
                assert len(result) == 2
                assert "REASON" in result
                assert "EXPLAIN" in result
                assert result["REASON"] == pytest.approx(0.3)
                assert result["EXPLAIN"] == pytest.approx(0.25)

                # Verify top_k was called correctly
                mock_topk.assert_called_once()
                assert mock_topk.call_args[1]["k"] == 2

    @patch("torch.nn.functional.softmax")
    def test_get_top_k_experts_limit(self, mock_softmax, mock_tokenizer, mock_model):
        """Test getting top k experts with k larger than number of experts."""
        # Set up mocks
        mock_probs = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1])
        mock_softmax.return_value = mock_probs

        # Setup topk mock
        with patch("torch.topk") as mock_topk:
            mock_topk.return_value = (
                torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1]),
                torch.tensor([0, 1, 2, 3, 4]),  # indices of all values
            )

            # Create classifier with patched objects
            with patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ), patch(
                "transformers.AutoModelForSequenceClassification.from_pretrained",
                return_value=mock_model,
            ):

                classifier = TransactionClassifier(device="cpu")

                # Test getting top 10 experts (more than we have)
                result = classifier.get_top_k_experts(
                    "What are the rules for mulligans?", k=10
                )

                # Should return all 5 experts
                assert len(result) == 5
                assert "REASON" in result
                assert "EXPLAIN" in result
                assert "TEACH" in result
                assert "PREDICT" in result
                assert "RETROSPECT" in result

                # Verify min() in topk works as expected
                mock_topk.assert_called_once()
                assert mock_topk.call_args[1]["k"] == 5  # Should be limited to 5

    def test_classify_empty_query(self, mock_tokenizer, mock_model):
        """Test classification with an empty query."""

        # Set up mocks for empty query
        # Create a mockable dictionary that has a 'to' method
        class DeviceMovable(dict):
            def __init__(self, data):
                super().__init__(data)

            def to(self, device):
                return self

        mock_tokenizer.return_value = DeviceMovable(
            {
                "input_ids": torch.tensor([[101, 102]]),  # Just special tokens
                "attention_mask": torch.tensor([[1, 1]]),
            }
        )

        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        mock_model.return_value = mock_output

        # Create classifier with patched objects
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ), patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ), patch(
            "torch.nn.functional.softmax",
            return_value=torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
        ):

            classifier = TransactionClassifier(threshold=0.3, device="cpu")

            # Test classification of empty query
            result = classifier.classify("")

            # Should default to the first expert when all have equal probability
            assert len(result) == 1
            assert "REASON" in result
            assert result["REASON"] == pytest.approx(0.2)

    @patch("torch.nn.functional.softmax")
    def test_expert_confidence_values(self, mock_softmax, mock_tokenizer, mock_model):
        """Test that confidence values are correctly processed."""
        # Set up specific probability values
        mock_probs = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
        mock_softmax.return_value = mock_probs

        # Create classifier with patched objects
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ), patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ):

            classifier = TransactionClassifier(threshold=0.15, device="cpu")

            # Test classification
            result = classifier.classify(
                "How many cards should I sideboard against burn decks?"
            )

            # Verify correct experts and probabilities
            assert "REASON" in result
            assert "EXPLAIN" in result
            assert "TEACH" in result
            assert "PREDICT" not in result
            assert "RETROSPECT" not in result

            # Check probability values and type
            assert isinstance(result["REASON"], float)
            assert result["REASON"] == pytest.approx(0.5)
            assert result["EXPLAIN"] == pytest.approx(0.2)
            assert result["TEACH"] == pytest.approx(0.15)

    def test_different_query_types(self, mock_tokenizer, mock_model):
        """Test classification of different types of queries."""
        # Define test queries and their expected primary classification
        test_cases = [
            (
                "What happens if I target a creature with Bolt and it gains hexproof in response?",
                "REASON",
            ),
            ("Can you explain the difference between hexproof and shroud?", "EXPLAIN"),
            ("How do I build a good mana base for a three-color deck?", "TEACH"),
            ("What will happen to card prices when the new set releases?", "PREDICT"),
            ("Why did my Azorius Control deck lose to Burn?", "RETROSPECT"),
        ]

        # Use patch to avoid actually loading models
        with patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
        ), patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ):

            # Test each query type separately
            for query, expected_expert in test_cases:
                # Override mock for each query type
                mock_output = MagicMock()
                logits = torch.zeros(5)
                # Use label2id directly to get index for the expert
                expert_idx = TransactionClassifier().label2id[expected_expert]
                logits[expert_idx] = 2.0  # Set high logit for expected expert
                mock_output.logits = logits.unsqueeze(0)
                mock_model.return_value = mock_output

                # Patch softmax to return proper probabilities
                probs = torch.zeros(5)
                # Use label2id directly to get index for the expert
                probs[expert_idx] = 0.8

                with patch("torch.nn.functional.softmax", return_value=probs):
                    classifier = TransactionClassifier(device="cpu")
                    result = classifier.classify(query)

                    # Verify the expected expert is identified with highest confidence
                    assert expected_expert in result
                    assert result[expected_expert] >= 0.5
