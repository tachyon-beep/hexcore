# tests/inference/test_enhanced_pipeline.py

import sys
import torch
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.inference.enhanced_pipeline import EnhancedMTGInferencePipeline
from src.models.cross_expert import CrossExpertAttention


class TestEnhancedInferencePipeline:
    """Integration tests for the enhanced inference pipeline."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock base model."""
        mock = MagicMock()
        # Mock generate method
        mock.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        # Mock hidden states for testing cross-expert attention
        mock_output = MagicMock()
        mock_output.hidden_states = [torch.randn(1, 5, 32) for _ in range(4)]
        mock.return_value = mock_output

        # Mock lm_head for the _generate_from_hidden_states method
        lm_head = MagicMock()
        # Make lm_head return a tensor when called
        lm_head.return_value = torch.randn(
            1, 5, 100
        )  # Mimics logits with vocab size 100
        mock.lm_head = lm_head

        return mock

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        mock = MagicMock()

        # Create a proper mock for the return value with a 'to' method
        tokenizer_output = MagicMock()
        tokenizer_output.input_ids = torch.tensor([[1, 2, 3]])
        tokenizer_output.attention_mask = torch.tensor([[1, 1, 1]])

        # Add the 'to' method that returns self
        tokenizer_output.to.return_value = tokenizer_output

        # Make the mock return the tokenizer_output object
        mock.return_value = tokenizer_output

        # Mock decode method for generating responses
        mock.decode.return_value = "Mocked response text"

        # Add encode method for conversation management
        mock.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs

        return mock

    @pytest.fixture
    def mock_classifier(self):
        """Create a mock transaction classifier."""
        mock = MagicMock()
        # Mock classify method for single expert
        mock.classify.return_value = {"REASON": 0.8}
        # Mock get_top_k_experts for multi-expert
        mock.get_top_k_experts.return_value = {"REASON": 0.6, "EXPLAIN": 0.4}
        return mock

    @pytest.fixture
    def mock_retriever(self):
        """Create a mock knowledge retriever."""
        mock = MagicMock()
        # Mock retrieve method
        mock.retrieve.return_value = [{"text": "Retrieved rule 1", "type": "rule"}]
        # Mock retrieve_by_categories method
        mock.retrieve_by_categories.return_value = {
            "rule": [{"text": "Retrieved rule 1", "type": "rule"}],
            "card": [{"text": "Card info", "type": "card"}],
        }
        return mock

    @pytest.fixture
    def mock_data_loader(self):
        """Create a mock data loader."""
        mock = MagicMock()
        # Mock cards attribute
        mock.cards = {
            "Lightning Bolt": {"name": "Lightning Bolt", "oracle_text": "Deal 3 damage"}
        }
        # Mock get_card method
        mock.get_card.return_value = {
            "name": "Lightning Bolt",
            "oracle_text": "Deal 3 damage",
        }
        # Mock get_rule method
        mock.get_rule.return_value = "When X, then Y"
        return mock

    @pytest.fixture
    def mock_expert_manager(self):
        """Create a mock expert manager."""
        mock = MagicMock()
        # Mock apply_adapter method
        mock.apply_adapter.return_value = True
        # Mock get_active_model method
        mock.get_active_model.return_value = MagicMock()
        # Mock base_model attribute
        mock.base_model = MagicMock()
        # Mock get_memory_usage_stats
        mock.get_memory_usage_stats.return_value = {"REASON": 700, "EXPLAIN": 650}
        return mock

    @pytest.fixture
    def mock_kv_cache_manager(self):
        """Create a mock KV cache manager."""
        mock = MagicMock()
        # Mock clear_cache method
        mock.clear_cache.return_value = None
        return mock

    @pytest.fixture
    def enhanced_pipeline(
        self,
        mock_model,
        mock_tokenizer,
        mock_classifier,
        mock_retriever,
        mock_data_loader,
        mock_expert_manager,
        mock_kv_cache_manager,
    ):
        """Create an enhanced inference pipeline with all mock components."""
        # Create a real cross-expert attention module
        cross_expert = CrossExpertAttention(hidden_size=32)

        # Mock parameters for embedding device detection
        mock_embed = MagicMock()
        mock_param = MagicMock()
        # Create a device property with correct structure
        mock_device = MagicMock()
        mock_device.type = "cpu"
        # Use property() to create a getter that returns our mock device
        type(mock_param).device = property(lambda self: mock_device)
        mock_embed.parameters.return_value = iter([mock_param])

        # Setup the model's embed_tokens attribute
        mock_model.embed_tokens = mock_embed
        mock_model.get_input_embeddings = MagicMock(return_value=mock_embed)

        # Create the pipeline
        pipeline = EnhancedMTGInferencePipeline(
            model=mock_model,
            tokenizer=mock_tokenizer,
            classifier=mock_classifier,
            retriever=mock_retriever,
            data_loader=mock_data_loader,
            expert_manager=mock_expert_manager,
            cross_expert_attention=cross_expert,
            device="cpu",
            kv_cache_manager=mock_kv_cache_manager,
            enable_monitoring=True,
            enable_circuit_breakers=True,
        )

        # Explicitly set embedding_device to a proper device string
        pipeline.embedding_device = "cpu"
        pipeline._determine_embedding_device = MagicMock(return_value="cpu")

        return pipeline

    def test_basic_query_processing(
        self, enhanced_pipeline, mock_classifier, mock_expert_manager
    ):
        """Test basic query processing with enhanced pipeline."""
        # Set up mock classifier to return a single expert
        mock_classifier.classify.return_value = {"REASON": 0.9}

        # Generate a response
        query = "How does Lightning Bolt work?"
        result = enhanced_pipeline.generate_response(
            query, use_multiple_experts=False, ensure_device_consistency=True
        )

        # Verify classifier was called with the query
        mock_classifier.classify.assert_called_once()

        # Verify expert adapter was applied
        assert mock_expert_manager.apply_adapter.call_count > 0

        # Verify response was generated
        assert "response" in result
        assert "expert_types" in result
        assert result["expert_types"] == ["REASON"]
        assert result["confidences"]["REASON"] == pytest.approx(0.9)
        assert "metrics" in result
        assert result["success"] is True

    def test_multi_expert_generation(
        self, enhanced_pipeline, mock_classifier, mock_expert_manager
    ):
        """Test query processing with multiple experts."""
        # Set up mock classifier to return multiple experts
        mock_classifier.get_top_k_experts.return_value = {"REASON": 0.7, "EXPLAIN": 0.3}

        # Generate a response with multiple experts
        query = "How does Lightning Bolt work in the current meta?"
        result = enhanced_pipeline.generate_response(
            query, use_multiple_experts=True, ensure_device_consistency=True
        )

        # Verify top_k experts were requested
        mock_classifier.get_top_k_experts.assert_called_once()

        # Verify response contains expected data
        assert "response" in result
        assert "expert_types" in result
        assert len(result["expert_types"]) == 2
        assert "REASON" in result["expert_types"]
        assert "EXPLAIN" in result["expert_types"]
        assert result["success"] is True

    def test_error_handling_and_fallback(
        self, enhanced_pipeline, mock_classifier, mock_expert_manager
    ):
        """Test error handling and fallback mechanisms."""
        # Make the expert manager fail to apply adapters
        mock_expert_manager.apply_adapter.return_value = False

        # Generate a response
        query = "How does Lightning Bolt work?"
        result = enhanced_pipeline.generate_response(query)

        # Verify response still works despite failure
        assert "response" in result
        # Even with failures, we should get some kind of response
        assert result["response"] != ""

    def test_conversation_context(self, enhanced_pipeline):
        """Test multi-turn conversation with context."""
        # First turn
        query1 = "What does Lightning Bolt do?"
        result1 = enhanced_pipeline.generate_response(
            query1, conversation_id="test_convo_1"
        )

        # Second turn with the same conversation ID
        query2 = "What about Shock?"
        result2 = enhanced_pipeline.generate_response(
            query2, conversation_id="test_convo_1"
        )

        # Verify the conversation manager has stored the interactions
        assert len(enhanced_pipeline.conversation_manager.history) == 2
        assert enhanced_pipeline.conversation_manager.history[0]["query"] == query1
        assert enhanced_pipeline.conversation_manager.history[1]["query"] == query2

    def test_circuit_breaker_functionality(self, enhanced_pipeline, mock_classifier):
        """Test circuit breaker functionality in an end-to-end scenario."""
        # Configure circuit breaker for testing
        circuit_breaker = enhanced_pipeline.retrieval_circuit_breaker
        circuit_breaker.failure_count = 0
        circuit_breaker.is_open = False
        circuit_breaker.failure_threshold = 2  # Lower threshold for testing

        # Configure the retriever to throw an exception to test circuit breaker
        retriever = enhanced_pipeline.retriever
        original_retrieve = retriever.retrieve

        # Create a failing retriever function that will trigger the circuit breaker
        def failing_retrieve(*args, **kwargs):
            raise RuntimeError("Simulated retrieval failure")

        # Replace the real retrieve method with our failing one
        retriever.retrieve = failing_retrieve

        try:
            # Verify circuit is initially closed
            assert not circuit_breaker.is_open

            # First generate response - should fail but use fallback
            result1 = enhanced_pipeline.generate_response("Test query")

            # Verify response still works despite retrieval failure
            assert result1["response"] != ""
            assert "error" not in result1 or not result1["error"]
            assert result1["success"] is True

            # Second generation - should fail again
            result2 = enhanced_pipeline.generate_response("Another test query")

            # Verify second response worked
            assert result2["response"] != ""

            # Third attempt - circuit should now be open
            result3 = enhanced_pipeline.generate_response("Third test query")

            # Verify circuit breaker is now open
            assert (
                circuit_breaker.is_open
            ), "Circuit breaker should be open after multiple failures"

            # Verify we still get a valid response using the fallback
            assert result3["response"] != ""
            assert result3["success"] is True

            # This confirms end-to-end that despite component failures:
            # 1. The circuit breaker correctly opens after threshold failures
            # 2. The pipeline continues to function with fallbacks
            # 3. The user experience is maintained despite failures

        finally:
            # Restore original retriever function
            retriever.retrieve = original_retrieve
