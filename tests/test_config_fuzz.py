"""
Property-based tests (fuzzing) for config.py using Hypothesis.

These tests explore edge cases and unexpected inputs to find bugs
that unit tests might miss.
"""
import pytest
from hypothesis import given, strategies as st, assume, settings, Phase

from pathlib import Path

from delia.config import (
    Config, BackendHealth, ModelInfo, parse_model_name, detect_model_tier,
    _detect_model_tier_cached, MODEL_FAMILIES, CODER_KEYWORDS
)


class TestParseModelNameFuzz:
    """Property-based tests for model name parsing."""

    @pytest.mark.fuzz
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=200, phases=[Phase.generate, Phase.target])
    def test_parse_model_name_never_crashes(self, model_name):
        """parse_model_name should handle any string without crashing."""
        result = parse_model_name(model_name)

        # Should always return a ModelInfo
        assert isinstance(result, ModelInfo)
        assert isinstance(result.params_b, (int, float))
        assert result.params_b >= 0
        assert isinstance(result.is_coder, bool)
        assert isinstance(result.is_moe, bool)
        assert isinstance(result.is_instruct, bool)
        assert result.raw_name == model_name

    @pytest.mark.fuzz
    @given(st.sampled_from([
        "qwen3:14b", "qwen2.5-coder:14b", "qwen3:30b-a3b",
        "llama-3.1-70b-instruct", "llama-3-8b", "codellama:34b",
        "mixtral-8x7b", "mistral-7b-instruct",
        "deepseek-coder-33b", "deepseek-v2:16b",
        "phi-3-mini", "phi-4:14b",
        "gemma-2-9b", "codegemma:7b",
        "starcoder2:15b", "wizardcoder:34b"
    ]))
    def test_parse_known_models_detects_params(self, model_name):
        """Known model names should have reasonable param detection."""
        result = parse_model_name(model_name)

        # Most known models should have detectable param counts
        # (some like phi-3-mini won't have explicit size)
        if any(size in model_name for size in ["7b", "8b", "9b", "14b", "15b", "16b", "30b", "33b", "34b", "70b"]):
            assert result.params_b > 0, f"Failed to detect params in {model_name}"

    @pytest.mark.fuzz
    @given(st.integers(min_value=1, max_value=1000))
    def test_parse_model_with_param_sizes(self, param_size):
        """Models with param sizes should be parsed correctly."""
        for suffix in ["b", "B"]:
            model_name = f"test-model-{param_size}{suffix}"
            result = parse_model_name(model_name)
            assert result.params_b == float(param_size), f"Expected {param_size}, got {result.params_b}"

    @pytest.mark.fuzz
    @given(st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False))
    def test_parse_model_with_decimal_params(self, param_size):
        """Models with decimal param sizes like 6.7b should be parsed."""
        model_name = f"test-model-{param_size:.1f}b"
        result = parse_model_name(model_name)
        assert abs(result.params_b - round(param_size, 1)) < 0.2

    @pytest.mark.fuzz
    @given(st.sampled_from(list(CODER_KEYWORDS)))
    def test_coder_keywords_detected(self, keyword):
        """All coder keywords should trigger is_coder=True."""
        model_name = f"test-{keyword}-model"
        result = parse_model_name(model_name)
        assert result.is_coder, f"Coder keyword '{keyword}' not detected"

    @pytest.mark.fuzz
    @given(st.integers(min_value=1, max_value=16), st.integers(min_value=1, max_value=70))
    def test_moe_pattern_detection(self, experts, per_expert):
        """MoE patterns like 8x7b should be detected and calculated correctly."""
        model_name = f"mixtral-{experts}x{per_expert}b"
        result = parse_model_name(model_name)

        assert result.is_moe, f"MoE not detected in {model_name}"
        expected_total = experts * per_expert
        assert result.params_b == float(expected_total), f"Expected {expected_total}B, got {result.params_b}B"


class TestDetectModelTierFuzz:
    """Property-based tests for model tier detection."""

    @pytest.mark.fuzz
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=200)
    def test_detect_tier_never_crashes(self, model_name):
        """detect_model_tier should handle any string without crashing."""
        result = detect_model_tier(model_name)
        assert result in {"quick", "coder", "moe"}

    @pytest.mark.fuzz
    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=100)
    def test_detect_tier_with_known_models(self, model_name):
        """detect_model_tier should respect known_models mapping."""
        known = {"quick": "fast-model", "coder": "code-model", "moe": "big-model"}

        for tier, model in known.items():
            result = detect_model_tier(model, known)
            assert result == tier

    @pytest.mark.fuzz
    @given(st.integers(min_value=30, max_value=1000))
    def test_large_models_get_moe_tier(self, param_size):
        """Models with >=30B params should get moe tier."""
        model_name = f"large-model-{param_size}b"
        result = detect_model_tier(model_name)
        assert result == "moe", f"{param_size}B model should be moe tier"

    @pytest.mark.fuzz
    @given(st.integers(min_value=15, max_value=29))
    def test_medium_models_get_coder_tier(self, param_size):
        """Models with 15-29B params should get coder tier."""
        model_name = f"medium-model-{param_size}b"
        result = detect_model_tier(model_name)
        assert result == "coder", f"{param_size}B model should be coder tier"

    @pytest.mark.fuzz
    @given(st.integers(min_value=1, max_value=14))
    def test_small_models_default_to_quick(self, param_size):
        """Small models without coder keywords should default to quick."""
        model_name = f"tiny-model-{param_size}b"
        result = detect_model_tier(model_name)
        # Small non-coder models should be quick
        assert result == "quick", f"{param_size}B generic model should be quick tier"


class TestBackendHealthFuzz:
    """Property-based tests for BackendHealth circuit breaker."""

    @pytest.mark.fuzz
    @given(st.integers(min_value=0, max_value=100))
    def test_failure_counting(self, num_failures):
        """Recording failures should increment counter correctly."""
        health = BackendHealth(name="test")

        for _ in range(num_failures):
            health.record_failure("timeout")

        assert health.consecutive_failures == num_failures

    @pytest.mark.fuzz
    @given(st.integers(min_value=0, max_value=10))
    def test_circuit_opens_at_threshold(self, failures_before_threshold):
        """Circuit should open after threshold failures."""
        health = BackendHealth(name="test", failure_threshold=3)

        # Failures below threshold
        for _ in range(min(failures_before_threshold, 2)):
            health.record_failure("timeout")
            assert health.is_available(), "Circuit shouldn't open before threshold"

        # Hit threshold
        for _ in range(3 - min(failures_before_threshold, 2)):
            health.record_failure("timeout")

        # Now circuit should be open
        assert not health.is_available(), "Circuit should be open after threshold"

    @pytest.mark.fuzz
    @given(st.integers(min_value=1, max_value=500_000))
    def test_context_size_learning(self, context_size):
        """Context size learning should track successful sizes."""
        health = BackendHealth(name="test")

        # Record success with context
        health.record_success(context_size)

        # Should track max successful context
        assert health.max_successful_context == context_size
        assert health.consecutive_failures == 0

    @pytest.mark.fuzz
    @given(
        st.integers(min_value=1, max_value=100_000),
        st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    def test_context_reduction_on_failure(self, failed_size, reduction_factor):
        """Context size should be reduced after timeout failures."""
        health = BackendHealth(name="test", context_reduction_factor=reduction_factor)

        initial_estimate = health.safe_context_estimate
        health.record_failure("timeout", failed_size)

        # Safe estimate should be reduced
        expected_new = int(failed_size * reduction_factor)
        assert health.safe_context_estimate <= initial_estimate
        assert health.safe_context_estimate <= expected_new or health.safe_context_estimate == initial_estimate

    @pytest.mark.fuzz
    @given(st.integers(min_value=0, max_value=20))
    def test_success_resets_circuit(self, num_failures):
        """Recording success should reset the circuit."""
        health = BackendHealth(name="test", failure_threshold=3)

        # Cause failures
        for _ in range(num_failures):
            health.record_failure("error")

        # Reset with success
        health.record_success(1000)

        assert health.consecutive_failures == 0
        assert health.is_available()
        assert health.circuit_open_until == 0.0


class TestConfigDefaults:
    """Test that Config defaults are sensible."""

    def test_config_has_required_fields(self):
        """Config should have all required fields with sensible defaults."""
        cfg = Config()

        # Backend settings
        assert cfg.backend in {"ollama", "llamacpp"}
        assert cfg.ollama_base.startswith("http")
        assert cfg.llamacpp_base.startswith("http")

        # Model configs exist
        assert cfg.model_quick is not None
        assert cfg.model_coder is not None
        assert cfg.model_moe is not None
        assert cfg.model_thinking is not None

        # Task sets are non-empty
        assert len(cfg.moe_tasks) > 0
        assert len(cfg.coder_tasks) > 0
        assert len(cfg.thinking_tasks) > 0

        # Thresholds are reasonable
        assert cfg.large_content_threshold > 0
        assert cfg.medium_content_threshold > 0
        assert cfg.large_content_threshold > cfg.medium_content_threshold

    @pytest.mark.fuzz
    @given(st.sampled_from(["ollama", "llamacpp", "unknown"]))
    def test_get_backend_type(self, backend):
        """get_backend_type should return valid type."""
        cfg = Config()
        result = cfg.get_backend_type(backend)
        assert result in {"local", "remote"}

    def test_model_config_properties(self):
        """ModelConfig properties should work correctly."""
        cfg = Config()

        for model in [cfg.model_quick, cfg.model_coder, cfg.model_moe, cfg.model_thinking]:
            assert model.max_input_bytes == model.max_input_kb * 1024
            assert model.vram_gb > 0
            assert model.context_tokens > 0
            assert model.num_ctx > 0
