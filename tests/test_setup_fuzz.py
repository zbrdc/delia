"""
Fuzz tests for the user setup process.

Tests edge cases in:
- Environment variable parsing
- settings.json configuration loading
- Backend initialization with malformed data
- First-run user experience scenarios
"""
import pytest
import os
import json
import tempfile
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, Phase
from unittest.mock import patch, MagicMock


from delia.config import Config, BackendHealth, parse_model_name, detect_model_tier
from delia.backend_manager import BackendConfig, BackendManager, SETTINGS_FILE

# Characters valid for environment variables (no null bytes, no surrogates)
# Surrogates (\ud800-\udfff) are not valid UTF-8 and cannot be set in env vars
ENV_SAFE_CHARS = st.characters(
    blacklist_characters='\x00',
    blacklist_categories=('Cs',)  # Surrogate category
)


class TestEnvironmentVariableParsing:
    """Fuzz tests for environment variable handling in Config."""

    @pytest.mark.fuzz
    @given(st.text(alphabet=ENV_SAFE_CHARS, max_size=200))
    @settings(max_examples=100)
    def test_delia_backend_env_never_crashes(self, value):
        """DELIA_BACKEND env var should handle any valid UTF-8 string."""
        with patch.dict(os.environ, {"DELIA_BACKEND": value}, clear=False):
            cfg = Config()
            # Should always produce a string
            assert isinstance(cfg.backend, str)
            # Should be the env value
            assert cfg.backend == value

    @pytest.mark.fuzz
    @given(st.text(alphabet=ENV_SAFE_CHARS, max_size=500))
    @settings(max_examples=100)
    def test_ollama_base_env_never_crashes(self, value):
        """OLLAMA_BASE env var should handle any valid UTF-8 string."""
        with patch.dict(os.environ, {"OLLAMA_BASE": value}, clear=False):
            cfg = Config()
            assert isinstance(cfg.ollama_base, str)

    @pytest.mark.fuzz
    @given(st.text(alphabet=ENV_SAFE_CHARS, max_size=500))
    @settings(max_examples=100)
    def test_llamacpp_base_env_never_crashes(self, value):
        """LLAMACPP_BASE env var should handle any valid UTF-8 string."""
        with patch.dict(os.environ, {"LLAMACPP_BASE": value}, clear=False):
            cfg = Config()
            assert isinstance(cfg.llamacpp_base, str)

    @pytest.mark.fuzz
    @given(st.text(alphabet=ENV_SAFE_CHARS, max_size=100))
    @settings(max_examples=50)
    def test_llamacpp_ctx_size_env_handles_non_integers(self, value):
        """LLAMACPP_CTX_SIZE should handle non-integer strings gracefully."""
        with patch.dict(os.environ, {"LLAMACPP_CTX_SIZE": value}, clear=False):
            try:
                cfg = Config()
                assert isinstance(cfg.llamacpp_context_tokens, int)
            except ValueError:
                # Expected for non-numeric values
                pass

    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("YES", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("", False),
        ("random", False),
        ("truthy", False),  # Not exact match
    ])
    def test_auth_enabled_boolean_parsing(self, value, expected):
        """DELIA_AUTH_ENABLED should parse boolean strings correctly."""
        with patch.dict(os.environ, {"DELIA_AUTH_ENABLED": value}, clear=False):
            cfg = Config()
            assert cfg.auth_enabled == expected

    @pytest.mark.parametrize("value,expected", [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ])
    def test_tracking_enabled_boolean_parsing(self, value, expected):
        """DELIA_TRACKING_ENABLED should parse boolean strings correctly."""
        with patch.dict(os.environ, {"DELIA_TRACKING_ENABLED": value}, clear=False):
            cfg = Config()
            assert cfg.tracking_enabled == expected

    @pytest.mark.fuzz
    @given(st.text(alphabet=ENV_SAFE_CHARS, max_size=50))
    @settings(max_examples=50)
    def test_max_concurrent_env_handles_non_integers(self, value):
        """DELIA_MAX_CONCURRENT should handle non-integer strings."""
        with patch.dict(os.environ, {"DELIA_MAX_CONCURRENT": value}, clear=False):
            try:
                cfg = Config()
                assert isinstance(cfg.max_concurrent_requests_per_backend, int)
            except ValueError:
                # Expected for non-numeric values
                pass

    @pytest.mark.parametrize("env_var", [
        "DELIA_BACKEND",
        "OLLAMA_BASE",
        "LLAMACPP_BASE",
        "OLLAMA_TYPE",
        "LLAMACPP_TYPE",
        "THINKING_MODEL",
    ])
    def test_missing_env_vars_use_defaults(self, env_var):
        """Missing env vars should use sensible defaults."""
        # Ensure the env var is not set
        env_copy = os.environ.copy()
        if env_var in env_copy:
            del env_copy[env_var]

        with patch.dict(os.environ, env_copy, clear=True):
            cfg = Config()
            # Should not raise and should have a value
            attr_map = {
                "DELIA_BACKEND": "backend",
                "OLLAMA_BASE": "ollama_base",
                "LLAMACPP_BASE": "llamacpp_base",
                "OLLAMA_TYPE": "ollama_type",
                "LLAMACPP_TYPE": "llamacpp_type",
                "THINKING_MODEL": None,  # Nested in model_thinking
            }
            if attr_map.get(env_var):
                assert getattr(cfg, attr_map[env_var]) is not None


class TestSettingsJsonParsing:
    """Fuzz tests for settings.json parsing."""

    def test_missing_settings_file_creates_defaults(self, tmp_path):
        """Missing settings.json should initialize with defaults."""
        fake_settings = tmp_path / "settings.json"
        manager = BackendManager(settings_file=fake_settings)
        # Settings loaded automatically in __init__
        # Should have empty backends or defaults
        assert isinstance(manager.backends, dict)

    def test_empty_settings_file(self, tmp_path):
        """Empty settings.json should be handled gracefully."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text("")
        manager = BackendManager(settings_file=fake_settings)
        # Should handle empty file and create defaults
        assert isinstance(manager.backends, dict)

    def test_invalid_json_in_settings(self, tmp_path):
        """Invalid JSON should be handled gracefully."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text("{ invalid json }")
        manager = BackendManager(settings_file=fake_settings)
        # Should handle invalid JSON and create defaults
        assert isinstance(manager.backends, dict)

    def test_arbitrary_file_content_non_hypothesis(self, tmp_path):
        """Arbitrary file content should not crash the manager."""
        test_contents = [
            "",
            "null",
            "[]",
            "{}",
            "not json at all",
            '{"backends": "wrong type"}',
            '{"backends": null}',
            '{"backends": [null]}',
            '{"backends": [{}]}',
            '{"backends": [{"id": "test"}]}',
        ]
        for content in test_contents:
            fake_settings = tmp_path / "settings.json"
            fake_settings.write_text(content)
            manager = BackendManager(settings_file=fake_settings)
            # Should never crash, just handle gracefully
            assert isinstance(manager.backends, dict)

    def test_settings_with_null_backends(self, tmp_path):
        """settings.json with null backends should be handled."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text(json.dumps({"backends": None}))
        manager = BackendManager(settings_file=fake_settings)
        # Should handle null backends gracefully
        assert isinstance(manager.backends, dict)

    def test_settings_with_empty_backends_list(self, tmp_path):
        """settings.json with empty backends list should work."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text(json.dumps({"backends": []}))
        manager = BackendManager(settings_file=fake_settings)
        assert len(manager.backends) == 0

    @pytest.mark.parametrize("malformed", [
        {"backends": "not a list"},
        {"backends": 123},
        {"backends": [None]},
        {"backends": [{"id": None}]},
        {"backends": [{"id": "test", "url": None}]},
        {"backends": [{"id": "test", "enabled": "not bool"}]},
        {"backends": [{"id": "test", "priority": "not int"}]},
        {"backends": [{"id": "test", "timeout_seconds": "not float"}]},
    ])
    def test_malformed_backend_entries(self, malformed, tmp_path):
        """Malformed backend entries should be handled gracefully."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text(json.dumps(malformed))
        manager = BackendManager(settings_file=fake_settings)
        try:
            manager.load_settings()
        except (TypeError, ValueError, AttributeError):
            pass  # Expected for some malformed inputs
        # Should still have a valid backends dict
        assert isinstance(manager.backends, dict)


class TestBackendConfigFromDict:
    """Fuzz tests for BackendConfig.from_dict()."""

    def test_minimal_dict(self):
        """Minimal dict should work with defaults."""
        cfg = BackendConfig.from_dict({})
        assert cfg.id == "unknown"
        assert cfg.name == "Unknown Backend"
        assert cfg.enabled == True

    @pytest.mark.fuzz
    @given(st.dictionaries(
        keys=st.text(max_size=20),
        values=st.one_of(
            st.text(max_size=100),
            st.integers(),
            st.floats(allow_nan=False),
            st.booleans(),
            st.none(),
        ),
        max_size=10
    ))
    @settings(max_examples=100)
    def test_arbitrary_dict_never_crashes(self, data):
        """Arbitrary dict should not crash from_dict."""
        try:
            cfg = BackendConfig.from_dict(data)
            assert cfg is not None
        except (TypeError, ValueError):
            pass  # Some combinations may raise

    @pytest.mark.parametrize("field,bad_value", [
        ("enabled", "yes"),  # String instead of bool
        ("priority", "high"),  # String instead of int
        ("timeout_seconds", "300"),  # String instead of float
        ("models", "not a dict"),  # String instead of dict
        ("context_limit", 1e100),  # Very large number
    ])
    def test_type_coercion_edge_cases(self, field, bad_value):
        """Fields with wrong types should be handled."""
        data = {"id": "test", field: bad_value}
        try:
            cfg = BackendConfig.from_dict(data)
            # If it succeeds, the field should have some value
            assert hasattr(cfg, field)
        except (TypeError, ValueError):
            pass  # Expected for incompatible types


class TestBackendConfigToDict:
    """Tests for BackendConfig serialization."""

    def test_roundtrip(self):
        """to_dict -> from_dict should preserve values."""
        original = BackendConfig(
            id="test-backend",
            name="Test Backend",
            provider="ollama",
            type="local",
            url="http://localhost:11434",
            enabled=True,
            priority=10,
            models={"quick": "qwen3:14b"},
            context_limit=8192,
        )
        data = original.to_dict()
        restored = BackendConfig.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.provider == original.provider
        assert restored.url == original.url
        assert restored.enabled == original.enabled
        assert restored.priority == original.priority
        assert restored.models == original.models

    def test_to_dict_excludes_runtime_state(self):
        """to_dict should not include runtime state like _client."""
        cfg = BackendConfig(
            id="test",
            name="Test",
            provider="ollama",
            type="local",
            url="http://localhost:11434",
        )
        data = cfg.to_dict()
        assert "_client" not in data
        assert "_available" not in data


class TestFirstRunScenarios:
    """Test scenarios a user might encounter on first run."""

    def test_no_ollama_running(self, tmp_path):
        """When Ollama isn't running, should handle gracefully."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text(json.dumps({
            "backends": [{
                "id": "local-ollama",
                "name": "Local Ollama",
                "provider": "ollama",
                "type": "local",
                "url": "http://localhost:99999",  # Invalid port
                "enabled": True,
            }]
        }))
        manager = BackendManager(settings_file=fake_settings)
        # Settings loaded automatically in __init__

        backend = manager.get_backend("local-ollama")
        assert backend is not None
        # Backend exists but won't be available

    def test_url_without_protocol(self, tmp_path):
        """URL without http:// prefix should be handled."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text(json.dumps({
            "backends": [{
                "id": "test",
                "name": "Test",
                "provider": "ollama",
                "type": "local",
                "url": "localhost:11434",  # Missing http://
            }]
        }))
        manager = BackendManager(settings_file=fake_settings)
        # Settings loaded automatically in __init__

        backend = manager.get_backend("test")
        assert backend is not None
        assert backend.url == "localhost:11434"

    def test_unicode_in_backend_name(self, tmp_path):
        """Unicode characters in backend name should work."""
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text(json.dumps({
            "backends": [{
                "id": "local-🍉",
                "name": "Local Watermelon 🍉",
                "provider": "ollama",
                "type": "local",
                "url": "http://localhost:11434",
            }]
        }))
        manager = BackendManager(settings_file=fake_settings)
        # Settings loaded automatically in __init__

        backend = manager.get_backend("local-🍉")
        assert backend is not None
        assert "🍉" in backend.name

    def test_very_long_url(self, tmp_path):
        """Very long URL should be handled."""
        long_url = "http://localhost:11434/" + "a" * 10000
        fake_settings = tmp_path / "settings.json"
        fake_settings.write_text(json.dumps({
            "backends": [{
                "id": "test",
                "name": "Test",
                "provider": "ollama",
                "type": "local",
                "url": long_url,
            }]
        }))
        manager = BackendManager(settings_file=fake_settings)
        # Settings loaded automatically in __init__

        backend = manager.get_backend("test")
        assert backend is not None
        assert len(backend.url) > 10000


class TestCircuitBreakerEdgeCases:
    """Edge cases for the circuit breaker pattern."""

    def test_negative_failure_threshold(self):
        """Negative failure threshold should be handled."""
        health = BackendHealth(name="test", failure_threshold=-1)
        # Should still function
        health.record_failure("timeout")
        # With negative threshold, circuit opens immediately
        assert not health.is_available()

    def test_zero_cooldown(self):
        """Zero cooldown should recover immediately."""
        health = BackendHealth(
            name="test",
            failure_threshold=1,
            base_cooldown_seconds=0.0
        )
        health.record_failure("timeout")
        # Should be available immediately with zero cooldown
        import time
        time.sleep(0.01)
        assert health.is_available()

    def test_very_large_context_size(self):
        """Very large context sizes should be handled."""
        health = BackendHealth(name="test")
        health.record_success(10**12)  # 1 trillion bytes
        assert health.max_successful_context == 10**12

    @pytest.mark.fuzz
    @given(st.integers(min_value=0, max_value=10**9))
    @settings(max_examples=50)
    def test_arbitrary_context_sizes(self, size):
        """Arbitrary context sizes should be handled."""
        health = BackendHealth(name="test")
        health.record_success(size)
        assert health.max_successful_context == size
        assert health.consecutive_failures == 0


class TestModelParsingSetupEdgeCases:
    """Edge cases users might encounter when specifying models."""

    @pytest.mark.parametrize("model_name", [
        "",  # Empty string
        " ",  # Just whitespace
        ":",  # Just colon
        ":14b",  # Missing name before colon
        "model:",  # Missing tag after colon
        "model:14b:extra:colons",  # Multiple colons
        "/path/to/model.gguf",  # File path
        "http://example.com/model",  # URL
        "registry.example.com/model:tag",  # Registry path
    ])
    def test_unusual_model_name_formats(self, model_name):
        """Unusual model name formats should not crash."""
        result = parse_model_name(model_name)
        assert result is not None
        assert result.raw_name == model_name

    @pytest.mark.parametrize("model_name,expected_tier", [
        # 14B generic model → quick tier (< 15B threshold)
        ("qwen3:14b", "quick"),
        # 70B large model → moe tier (>= 30B)
        ("llama3:70b", "moe"),
        # Coder-specialized model → coder tier (has "coder" keyword)
        ("qwen2.5-coder:7b", "coder"),
        # MoE pattern → moe tier
        ("mixtral-8x7b", "moe"),
        # Small generic model → quick tier
        ("tiny-model-1b", "quick"),
        # 15B model → coder tier (>= 15B threshold)
        ("model-15b", "coder"),
        # 30B model → moe tier (>= 30B threshold)
        ("model-30b", "moe"),
    ])
    def test_tier_detection_common_models(self, model_name, expected_tier):
        """Common model patterns should detect correct tier based on params and keywords."""
        result = detect_model_tier(model_name)
        assert result == expected_tier


@pytest.fixture
def tmp_path(tmp_path_factory):
    """Create a temporary directory for each test."""
    return tmp_path_factory.mktemp("delia_test")
