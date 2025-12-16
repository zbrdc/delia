# Copyright (C) 2024 Delia Contributors
#
# Tests for hedged request functionality

"""Tests for hedged request execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestExecuteHedgedCall:
    """Test execute_hedged_call function."""

    def _create_mock_backend(
        self,
        backend_id: str,
        enabled: bool = True,
        backend_type: str = "local",
    ):
        """Create a mock BackendConfig."""
        from delia.backend_manager import BackendConfig

        return BackendConfig(
            id=backend_id,
            name=f"Test {backend_id}",
            provider="test",
            type=backend_type,
            url=f"http://{backend_id}:8080",
            enabled=enabled,
            priority=0,
            models={"quick": "test-7b", "coder": "test-14b", "moe": "test-30b"},
        )

    def _create_mock_ctx(self, call_llm_results: list[dict]):
        """Create a mock DelegateContext with configured call_llm results.

        Args:
            call_llm_results: List of result dicts that call_llm will return in order
        """
        ctx = MagicMock()
        ctx.call_llm = AsyncMock(side_effect=call_llm_results)
        return ctx

    @pytest.mark.asyncio
    async def test_single_backend_uses_regular_call(self):
        """Single backend falls back to regular execute_delegate_call."""
        from delia.delegation import execute_hedged_call

        backend = self._create_mock_backend("single")

        with patch("delia.delegation.execute_delegate_call") as mock_execute:
            mock_execute.return_value = ("response text", 100)

            result = await execute_hedged_call(
                ctx=MagicMock(),
                backends=[backend],
                selected_model="test-model",
                content="test content",
                system="system prompt",
                task_type="review",
                original_task="review",
                detected_language="python",
                delay_ms=50,
            )

            # Should return (text, tokens, backend)
            assert result == ("response text", 100, backend)
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_backends_raises_error(self):
        """Empty backends list raises ValueError."""
        from delia.delegation import execute_hedged_call

        ctx = self._create_mock_ctx([])

        with pytest.raises(ValueError, match="No backends provided"):
            await execute_hedged_call(
                ctx=ctx,
                backends=[],
                selected_model="test-model",
                content="test content",
                system="system prompt",
                task_type="review",
                original_task="review",
                detected_language="python",
            )

    @pytest.mark.asyncio
    async def test_first_success_wins(self):
        """First successful response wins and others are cancelled."""
        from delia.delegation import execute_hedged_call

        backend1 = self._create_mock_backend("fast")
        backend2 = self._create_mock_backend("slow")

        # Fast backend succeeds quickly
        call_count = 0
        async def mock_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            backend_obj = kwargs.get("backend_obj")
            if backend_obj.id == "fast":
                return {"success": True, "response": "fast response", "tokens": 50}
            else:
                # Slow backend - should be cancelled before this completes
                await asyncio.sleep(10)
                return {"success": True, "response": "slow response", "tokens": 100}

        ctx = MagicMock()
        ctx.call_llm = mock_call_llm

        with patch("delia.delegation.get_affinity_tracker"):
            result = await execute_hedged_call(
                ctx=ctx,
                backends=[backend1, backend2],
                selected_model="test-model",
                content="test content",
                system="system prompt",
                task_type="review",
                original_task="review",
                detected_language="python",
                delay_ms=10,  # Short delay for test
            )

        response_text, tokens, winning_backend = result
        assert response_text == "fast response"
        assert tokens == 50
        assert winning_backend.id == "fast"

    @pytest.mark.asyncio
    async def test_first_failure_tries_next(self):
        """If first backend fails, next successful one wins."""
        from delia.delegation import execute_hedged_call

        backend1 = self._create_mock_backend("fail")
        backend2 = self._create_mock_backend("success")

        async def mock_call_llm(*args, **kwargs):
            backend_obj = kwargs.get("backend_obj")
            if backend_obj.id == "fail":
                return {"success": False, "error": "Backend failed"}
            else:
                return {"success": True, "response": "success response", "tokens": 75}

        ctx = MagicMock()
        ctx.call_llm = mock_call_llm

        with patch("delia.delegation.get_affinity_tracker"):
            result = await execute_hedged_call(
                ctx=ctx,
                backends=[backend1, backend2],
                selected_model="test-model",
                content="test content",
                system="system prompt",
                task_type="review",
                original_task="review",
                detected_language="python",
                delay_ms=0,  # No delay
            )

        response_text, tokens, winning_backend = result
        assert response_text == "success response"
        assert tokens == 75
        assert winning_backend.id == "success"

    @pytest.mark.asyncio
    async def test_all_failures_raises_exception(self):
        """If all backends fail, raises exception with error summary."""
        from delia.delegation import execute_hedged_call

        backend1 = self._create_mock_backend("fail1")
        backend2 = self._create_mock_backend("fail2")

        async def mock_call_llm(*args, **kwargs):
            backend_obj = kwargs.get("backend_obj")
            return {"success": False, "error": f"{backend_obj.id} failed"}

        ctx = MagicMock()
        ctx.call_llm = mock_call_llm

        with patch("delia.delegation.get_affinity_tracker"):
            with pytest.raises(Exception, match="Hedged call failed"):
                await execute_hedged_call(
                    ctx=ctx,
                    backends=[backend1, backend2],
                    selected_model="test-model",
                    content="test content",
                    system="system prompt",
                    task_type="review",
                    original_task="review",
                    detected_language="python",
                    delay_ms=0,
                )

    @pytest.mark.asyncio
    async def test_affinity_recorded_for_winner_only(self):
        """Affinity is recorded only for the winning backend."""
        from delia.delegation import execute_hedged_call

        backend1 = self._create_mock_backend("fast")
        backend2 = self._create_mock_backend("slow")

        async def mock_call_llm(*args, **kwargs):
            backend_obj = kwargs.get("backend_obj")
            if backend_obj.id == "fast":
                return {"success": True, "response": "fast response", "tokens": 50}
            else:
                await asyncio.sleep(10)
                return {"success": True, "response": "slow response", "tokens": 100}

        ctx = MagicMock()
        ctx.call_llm = mock_call_llm

        mock_tracker = MagicMock()
        with patch("delia.delegation.get_affinity_tracker", return_value=mock_tracker):
            await execute_hedged_call(
                ctx=ctx,
                backends=[backend1, backend2],
                selected_model="test-model",
                content="test content",
                system="system prompt",
                task_type="review",
                original_task="review",
                detected_language="python",
                delay_ms=10,
            )

        # Only the winning backend should have affinity recorded
        calls = mock_tracker.update.call_args_list
        # Should have exactly one call for the winner with quality score
        winner_calls = [
            c for c in calls
            if c[0][0] == "fast" and c[1].get("quality", 0) > 0.5
        ]
        assert len(winner_calls) == 1

    @pytest.mark.asyncio
    async def test_thinking_tags_stripped(self):
        """Response has thinking tags stripped."""
        from delia.delegation import execute_hedged_call

        backend = self._create_mock_backend("test")

        async def mock_call_llm(*args, **kwargs):
            return {
                "success": True,
                "response": "<think>internal thoughts</think>actual response",
                "tokens": 50,
            }

        ctx = MagicMock()
        ctx.call_llm = mock_call_llm

        with patch("delia.delegation.execute_delegate_call") as mock_execute:
            mock_execute.return_value = ("actual response", 50)

            result = await execute_hedged_call(
                ctx=ctx,
                backends=[backend],
                selected_model="test-model",
                content="test content",
                system="system prompt",
                task_type="review",
                original_task="review",
                detected_language="python",
            )

        response_text, _, _ = result
        # For single backend, it delegates to execute_delegate_call which strips tags
        assert "<think>" not in response_text


class TestHedgingConfig:
    """Test hedging configuration in routing config."""

    def test_default_hedging_config_in_settings(self):
        """Default settings include hedging configuration."""
        from delia.backend_manager import BackendManager
        from pathlib import Path
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            # Create manager (will create default settings file)
            manager = BackendManager(settings_file=settings_path)

            # Read the created settings file directly
            with open(settings_path) as f:
                settings = json.load(f)

            # Check that hedging config exists in routing
            hedging = settings.get("routing", {}).get("hedging", {})
            assert "enabled" in hedging
            assert "delay_ms" in hedging
            assert "max_backends" in hedging

            # Check defaults
            assert hedging["enabled"] is False
            assert hedging["delay_ms"] == 50
            assert hedging["max_backends"] == 2


    def test_default_voting_config_in_settings(self):
        """Default settings include voting configuration (MDAP k-voting)."""
        from delia.backend_manager import BackendManager
        from pathlib import Path
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.json"
            manager = BackendManager(settings_file=settings_path)

            with open(settings_path) as f:
                settings = json.load(f)

            # Check that voting config exists in routing
            voting = settings.get("routing", {}).get("voting", {})
            assert "enabled" in voting
            assert "k" in voting
            assert "auto_kmin" in voting
            assert "max_backends" in voting
            assert "max_response_length" in voting
            assert "similarity_threshold" in voting

            # Check defaults match MDAP paper recommendations
            assert voting["enabled"] is False  # Opt-in
            assert voting["k"] == 2  # First-to-ahead-by-2
            assert voting["auto_kmin"] is True  # Auto-calculate k
            assert voting["max_backends"] == 3
            assert voting["max_response_length"] == 700  # MDAP threshold
            assert voting["similarity_threshold"] == 0.85


class TestVotingExecution:
    """Test execute_voting_call function."""

    def _create_mock_backend(self, backend_id: str):
        """Create a mock BackendConfig."""
        from delia.backend_manager import BackendConfig

        return BackendConfig(
            id=backend_id,
            name=f"Test {backend_id}",
            provider="test",
            type="local",
            url=f"http://{backend_id}:8080",
            enabled=True,
            priority=0,
            models={"quick": "test-7b", "coder": "test-14b"},
        )

    @pytest.mark.asyncio
    async def test_voting_reaches_consensus_with_k_matching(self):
        """Voting reaches consensus when k backends return same answer."""
        from delia.delegation import execute_voting_call

        backend1 = self._create_mock_backend("b1")
        backend2 = self._create_mock_backend("b2")
        backend3 = self._create_mock_backend("b3")

        async def mock_call_llm(*args, **kwargs):
            # All backends return same answer
            return {"success": True, "response": "The answer is 42", "tokens": 20}

        ctx = MagicMock()
        ctx.call_llm = mock_call_llm

        with patch("delia.delegation.get_affinity_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()

            response, tokens, winner, metadata = await execute_voting_call(
                ctx=ctx,
                backends=[backend1, backend2, backend3],
                selected_model="test-model",
                content="test content",
                system="system prompt",
                task_type="review",
                original_task="review",
                detected_language="python",
                voting_k=2,
                delay_ms=0,
            )

        assert "42" in response
        assert metadata["mode"] == "voting"
        assert metadata["k"] == 2

    @pytest.mark.asyncio
    async def test_voting_red_flags_long_responses(self):
        """Voting red-flags responses over token limit."""
        from delia.delegation import execute_voting_call

        backend1 = self._create_mock_backend("b1")
        backend2 = self._create_mock_backend("b2")
        backend3 = self._create_mock_backend("b3")

        call_count = 0

        async def mock_call_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            backend_obj = kwargs.get("backend_obj")
            if backend_obj.id == "b1":
                # First backend returns too long response (red-flagged)
                return {
                    "success": True,
                    "response": "word " * 800,  # ~3200 chars = ~800 tokens
                    "tokens": 800,
                }
            else:
                # Other backends return valid responses
                await asyncio.sleep(0.01)  # Slight delay so b1 is processed first
                return {
                    "success": True,
                    "response": "Short valid answer",
                    "tokens": 10,
                }

        ctx = MagicMock()
        ctx.call_llm = mock_call_llm

        with patch("delia.delegation.get_affinity_tracker") as mock_tracker:
            mock_tracker.return_value = MagicMock()

            response, tokens, winner, metadata = await execute_voting_call(
                ctx=ctx,
                backends=[backend1, backend2, backend3],
                selected_model="test-model",
                content="test",
                system="system",
                task_type="review",
                original_task="review",
                detected_language="python",
                voting_k=2,  # Need 2 matching votes
                delay_ms=0,
            )

        # First response should be red-flagged, other two reach consensus
        assert metadata["red_flagged"] >= 1
        assert "Short valid" in response

    @pytest.mark.asyncio
    async def test_single_backend_falls_back_to_regular_call(self):
        """Single backend uses regular execution instead of voting."""
        from delia.delegation import execute_voting_call

        backend = self._create_mock_backend("single")

        with patch("delia.delegation.execute_delegate_call") as mock_execute:
            mock_execute.return_value = ("single response", 50)

            response, tokens, winner, metadata = await execute_voting_call(
                ctx=MagicMock(),
                backends=[backend],
                selected_model="test-model",
                content="test",
                system="system",
                task_type="review",
                original_task="review",
                detected_language="python",
                voting_k=2,
            )

        mock_execute.assert_called_once()
        assert metadata["mode"] == "single"
