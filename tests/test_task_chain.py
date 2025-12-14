# Delia - Local LLM Orchestration
# Copyright (C) 2024 Dan Yishai
# Licensed under GPL-3.0

"""Comprehensive tests for task chain module."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["DELIA_DATA_DIR"] = "/tmp/delia-test-chain"


class TestChainStep:
    """Tests for ChainStep dataclass."""

    def test_from_dict_minimal(self):
        from delia.task_chain import ChainStep
        step = ChainStep.from_dict({
            "id": "test",
            "task": "quick",
            "content": "Hello",
        })
        assert step.id == "test"
        assert step.task == "quick"
        assert step.content == "Hello"
        assert step.model is None
        assert step.output_var is None

    def test_from_dict_full(self):
        from delia.task_chain import ChainStep
        step = ChainStep.from_dict({
            "id": "test",
            "task": "generate",
            "content": "Code here",
            "model": "coder",
            "language": "python",
            "output_var": "code",
            "pass_to_next": True,
        })
        assert step.model == "coder"
        assert step.output_var == "code"
        assert step.pass_to_next is True


class TestVariableSubstitution:
    """Tests for variable substitution."""

    def test_substitute_simple(self):
        from delia.task_chain import substitute_variables
        result = substitute_variables(
            "Review: ${code}",
            {"code": "def hello(): pass"}
        )
        assert result == "Review: def hello(): pass"

    def test_substitute_multiple(self):
        from delia.task_chain import substitute_variables
        result = substitute_variables(
            "${a} + ${b} = ${c}",
            {"a": "1", "b": "2", "c": "3"}
        )
        assert result == "1 + 2 = 3"

    def test_substitute_missing_var(self):
        from delia.task_chain import substitute_variables
        result = substitute_variables(
            "Value: ${missing}",
            {}
        )
        assert result == "Value: ${missing}"  # Left as-is

    def test_substitute_no_vars(self):
        from delia.task_chain import substitute_variables
        result = substitute_variables("No variables", {"x": "y"})
        assert result == "No variables"


class TestParseChainSteps:
    """Tests for JSON parsing."""

    def test_parse_valid(self):
        from delia.task_chain import parse_chain_steps
        steps_json = json.dumps([
            {"id": "step1", "task": "quick", "content": "Hello"},
            {"id": "step2", "task": "review", "content": "World"},
        ])
        steps = parse_chain_steps(steps_json)
        assert len(steps) == 2
        assert steps[0].id == "step1"

    def test_parse_invalid_json(self):
        from delia.task_chain import parse_chain_steps
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_chain_steps("not json")

    def test_parse_not_array(self):
        from delia.task_chain import parse_chain_steps
        with pytest.raises(ValueError, match="must be a JSON array"):
            parse_chain_steps('{"key": "value"}')

    def test_parse_empty_array(self):
        from delia.task_chain import parse_chain_steps
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_chain_steps('[]')

    def test_parse_missing_id(self):
        from delia.task_chain import parse_chain_steps
        with pytest.raises(ValueError, match="missing required 'id'"):
            parse_chain_steps('[{"task": "quick", "content": "test"}]')

    def test_parse_duplicate_ids(self):
        from delia.task_chain import parse_chain_steps
        with pytest.raises(ValueError, match="Duplicate step id"):
            parse_chain_steps(json.dumps([
                {"id": "same", "task": "quick", "content": "a"},
                {"id": "same", "task": "quick", "content": "b"},
            ]))


class TestChainResult:
    """Tests for ChainResult."""

    def test_to_dict(self):
        from delia.task_chain import ChainResult, StepResult
        result = ChainResult(
            success=True,
            steps_completed=2,
            steps_total=2,
            outputs={"step1": "output1"},
            errors=[],
            step_results=[
                StepResult(step_id="step1", success=True, output="output1"),
            ],
            elapsed_ms=100,
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["steps_completed"] == 2


class TestExecuteChain:
    """Tests for chain execution."""

    @pytest.fixture
    def mock_ctx(self):
        """Create mock DelegateContext."""
        ctx = MagicMock()
        ctx.select_model = AsyncMock(return_value="test-model")
        ctx.get_active_backend = MagicMock(return_value=MagicMock())
        ctx.call_llm = AsyncMock(return_value={
            "response": "Mock response",
            "tokens": 10,
        })
        ctx.get_client_id = MagicMock(return_value=None)
        ctx.tracker = None
        return ctx

    @pytest.mark.asyncio
    async def test_execute_single_step(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.return_value = "Step output"

            steps = [ChainStep(id="step1", task="quick", content="Test")]
            result = await execute_chain(steps, mock_ctx)

            assert result.success is True
            assert result.steps_completed == 1
            assert "step1" in result.outputs

    @pytest.mark.asyncio
    async def test_execute_with_variable_substitution(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.side_effect = ["First output", "Second with First output"]

            steps = [
                ChainStep(id="step1", task="quick", content="First", output_var="first"),
                ChainStep(id="step2", task="quick", content="Use ${first}"),
            ]
            result = await execute_chain(steps, mock_ctx)

            assert result.success is True
            # Verify variable substitution happened
            calls = mock_delegate.call_args_list
            assert len(calls) == 2
            # Second call should have substituted content
            second_call_content = calls[1].kwargs.get('content') or calls[1][1].get('content')
            assert "First output" in second_call_content

    @pytest.mark.asyncio
    async def test_execute_with_error(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.side_effect = Exception("LLM error")

            steps = [ChainStep(id="step1", task="quick", content="Test")]
            result = await execute_chain(steps, mock_ctx)

            assert result.success is False
            assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_execute_continue_on_error(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.side_effect = [Exception("Error"), "Success"]

            steps = [
                ChainStep(id="step1", task="quick", content="Fail"),
                ChainStep(id="step2", task="quick", content="Pass"),
            ]
            result = await execute_chain(steps, mock_ctx, continue_on_error=True)

            assert result.success is False  # Overall failed
            assert result.steps_completed == 1  # But step2 completed

    @pytest.mark.asyncio
    async def test_execute_with_pass_to_next(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.side_effect = ["First", "Second"]

            steps = [
                ChainStep(id="step1", task="quick", content="One", pass_to_next=True),
                ChainStep(id="step2", task="quick", content="Two"),
            ]
            result = await execute_chain(steps, mock_ctx)

            # Verify pass_to_next appended output
            calls = mock_delegate.call_args_list
            assert len(calls) == 2
            # Second call should have appended previous output
            second_call_content = calls[1].kwargs.get('content') or calls[1][1].get('content')
            assert "Previous Step Output" in second_call_content
            assert "First" in second_call_content

    @pytest.mark.asyncio
    async def test_execute_stores_output_var(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.return_value = "Generated code"

            steps = [ChainStep(id="gen", task="generate", content="Write code", output_var="my_code")]
            result = await execute_chain(steps, mock_ctx)

            assert "gen" in result.outputs
            assert "my_code" in result.outputs
            assert result.outputs["my_code"] == "Generated code"

    @pytest.mark.asyncio
    async def test_execute_session_id_propagated(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.return_value = "Output"

            steps = [ChainStep(id="step1", task="quick", content="Test")]
            await execute_chain(steps, mock_ctx, session_id="session-123")

            # Verify session_id was passed to delegate_impl
            calls = mock_delegate.call_args_list
            assert calls[0].kwargs.get('session_id') == "session-123"

    @pytest.mark.asyncio
    async def test_execute_multiple_steps_sequential(self, mock_ctx):
        from delia.task_chain import ChainStep, execute_chain

        with patch('delia.task_chain.delegate_impl', new_callable=AsyncMock) as mock_delegate:
            mock_delegate.side_effect = ["Output 1", "Output 2", "Output 3"]

            steps = [
                ChainStep(id="a", task="quick", content="A"),
                ChainStep(id="b", task="quick", content="B"),
                ChainStep(id="c", task="quick", content="C"),
            ]
            result = await execute_chain(steps, mock_ctx)

            assert result.success is True
            assert result.steps_completed == 3
            assert len(result.step_results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
