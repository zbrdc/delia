# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Comprehensive tests for task workflow module."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["DELIA_DATA_DIR"] = "/tmp/delia-test-workflow"


class TestWorkflowNode:
    """Tests for WorkflowNode dataclass."""

    def test_from_dict_minimal(self):
        from delia.task_workflow import WorkflowNode
        node = WorkflowNode.from_dict({
            "id": "test",
            "task": "quick",
            "content": "Hello",
        })
        assert node.id == "test"
        assert node.depends_on is None
        assert node.retry_count == 0

    def test_from_dict_full(self):
        from delia.task_workflow import WorkflowNode
        node = WorkflowNode.from_dict({
            "id": "test",
            "task": "plan",
            "content": "Content",
            "depends_on": ["other"],
            "on_success": "next",
            "on_failure": "fallback",
            "retry_count": 3,
            "backoff_factor": 2.0,
        })
        assert node.depends_on == ["other"]
        assert node.on_success == "next"
        assert node.retry_count == 3


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition dataclass."""

    def test_from_dict(self):
        from delia.task_workflow import WorkflowDefinition
        defn = WorkflowDefinition.from_dict({
            "name": "Test Workflow",
            "entry": "start",
            "timeout_minutes": 15,
            "nodes": [
                {"id": "start", "task": "quick", "content": "Begin"},
            ],
        })
        assert defn.name == "Test Workflow"
        assert defn.entry == "start"
        assert defn.timeout_minutes == 15
        assert len(defn.nodes) == 1


class TestCycleDetection:
    """Tests for cycle detection."""

    def test_no_cycle(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, detect_cycles
        defn = WorkflowDefinition(
            name="Test",
            entry="a",
            nodes=[
                WorkflowNode(id="a", task="quick", content="", on_success="b"),
                WorkflowNode(id="b", task="quick", content="", on_success="c"),
                WorkflowNode(id="c", task="quick", content=""),
            ],
        )

        cycle = detect_cycles(defn)
        assert cycle is None

    def test_simple_cycle(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, detect_cycles
        defn = WorkflowDefinition(
            name="Test",
            entry="a",
            nodes=[
                WorkflowNode(id="a", task="quick", content="", on_success="b"),
                WorkflowNode(id="b", task="quick", content="", on_success="a"),  # Cycle!
            ],
        )

        cycle = detect_cycles(defn)
        assert cycle is not None
        assert "a" in cycle and "b" in cycle

    def test_self_cycle(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, detect_cycles
        defn = WorkflowDefinition(
            name="Test",
            entry="a",
            nodes=[
                WorkflowNode(id="a", task="quick", content="", on_success="a"),  # Self-cycle
            ],
        )

        cycle = detect_cycles(defn)
        assert cycle is not None

    def test_depends_on_not_a_cycle(self):
        """depends_on creates prerequisites, not forward edges, so no cycle."""
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, detect_cycles
        defn = WorkflowDefinition(
            name="Test",
            entry="a",
            nodes=[
                WorkflowNode(id="a", task="quick", content="", depends_on=["b"]),
                WorkflowNode(id="b", task="quick", content="", depends_on=["a"]),
            ],
        )

        # This should not be detected as cycle since depends_on is not an edge
        # However, it's invalid for other reasons (circular dependencies)
        cycle = detect_cycles(defn)
        # detect_cycles only checks on_success/on_failure, not depends_on
        assert cycle is None


class TestValidateWorkflow:
    """Tests for workflow validation."""

    def test_valid_workflow(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, validate_workflow
        defn = WorkflowDefinition(
            name="Test",
            entry="start",
            nodes=[
                WorkflowNode(id="start", task="quick", content="Begin"),
            ],
        )
        errors = validate_workflow(defn)
        assert len(errors) == 0

    def test_missing_entry(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, validate_workflow
        defn = WorkflowDefinition(
            name="Test",
            entry="",
            nodes=[
                WorkflowNode(id="start", task="quick", content="Begin"),
            ],
        )
        errors = validate_workflow(defn)
        assert any("Missing entry" in e for e in errors)

    def test_entry_not_found(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, validate_workflow
        defn = WorkflowDefinition(
            name="Test",
            entry="nonexistent",
            nodes=[
                WorkflowNode(id="start", task="quick", content="Begin"),
            ],
        )
        errors = validate_workflow(defn)
        assert any("not found" in e for e in errors)

    def test_duplicate_ids(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, validate_workflow
        defn = WorkflowDefinition(
            name="Test",
            entry="start",
            nodes=[
                WorkflowNode(id="start", task="quick", content="A"),
                WorkflowNode(id="start", task="quick", content="B"),
            ],
        )
        errors = validate_workflow(defn)
        assert any("Duplicate" in e for e in errors)

    def test_invalid_depends_on(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, validate_workflow
        defn = WorkflowDefinition(
            name="Test",
            entry="start",
            nodes=[
                WorkflowNode(id="start", task="quick", content="", depends_on=["missing"]),
            ],
        )
        errors = validate_workflow(defn)
        assert any("unknown node" in e for e in errors)

    def test_cycle_validation(self):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, validate_workflow
        defn = WorkflowDefinition(
            name="Test",
            entry="a",
            nodes=[
                WorkflowNode(id="a", task="quick", content="", on_success="b"),
                WorkflowNode(id="b", task="quick", content="", on_success="a"),
            ],
        )
        errors = validate_workflow(defn)
        assert any("Cycle" in e for e in errors)


class TestParseWorkflowDefinition:
    """Tests for JSON parsing."""

    def test_parse_valid(self):
        from delia.task_workflow import parse_workflow_definition
        defn = parse_workflow_definition(json.dumps({
            "name": "Test",
            "entry": "start",
            "nodes": [
                {"id": "start", "task": "quick", "content": "Begin"},
            ],
        }))
        assert defn.name == "Test"

    def test_parse_invalid_json(self):
        from delia.task_workflow import parse_workflow_definition
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_workflow_definition("not json")

    def test_parse_validation_error(self):
        from delia.task_workflow import parse_workflow_definition
        with pytest.raises(ValueError, match="validation failed"):
            parse_workflow_definition(json.dumps({
                "name": "Test",
                "entry": "nonexistent",
                "nodes": [{"id": "start", "task": "quick", "content": ""}],
            }))


class TestExecuteWorkflow:
    """Tests for workflow execution."""

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock()
        ctx.select_model = AsyncMock(return_value="test-model")
        ctx.get_active_backend = MagicMock(return_value=MagicMock())
        ctx.call_llm = AsyncMock(return_value={"response": "Mock", "tokens": 10})
        ctx.get_client_id = MagicMock(return_value=None)
        ctx.tracker = None
        return ctx

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self, mock_ctx):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.return_value = "Node output"

            defn = WorkflowDefinition(
                name="Test",
                entry="start",
                nodes=[WorkflowNode(id="start", task="quick", content="Begin")],
            )
            result = await execute_workflow(defn, mock_ctx)

            assert result.success is True
            assert "start" in result.nodes_completed

    @pytest.mark.asyncio
    async def test_execute_with_branching(self, mock_ctx):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.side_effect = ["First", "Second"]

            defn = WorkflowDefinition(
                name="Test",
                entry="a",
                nodes=[
                    WorkflowNode(id="a", task="quick", content="A", on_success="b"),
                    WorkflowNode(id="b", task="quick", content="B"),
                ],
            )
            result = await execute_workflow(defn, mock_ctx)

            assert result.success is True
            assert result.nodes_completed == ["a", "b"]

    @pytest.mark.asyncio
    async def test_execute_with_failure_fallback(self, mock_ctx):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.side_effect = [Exception("Error"), "Fallback success"]

            defn = WorkflowDefinition(
                name="Test",
                entry="main",
                nodes=[
                    WorkflowNode(id="main", task="quick", content="Fail", on_failure="fallback"),
                    WorkflowNode(id="fallback", task="quick", content="Recover"),
                ],
            )
            result = await execute_workflow(defn, mock_ctx)

            assert "main" in result.nodes_failed
            assert "fallback" in result.nodes_completed

    @pytest.mark.asyncio
    async def test_execute_with_retry(self, mock_ctx):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            # Fail twice, succeed third time
            mock.side_effect = [Exception("Fail 1"), Exception("Fail 2"), "Success"]

            defn = WorkflowDefinition(
                name="Test",
                entry="start",
                nodes=[
                    WorkflowNode(id="start", task="quick", content="Retry", retry_count=2),
                ],
            )
            result = await execute_workflow(defn, mock_ctx)

            assert result.success is True
            assert result.node_results[0].retries == 2

    @pytest.mark.asyncio
    async def test_skipped_nodes(self, mock_ctx):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.return_value = "Done"

            defn = WorkflowDefinition(
                name="Test",
                entry="a",
                nodes=[
                    WorkflowNode(id="a", task="quick", content="A"),  # No on_success
                    WorkflowNode(id="b", task="quick", content="B"),  # Never reached
                ],
            )
            result = await execute_workflow(defn, mock_ctx)

            assert "a" in result.nodes_completed
            assert "b" in result.nodes_skipped

    @pytest.mark.asyncio
    async def test_variable_substitution(self, mock_ctx):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.side_effect = ["Generated code", "Review complete"]

            defn = WorkflowDefinition(
                name="Test",
                entry="gen",
                nodes=[
                    WorkflowNode(id="gen", task="generate", content="Write code", output_var="code", on_success="review"),
                    WorkflowNode(id="review", task="review", content="Review ${code}"),
                ],
            )
            result = await execute_workflow(defn, mock_ctx)

            assert result.success is True
            # Verify second call had substituted content
            calls = mock.call_args_list
            second_call_content = calls[1].kwargs.get('content')
            assert "Generated code" in second_call_content

    @pytest.mark.asyncio
    async def test_dependency_satisfaction(self, mock_ctx):
        """Test that nodes with depends_on wait for dependencies."""
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.return_value = "Output"

            # Node 'c' depends on both 'a' and 'b'
            # But since we execute sequentially starting from entry, this tests the check
            defn = WorkflowDefinition(
                name="Test",
                entry="c",
                nodes=[
                    WorkflowNode(id="a", task="quick", content="A"),
                    WorkflowNode(id="b", task="quick", content="B"),
                    WorkflowNode(id="c", task="quick", content="C", depends_on=["a", "b"]),
                ],
            )

            result = await execute_workflow(defn, mock_ctx)

            # Since 'c' depends on 'a' and 'b' but they haven't run, execution should fail
            assert result.success is False
            assert any("dependency_error" in str(e) for e in result.errors)

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self, mock_ctx):
        """Test workflow timeout by mocking time."""
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow
        import time as time_module

        original_time = time_module.time
        mock_time = [original_time()]  # Start with current time
        call_count = [0]

        def time_mock():
            # After first call (which happens inside first node execution),
            # advance time significantly
            call_count[0] += 1
            if call_count[0] > 2:  # After a few calls (start + first node)
                mock_time[0] += 100  # Add 100 seconds
            return mock_time[0]

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.return_value = "Output"

            with patch('delia.task_workflow.time.time', side_effect=time_mock):
                defn = WorkflowDefinition(
                    name="Test",
                    entry="a",
                    timeout_minutes=0.01,  # 0.6 seconds
                    nodes=[
                        WorkflowNode(id="a", task="quick", content="A", on_success="b"),
                        WorkflowNode(id="b", task="quick", content="B"),
                    ],
                )

                result = await execute_workflow(defn, mock_ctx)

                # Should timeout after first node
                assert result.success is False
                assert any("timeout" in str(e).lower() for e in result.errors)
                assert "a" in result.nodes_completed
                assert "b" in result.nodes_skipped

    @pytest.mark.asyncio
    async def test_output_var_stored(self, mock_ctx):
        from delia.task_workflow import WorkflowDefinition, WorkflowNode, execute_workflow

        with patch('delia.task_workflow.delegate_impl', new_callable=AsyncMock) as mock:
            mock.return_value = "Generated output"

            defn = WorkflowDefinition(
                name="Test",
                entry="gen",
                nodes=[
                    WorkflowNode(id="gen", task="generate", content="Generate", output_var="my_var"),
                ],
            )

            result = await execute_workflow(defn, mock_ctx)

            assert "gen" in result.outputs
            assert "my_var" in result.outputs
            assert result.outputs["my_var"] == "Generated output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
