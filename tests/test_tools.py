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

"""Tests for delia.tools module - agentic tool use for local LLMs."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from delia.tools import (
    ToolDefinition,
    ToolRegistry,
    ParsedToolCall,
    parse_tool_calls,
    execute_tool,
    ToolResult,
    get_default_tools,
    run_agent_loop,
    AgentConfig,
    AgentResult,
)
from delia.tools.parser import has_tool_calls, format_tool_result
from delia.tools.executor import validate_path, truncate_output


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()

        async def dummy_handler(x: int) -> str:
            return str(x * 2)

        tool = ToolDefinition(
            name="double",
            description="Double a number",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            handler=dummy_handler,
        )
        registry.register(tool)

        assert "double" in registry
        assert registry.get("double") == tool
        assert registry.get("nonexistent") is None

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        registry = ToolRegistry()

        async def handler() -> str:
            return "ok"

        tool = ToolDefinition(
            name="test",
            description="Test",
            parameters={},
            handler=handler,
        )
        registry.register(tool)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool)

    def test_list_tools(self):
        """Test listing registered tools."""
        registry = ToolRegistry()

        async def handler() -> str:
            return "ok"

        registry.register(ToolDefinition("a", "A tool", {}, handler))
        registry.register(ToolDefinition("b", "B tool", {}, handler))

        tools = registry.list_tools()
        assert "a" in tools
        assert "b" in tools
        assert len(tools) == 2

    def test_to_openai_schema(self):
        """Test OpenAI schema generation."""
        async def handler() -> str:
            return "ok"

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
            handler=handler,
        )

        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert "properties" in schema["function"]["parameters"]

    def test_get_openai_schemas(self):
        """Test getting all schemas in OpenAI format."""
        registry = get_default_tools()
        schemas = registry.get_openai_schemas()

        # All tools: read_file, list_directory, search_code, web_fetch, web_search, web_news,
        # write_file, delete_file, shell_exec
        assert len(schemas) == 9
        assert all(s["type"] == "function" for s in schemas)

    def test_filter_registry(self):
        """Test filtering registry to subset of tools."""
        registry = get_default_tools()
        filtered = registry.filter(["read_file", "search_code"])

        assert len(filtered) == 2
        assert "read_file" in filtered
        assert "search_code" in filtered
        assert "list_directory" not in filtered


class TestToolParser:
    """Tests for tool call parsing."""

    def test_parse_text_format(self):
        """Test parsing XML-based tool call format."""
        text = '''Let me read that file.
<tool_call>
{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}
</tool_call>
'''
        calls = parse_tool_calls(text, native_mode=False)

        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments["path"] == "/tmp/test.txt"

    def test_parse_multiple_tool_calls(self):
        """Test parsing multiple tool calls."""
        text = '''<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>
<tool_call>{"name": "read_file", "arguments": {"path": "b.py"}}</tool_call>'''

        calls = parse_tool_calls(text, native_mode=False)
        assert len(calls) == 2
        assert calls[0].arguments["path"] == "a.py"
        assert calls[1].arguments["path"] == "b.py"

    def test_parse_native_format(self):
        """Test parsing OpenAI native format."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "/tmp/test.txt"}'
                        }
                    }]
                }
            }]
        }

        calls = parse_tool_calls(response, native_mode=True)
        assert len(calls) == 1
        assert calls[0].id == "call_123"
        assert calls[0].name == "read_file"
        assert calls[0].arguments["path"] == "/tmp/test.txt"

    def test_has_tool_calls_text(self):
        """Test detecting tool calls in text."""
        assert has_tool_calls("some <tool_call>...</tool_call> text")
        assert not has_tool_calls("no tools here")

    def test_parse_raw_json_fallback(self):
        """Test parsing raw JSON tool calls without XML wrapper."""
        # Model outputs raw JSON without <tool_call> tags
        text = '{"name": "list_directory", "arguments": {"path": "/home/dan"}}'
        calls = parse_tool_calls(text, native_mode=False)

        assert len(calls) == 1
        assert calls[0].name == "list_directory"
        assert calls[0].arguments["path"] == "/home/dan"

    def test_parse_raw_json_with_surrounding_text(self):
        """Test parsing raw JSON with surrounding explanation text."""
        text = '''I'll list the directory for you.
{"name": "list_directory", "arguments": {"path": "/tmp"}}
'''
        calls = parse_tool_calls(text, native_mode=False)

        assert len(calls) == 1
        assert calls[0].name == "list_directory"
        assert calls[0].arguments["path"] == "/tmp"

    def test_parse_raw_json_multiple(self):
        """Test parsing multiple raw JSON tool calls."""
        text = '''{"name": "read_file", "arguments": {"path": "a.py"}}

{"name": "read_file", "arguments": {"path": "b.py"}}'''

        calls = parse_tool_calls(text, native_mode=False)
        assert len(calls) == 2
        assert calls[0].arguments["path"] == "a.py"
        assert calls[1].arguments["path"] == "b.py"

    def test_has_tool_calls_raw_json(self):
        """Test detecting raw JSON tool calls."""
        assert has_tool_calls('{"name": "tool", "arguments": {"x": 1}}')
        assert has_tool_calls('Some text {"name": "tool", "arguments": {}}')
        # Should not match if missing required structure
        assert not has_tool_calls('{"name": "not a tool"}')
        assert not has_tool_calls('{"arguments": {"x": 1}}')

    def test_xml_format_takes_precedence(self):
        """Test that XML format is preferred over raw JSON."""
        # When both formats are present, XML should be used
        text = '''<tool_call>{"name": "xml_tool", "arguments": {"source": "xml"}}</tool_call>
{"name": "json_tool", "arguments": {"source": "json"}}'''

        calls = parse_tool_calls(text, native_mode=False)
        # Should only parse the XML one (fallback not triggered)
        assert len(calls) == 1
        assert calls[0].name == "xml_tool"

    def test_has_tool_calls_native(self):
        """Test detecting tool calls in native format."""
        assert has_tool_calls({"tool_calls": [{}]})
        assert has_tool_calls({"choices": [{"message": {"tool_calls": [{}]}}]})
        assert not has_tool_calls({"choices": [{"message": {"content": "hi"}}]})

    def test_format_tool_result(self):
        """Test formatting tool result for conversation."""
        result = format_tool_result("call_123", "read_file", "file contents")

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "read_file"
        assert result["content"] == "file contents"


class TestToolExecutor:
    """Tests for tool execution."""

    def test_validate_path_normal(self):
        """Test path validation for normal paths."""
        valid, error = validate_path("/tmp/test.txt")
        assert valid
        assert error == ""

    def test_validate_path_traversal(self):
        """Test path validation blocks traversal."""
        valid, error = validate_path("../../../etc/passwd")
        assert not valid
        assert "traversal" in error.lower()

    def test_validate_path_blocked(self):
        """Test path validation blocks sensitive paths."""
        valid, error = validate_path("~/.ssh/id_rsa")
        assert not valid
        assert "not allowed" in error

    def test_truncate_output_short(self):
        """Test truncation doesn't affect short output."""
        output, truncated = truncate_output("hello", max_size=100)
        assert output == "hello"
        assert not truncated

    def test_truncate_output_long(self):
        """Test truncation of long output."""
        output, truncated = truncate_output("a" * 200, max_size=100)
        assert len(output) < 200
        assert truncated
        assert "truncated" in output.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()

        async def echo(msg: str) -> str:
            return f"echo: {msg}"

        registry.register(ToolDefinition(
            name="echo",
            description="Echo a message",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            handler=echo,
        ))

        call = ParsedToolCall(id="call_1", name="echo", arguments={"msg": "hello"})
        result = await execute_tool(call, registry)

        assert result.success
        assert "echo: hello" in result.output
        assert result.tool_name == "echo"

    @pytest.mark.asyncio
    async def test_execute_tool_unknown(self):
        """Test execution of unknown tool."""
        registry = ToolRegistry()
        call = ParsedToolCall(id="call_1", name="nonexistent", arguments={})

        result = await execute_tool(call, registry)

        assert not result.success
        assert "Unknown tool" in result.output


class TestBuiltinTools:
    """Tests for built-in tools."""

    def test_default_tools_registered(self):
        """Test that default tools are properly registered."""
        registry = get_default_tools()

        # Read-only tools
        assert "read_file" in registry
        assert "list_directory" in registry
        assert "search_code" in registry
        assert "web_fetch" in registry
        assert "web_search" in registry
        assert "web_news" in registry

        # Dangerous tools (always registered, but require confirmation)
        assert "write_file" in registry
        assert "delete_file" in registry
        assert "shell_exec" in registry

        # Verify dangerous tools are marked as such
        assert registry.get("write_file").dangerous is True
        assert registry.get("delete_file").dangerous is True
        assert registry.get("shell_exec").dangerous is True

    @pytest.mark.asyncio
    async def test_read_file_success(self):
        """Test reading a file."""
        from delia.tools.builtins import read_file

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            path = f.name

        try:
            result = await read_file(path)
            assert "line 1" in result
            assert "line 2" in result
            assert "1│" in result or "1|" in result  # Line number
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        """Test reading non-existent file."""
        from delia.tools.builtins import read_file

        result = await read_file("/nonexistent/path/file.txt")
        assert "Error" in result
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_list_directory(self):
        """Test listing a directory."""
        from delia.tools.builtins import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            (Path(tmpdir) / "test.py").write_text("# test")
            (Path(tmpdir) / "data.json").write_text("{}")

            result = await list_directory(tmpdir)
            assert "test.py" in result
            assert "data.json" in result

    @pytest.mark.asyncio
    async def test_list_directory_with_pattern(self):
        """Test listing directory with glob pattern."""
        from delia.tools.builtins import list_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("# test")
            (Path(tmpdir) / "data.json").write_text("{}")

            result = await list_directory(tmpdir, pattern="*.py")
            assert "test.py" in result
            assert "data.json" not in result

    @pytest.mark.asyncio
    async def test_search_code(self):
        """Test searching code."""
        from delia.tools.builtins import search_code

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("def hello_world():\n    pass\n")

            result = await search_code("hello_world", tmpdir)
            assert "hello_world" in result


class TestAgentLoop:
    """Tests for the agentic loop."""

    @pytest.mark.asyncio
    async def test_agent_no_tools_needed(self):
        """Test agent when LLM responds without tool calls."""
        async def mock_llm(messages, system):
            return "The answer is 42."

        registry = get_default_tools()
        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="What is the answer?",
            system_prompt=None,
            registry=registry,
            model="test",
        )

        assert result.success
        assert result.response == "The answer is 42."
        assert result.iterations == 1
        assert len(result.tool_calls) == 0
        assert result.stopped_reason == "completed"

    @pytest.mark.asyncio
    async def test_agent_with_tool_call(self):
        """Test agent that makes a tool call."""
        call_count = 0

        async def mock_llm(messages, system):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: request to read a file
                return '''<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</tool_call>'''
            else:
                # Second call: provide summary based on "tool result"
                return "The file contains test data."

        # Create a test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content\n")
            path = f.name

        # Mock read_file to use our temp file
        registry = ToolRegistry()

        async def mock_read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
            return "File contents: test content"

        registry.register(ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            },
            handler=mock_read_file,
        ))

        try:
            result = await run_agent_loop(
                call_llm=mock_llm,
                prompt="Read /tmp/test.txt and summarize",
                system_prompt=None,
                registry=registry,
                model="test",
            )

            assert result.success
            assert result.iterations == 2
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].name == "read_file"
            assert "test data" in result.response
        finally:
            Path(path).unlink()

    @pytest.mark.asyncio
    async def test_agent_max_iterations(self):
        """Test agent stops at max iterations."""
        async def mock_llm(messages, system):
            # Always request another tool call
            return '''<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</tool_call>'''

        registry = ToolRegistry()

        async def mock_read_file(path: str, **kwargs) -> str:
            return "content"

        registry.register(ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=mock_read_file,
        ))

        config = AgentConfig(max_iterations=3)
        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="Keep reading files",
            system_prompt=None,
            registry=registry,
            model="test",
            config=config,
        )

        assert not result.success
        assert result.iterations == 3
        assert result.stopped_reason == "max_iterations"
        assert len(result.tool_calls) == 3

    @pytest.mark.asyncio
    async def test_agent_llm_error(self):
        """Test agent handles LLM errors."""
        async def mock_llm(messages, system):
            raise Exception("LLM unavailable")

        registry = get_default_tools()
        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="Do something",
            system_prompt=None,
            registry=registry,
            model="test",
        )

        assert not result.success
        assert result.stopped_reason == "error"
        assert "LLM" in result.response

    @pytest.mark.asyncio
    async def test_agent_with_native_tool_calling(self):
        """Test agent with OpenAI native format."""
        call_count = 0

        async def mock_llm(messages, system):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return {
                    "choices": [{
                        "message": {
                            "content": "Let me read that file.",
                            "tool_calls": [{
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path": "/tmp/test.txt"}'
                                }
                            }]
                        }
                    }]
                }
            else:
                return {"choices": [{"message": {"content": "Done reading."}}]}

        registry = ToolRegistry()

        async def mock_read_file(path: str, **kwargs) -> str:
            return "file content"

        registry.register(ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=mock_read_file,
        ))

        config = AgentConfig(native_tool_calling=True)
        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="Read the file",
            system_prompt=None,
            registry=registry,
            model="test",
            config=config,
        )

        assert result.success
        assert result.iterations == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc123"


    @pytest.mark.asyncio
    async def test_agent_auto_detects_native_support(self):
        """Test that agent auto-detects native tool calling support from backend."""
        from delia.backend_manager import BackendConfig

        # Backend that supports native tool calling
        backend = BackendConfig(
            id="test-backend",
            name="Test Backend",
            provider="llamacpp",
            type="local",
            url="http://localhost:8080",
            supports_native_tool_calling=True,
        )

        # Mock LLM that returns OpenAI format
        async def mock_llm(messages, system):
            return {
                "choices": [{
                    "message": {
                        "content": "Task completed.",
                        "tool_calls": None
                    }
                }]
            }

        registry = get_default_tools()

        # Config should detect native support is enabled
        config = AgentConfig(native_tool_calling=backend.supports_native_tool_calling)

        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="Test task",
            system_prompt=None,
            registry=registry,
            model="test-model",
            config=config,
        )

        assert result.success
        assert config.native_tool_calling is True

    @pytest.mark.asyncio
    async def test_agent_falls_back_to_text_format(self):
        """Test that agent uses XML format when backend doesn't support native."""
        from delia.backend_manager import BackendConfig

        # Backend that doesn't support native tool calling
        backend = BackendConfig(
            id="test-backend",
            name="Test Backend",
            provider="ollama",
            type="local",
            url="http://localhost:11434",
            supports_native_tool_calling=False,
        )

        call_count = 0

        async def mock_llm(messages, system):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Should use XML format
                return '''<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</tool_call>'''
            else:
                return "File processed successfully."

        registry = ToolRegistry()

        async def mock_read_file(path: str, **kwargs) -> str:
            return "file content"

        registry.register(ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=mock_read_file,
        ))

        # Config uses text format when native not supported
        config = AgentConfig(native_tool_calling=backend.supports_native_tool_calling)

        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="Read the file",
            system_prompt=None,
            registry=registry,
            model="test-model",
            config=config,
        )

        assert result.success
        assert config.native_tool_calling is False
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_llamacpp_provider_includes_tools_in_payload(self):
        """Test that llama.cpp provider includes tools array in request."""
        from unittest.mock import AsyncMock, patch
        from delia.backend_manager import BackendConfig
        import httpx

        # Backend configured for native tool calling
        backend = BackendConfig(
            id="llamacpp-local",
            name="Local llama.cpp",
            provider="llamacpp",
            type="local",
            url="http://localhost:8080",
            supports_native_tool_calling=True,
        )

        # Mock response
        mock_response_data = {
            "choices": [{
                "message": {
                    "content": "Done.",
                    "tool_calls": None
                }
            }]
        }

        # Track the request payload
        captured_payload = None

        async def mock_post(url, **kwargs):
            nonlocal captured_payload
            captured_payload = kwargs.get("json", {})

            # Create a regular mock for response
            class MockResponse:
                status_code = 200
                def json(self):
                    return mock_response_data

            return MockResponse()

        registry = get_default_tools()

        async def mock_llm(messages, system):
            # Simulate llama.cpp backend call
            client = AsyncMock()
            client.post = mock_post

            # Build request like mcp_server would
            payload = {
                "model": "test-model",
                "messages": messages,
            }

            # Add tools if native mode
            if backend.supports_native_tool_calling:
                payload["tools"] = registry.get_openai_schemas()

            response = await client.post(f"{backend.url}/v1/chat/completions", json=payload)
            return response.json()

        config = AgentConfig(native_tool_calling=backend.supports_native_tool_calling)

        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="Test native tools",
            system_prompt=None,
            registry=registry,
            model="test-model",
            config=config,
        )

        # Verify tools were included in payload
        assert captured_payload is not None
        assert "tools" in captured_payload
        # All tools: read_file, list_directory, search_code, web_fetch, web_search, web_news,
        # write_file, delete_file, shell_exec
        assert len(captured_payload["tools"]) == 9
        assert all(t["type"] == "function" for t in captured_payload["tools"])

        # Verify expected tool names
        tool_names = {t["function"]["name"] for t in captured_payload["tools"]}
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        assert "search_code" in tool_names
        assert "web_fetch" in tool_names


class TestToolPrompt:
    """Tests for tool prompt generation."""

    def test_get_tool_prompt(self):
        """Test generating tool prompt for text-based models."""
        registry = get_default_tools()
        prompt = registry.get_tool_prompt()

        assert "read_file" in prompt
        assert "list_directory" in prompt
        assert "<tool_call>" in prompt
        assert "arguments" in prompt
