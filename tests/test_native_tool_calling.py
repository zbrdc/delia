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

"""Comprehensive tests for native tool calling support.

This module contains detailed tests for native OpenAI-compatible tool calling
across different providers (llama.cpp, Ollama, etc.) and scenarios.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from delia.backend_manager import BackendConfig
from delia.tools import (
    ToolRegistry,
    ToolDefinition,
    AgentConfig,
    run_agent_loop,
    get_default_tools,
    parse_tool_calls,
)


class TestBackendNativeSupport:
    """Tests for backend native tool calling support configuration."""

    def test_llamacpp_backend_with_native_support(self):
        """Test llama.cpp backend can be configured for native support."""
        backend = BackendConfig(
            id="llamacpp-local",
            name="Local llama.cpp",
            provider="llamacpp",
            type="local",
            url="http://localhost:8080",
            supports_native_tool_calling=True,
        )

        assert backend.supports_native_tool_calling is True
        assert backend.provider == "llamacpp"

    def test_ollama_backend_without_native_support(self):
        """Test Ollama backend defaults to no native support."""
        backend = BackendConfig(
            id="ollama-local",
            name="Local Ollama",
            provider="ollama",
            type="local",
            url="http://localhost:11434",
        )

        assert backend.supports_native_tool_calling is False

    def test_backend_native_support_serialization(self):
        """Test that supports_native_tool_calling persists in settings."""
        backend = BackendConfig(
            id="test",
            name="Test",
            provider="llamacpp",
            type="local",
            url="http://localhost:8080",
            supports_native_tool_calling=True,
        )

        # Serialize to dict
        data = backend.to_dict()
        assert "supports_native_tool_calling" in data
        assert data["supports_native_tool_calling"] is True

        # Deserialize back
        restored = BackendConfig.from_dict(data)
        assert restored.supports_native_tool_calling is True


class TestNativeToolPayload:
    """Tests for native tool calling payload format."""

    @pytest.mark.asyncio
    async def test_tools_array_format_matches_openai_spec(self):
        """Test that tools array matches OpenAI function calling spec."""
        registry = get_default_tools()
        schemas = registry.get_openai_schemas()

        # Verify structure matches OpenAI spec
        for schema in schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

            # Parameters should be valid JSON schema
            params = schema["function"]["parameters"]
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params

    @pytest.mark.asyncio
    async def test_native_tool_call_response_parsing(self):
        """Test parsing native tool call response from llama.cpp."""
        # Sample response from llama.cpp with tool calls
        response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'll read that file for you.",
                    "tool_calls": [
                        {
                            "id": "call_xyz789",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "/home/user/config.json", "start_line": 1}'
                            }
                        }
                    ]
                }
            }]
        }

        tool_calls = parse_tool_calls(response, native_mode=True)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_xyz789"
        assert tool_calls[0].name == "read_file"
        assert tool_calls[0].arguments["path"] == "/home/user/config.json"
        assert tool_calls[0].arguments["start_line"] == 1

    @pytest.mark.asyncio
    async def test_multiple_native_tool_calls_in_response(self):
        """Test parsing multiple tool calls from native response."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "a.py"}'
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "b.py"}'
                            }
                        }
                    ]
                }
            }]
        }

        tool_calls = parse_tool_calls(response, native_mode=True)

        assert len(tool_calls) == 2
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].arguments["path"] == "a.py"
        assert tool_calls[1].id == "call_2"
        assert tool_calls[1].arguments["path"] == "b.py"


class TestNativeToolConversation:
    """Tests for conversation handling with native tool calling."""

    @pytest.mark.asyncio
    async def test_conversation_includes_tool_call_structure(self):
        """Test that conversation properly includes tool call structure."""
        call_history = []

        async def mock_llm(messages, system):
            call_history.append(messages)

            if len(call_history) == 1:
                # First call: make a tool call
                return {
                    "choices": [{
                        "message": {
                            "content": "Let me check that.",
                            "tool_calls": [{
                                "id": "call_abc",
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
                # Second call: verify tool results are in conversation
                assert len(messages) >= 2
                # Should have assistant message with tool_calls
                assert any(m.get("role") == "assistant" for m in messages)
                # Should have tool result message
                assert any(m.get("role") == "tool" for m in messages)

                return {"choices": [{"message": {"content": "Done."}}]}

        registry = ToolRegistry()

        async def mock_read(path: str, **kwargs):
            return "file contents"

        registry.register(ToolDefinition(
            name="read_file",
            description="Read a file",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=mock_read,
        ))

        config = AgentConfig(native_tool_calling=True)

        result = await run_agent_loop(
            call_llm=mock_llm,
            prompt="Read /tmp/test.txt",
            system_prompt=None,
            registry=registry,
            model="test",
            config=config,
        )

        assert result.success
        assert len(call_history) == 2

    @pytest.mark.asyncio
    async def test_tool_result_message_format(self):
        """Test that tool result messages match OpenAI format."""
        from delia.tools.parser import format_tool_result

        result = format_tool_result(
            tool_call_id="call_123",
            tool_name="read_file",
            result="File contents here"
        )

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["name"] == "read_file"
        assert result["content"] == "File contents here"


class TestNativeVsTextFormat:
    """Tests comparing native and text-based tool calling."""

    @pytest.mark.asyncio
    async def test_same_tool_call_both_formats(self):
        """Test that the same tool call works in both formats."""
        # Native format
        native_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_code",
                            "arguments": '{"query": "def main", "path": "/src"}'
                        }
                    }]
                }
            }]
        }

        # Text format
        text_response = '''<tool_call>{"name": "search_code", "arguments": {"query": "def main", "path": "/src"}}</tool_call>'''

        native_calls = parse_tool_calls(native_response, native_mode=True)
        text_calls = parse_tool_calls(text_response, native_mode=False)

        # Both should parse to equivalent tool calls
        assert len(native_calls) == 1
        assert len(text_calls) == 1
        assert native_calls[0].name == text_calls[0].name
        assert native_calls[0].arguments == text_calls[0].arguments

    @pytest.mark.asyncio
    async def test_native_format_more_reliable_for_complex_args(self):
        """Test that native format handles complex arguments better."""
        # Complex arguments with nested structures
        complex_args = {
            "path": "/home/user/project",
            "patterns": ["*.py", "*.js"],
            "exclude": ["node_modules", "__pycache__"],
            "max_depth": 5
        }

        native_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_complex",
                        "type": "function",
                        "function": {
                            "name": "search_code",
                            "arguments": json.dumps(complex_args)
                        }
                    }]
                }
            }]
        }

        calls = parse_tool_calls(native_response, native_mode=True)

        assert len(calls) == 1
        assert calls[0].arguments == complex_args
        assert isinstance(calls[0].arguments["patterns"], list)
        assert len(calls[0].arguments["patterns"]) == 2


class TestNativeToolErrors:
    """Tests for error handling in native tool calling."""

    @pytest.mark.asyncio
    async def test_invalid_tool_name_in_native_response(self):
        """Test handling of invalid tool name in native response."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "nonexistent_tool",
                            "arguments": '{}'
                        }
                    }]
                }
            }]
        }

        calls = parse_tool_calls(response, native_mode=True)
        assert len(calls) == 1
        assert calls[0].name == "nonexistent_tool"

        # Execution should handle gracefully
        from delia.tools import execute_tool
        registry = get_default_tools()
        result = await execute_tool(calls[0], registry)

        assert not result.success
        assert "Unknown tool" in result.output

    @pytest.mark.asyncio
    async def test_malformed_json_in_arguments(self):
        """Test handling of malformed JSON in arguments."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_bad_json",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{invalid json here}'
                        }
                    }]
                }
            }]
        }

        # Should handle gracefully (returns empty list or error)
        calls = parse_tool_calls(response, native_mode=True)
        # Implementation may return empty list for invalid JSON
        # or a call with empty arguments
        assert isinstance(calls, list)


class TestSystemPromptWithNative:
    """Tests for system prompt handling with native tool calling."""

    @pytest.mark.asyncio
    async def test_system_prompt_excludes_tools_in_native_mode(self):
        """Test that tools aren't duplicated in system prompt when using native."""
        from delia.tools.agent import build_system_prompt

        base_system = "You are a helpful assistant."
        registry = get_default_tools()

        # Native mode - should NOT include tool descriptions
        native_prompt = build_system_prompt(base_system, registry, native_mode=True)
        assert base_system in native_prompt
        assert "<tool_call>" not in native_prompt
        assert "read_file" not in native_prompt

        # Text mode - SHOULD include tool descriptions
        text_prompt = build_system_prompt(base_system, registry, native_mode=False)
        assert base_system in text_prompt
        assert "<tool_call>" in text_prompt
        assert "read_file" in text_prompt

    def test_system_prompt_only_base_when_native(self):
        """Test that system prompt is minimal in native mode."""
        from delia.tools.agent import build_system_prompt

        base = "Be concise."
        registry = get_default_tools()

        native_prompt = build_system_prompt(base, registry, native_mode=True)

        # Should be just the base prompt (or very close to it)
        assert len(native_prompt) < len(base) + 100  # Allow for some whitespace
        assert "Be concise." in native_prompt
