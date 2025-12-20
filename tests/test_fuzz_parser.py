# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import json
from hypothesis import given, strategies as st, assume
from delia.tools.parser import parse_tool_calls, has_tool_calls

# Strategies for generating potential tool call components
st_tool_name = st.text(min_size=1, max_size=50)
st_arguments = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.one_of(
        st.text(max_size=100),
        st.integers(),
        st.booleans(),
        st.none(),
        st.lists(st.text(max_size=50), max_size=5)
    ),
    max_size=10
)

@st.composite
def valid_tool_json(draw):
    name = draw(st_tool_name)
    args = draw(st_arguments)
    return json.dumps({"name": name, "arguments": args})

@given(st.text())
def test_parse_tool_calls_never_crashes(response):
    """The parser should never raise an exception for any input string."""
    try:
        results = parse_tool_calls(response)
        assert isinstance(results, list)
    except Exception as e:
        pytest.fail(f"parse_tool_calls crashed with input {repr(response)}: {e}")

@given(st.text())
def test_has_tool_calls_never_crashes(response):
    """has_tool_calls should never raise an exception."""
    try:
        result = has_tool_calls(response)
        assert isinstance(result, bool)
    except Exception as e:
        pytest.fail(f"has_tool_calls crashed with input {repr(response)}: {e}")

@given(valid_tool_json())
def test_parse_valid_wrapped_tool(tool_json):
    """Valid tool calls inside XML tags should always be parsed correctly."""
    wrapped = f"<tool_call>{tool_json}</tool_call>"
    results = parse_tool_calls(wrapped)
    assert len(results) == 1
    
    expected = json.loads(tool_json)
    assert results[0].name == expected["name"]
    assert results[0].arguments == expected["arguments"]

@given(valid_tool_json())
def test_parse_valid_raw_tool(tool_json):
    """Valid tool calls without XML tags (fallback) should be parsed."""
    results = parse_tool_calls(tool_json)
    assert len(results) == 1
    
    expected = json.loads(tool_json)
    assert results[0].name == expected["name"]
    assert results[0].arguments == expected["arguments"]

@given(st.text(), valid_tool_json(), st.text())
def test_parse_tool_with_surrounding_text(prefix, tool_json, suffix):
    """Tool calls embedded in other text should be parsed."""
    wrapped = f"{prefix}<tool_call>{tool_json}</tool_call>{suffix}"
    results = parse_tool_calls(wrapped)
    # Note: If prefix/suffix contains <tool_call>, we might get more
    # but at least the one we inserted should be there
    assert len(results) >= 1
    
    expected = json.loads(tool_json)
    names = [r.name for r in results]
    assert expected["name"] in names

@given(st.text())
def test_malformed_xml_handling(inner):
    """Malformed XML wrappers should not cause crashes and should be handled safely."""
    # Test unclosed, nested, or misnamed tags
    cases = [
        f"<tool_call>{inner}",
        f"{inner}</tool_call>",
        f"<tool_call><tool_call>{inner}</tool_call>",
        f"<wrong_tag>{inner}</wrong_tag>",
        f"<{inner}>"
    ]
    for case in cases:
        # Should not crash
        parse_tool_calls(case)

@given(st.text())
def test_invalid_json_in_valid_tag(invalid_json):
    """Invalid JSON inside valid tags should be ignored gracefully."""
    assume(not invalid_json.strip().startswith("{")) # Ensure it's likely invalid
    wrapped = f"<tool_call>{invalid_json}</tool_call>"
    results = parse_tool_calls(wrapped)
    assert len(results) == 0

@given(st.text())
def test_unicode_and_garbage_arguments(garbage):
    """Extreme unicode and garbage characters in arguments shouldn't crash."""
    tool_json = json.dumps({
        "name": "test_tool",
        "arguments": {"input": garbage}
    })
    wrapped = f"<tool_call>{tool_json}</tool_call>"
    results = parse_tool_calls(wrapped)
    if results:
        assert results[0].name == "test_tool"
        assert results[0].arguments["input"] == garbage

@given(st.dictionaries(st.text(), st.text()))
def test_native_mode_fuzz(response_dict):
    """Fuzzing the native mode (dict input)."""
    try:
        parse_tool_calls(response_dict, native_mode=True)
    except Exception as e:
        pytest.fail(f"Native mode crashed: {e}")

if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
