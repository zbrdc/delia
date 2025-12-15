#!/usr/bin/env python3
"""
AFL++ Coverage-Guided Fuzzing Harness for Delia

High-priority targets identified via code analysis:
1. parse_tool_calls() - LLM tool call parsing (XML markers, JSON, native format)
2. detect_code_content() - 20+ regex patterns for code detection
3. parse_model_override() - Regex-based model tier selection
4. Validation functions - Input validation for MCP tools

Installation:
    pip install python-afl

Usage (single core):
    mkdir -p fuzz/corpus fuzz/findings
    echo '<tool_call>{"name": "test", "arguments": {}}</tool_call>' > fuzz/corpus/seed1.txt
    AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
      py-afl-fuzz -i fuzz/corpus -o fuzz/findings -- python fuzz/afl_harness.py

Usage (multi-core):
    # Main fuzzer
    py-afl-fuzz -M main -i fuzz/corpus -o fuzz/findings -- python fuzz/afl_harness.py
    # Secondary fuzzers
    py-afl-fuzz -S sec1 -i fuzz/corpus -o fuzz/findings -- python fuzz/afl_harness.py

Persistent mode (10-100x faster):
    py-afl-fuzz -i fuzz/corpus -o fuzz/findings -- python fuzz/afl_harness.py --persistent

Select specific target:
    AFL_TARGET=parser py-afl-fuzz -i fuzz/corpus -o fuzz/findings -- python fuzz/afl_harness.py
    Valid targets: parser, routing, validation, all
"""

import json
import os
import sys
from pathlib import Path

# Add project src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Disable structlog output during fuzzing
os.environ["DELIA_LOG_LEVEL"] = "CRITICAL"

# Try to import AFL
try:
    import afl
    AFL_AVAILABLE = True
except ImportError:
    AFL_AVAILABLE = False
    print("Warning: python-afl not installed. Running in debug mode.", file=sys.stderr)

# Import target modules
from delia.tools.parser import (
    parse_tool_calls,
    ParsedToolCall,
    TOOL_CALL_PATTERN,
    RAW_JSON_TOOL_PATTERN,
)
from delia.routing import (
    detect_code_content,
    parse_model_override,
)
from delia.validation import (
    validate_task,
    validate_content,
    validate_file_path,
    validate_model_hint,
    VALID_TASKS,
    MAX_CONTENT_LENGTH,
)


# Expected exceptions (normal behavior, not bugs)
EXPECTED_EXCEPTIONS = (
    json.JSONDecodeError,
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
    UnicodeDecodeError,
)


# ============================================================
# FUZZ TARGET 1: Tool Call Parser (HIGH PRIORITY)
# ============================================================

def fuzz_parse_tool_calls_text(data: bytes) -> None:
    """
    Fuzz target: parse_tool_calls() in text mode

    Attack vectors:
    - XML marker manipulation (<tool_call> tags)
    - Malformed JSON in tool calls
    - Nested/recursive structures
    - ReDoS in TOOL_CALL_PATTERN regex
    - Raw JSON fallback path
    """
    try:
        text = data.decode('utf-8', errors='replace')
        result = parse_tool_calls(text, native_mode=False)

        # Verify invariants
        if not isinstance(result, list):
            raise AssertionError(f"Expected list, got {type(result)}")
        for item in result:
            if not isinstance(item, ParsedToolCall):
                raise AssertionError(f"Expected ParsedToolCall, got {type(item)}")
            if not isinstance(item.name, str):
                raise AssertionError(f"Tool name must be str, got {type(item.name)}")
            if not isinstance(item.arguments, dict):
                raise AssertionError(f"Tool arguments must be dict, got {type(item.arguments)}")

    except EXPECTED_EXCEPTIONS:
        pass  # Normal behavior


def fuzz_parse_tool_calls_native(data: bytes) -> None:
    """
    Fuzz target: parse_tool_calls() in native mode (OpenAI format)

    Attack vectors:
    - Malformed OpenAI response structure
    - Missing tool_calls key
    - Invalid function structure
    - Arguments as string vs dict
    """
    try:
        text = data.decode('utf-8', errors='replace')

        # Try to parse as JSON first
        try:
            response_dict = json.loads(text)
            if not isinstance(response_dict, dict):
                return
        except json.JSONDecodeError:
            return

        result = parse_tool_calls(response_dict, native_mode=True)

        if not isinstance(result, list):
            raise AssertionError(f"Expected list, got {type(result)}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# FUZZ TARGET 2: Code Content Detection (HIGH PRIORITY)
# ============================================================

def fuzz_detect_code_content(data: bytes) -> None:
    """
    Fuzz target: detect_code_content()

    Attack vectors:
    - ReDoS in 20+ pre-compiled regex patterns
    - Division edge cases in score normalization
    - Memory exhaustion from large inputs
    - Unicode handling in regex patterns
    - Line counting edge cases
    """
    try:
        text = data.decode('utf-8', errors='replace')
        is_code, confidence, reasoning = detect_code_content(text)

        # Verify invariants
        if not isinstance(is_code, bool):
            raise AssertionError(f"is_code must be bool, got {type(is_code)}")
        if not isinstance(confidence, (int, float)):
            raise AssertionError(f"confidence must be numeric, got {type(confidence)}")
        if confidence < 0 or confidence > 1:
            raise AssertionError(f"confidence out of range: {confidence}")
        if not isinstance(reasoning, str):
            raise AssertionError(f"reasoning must be str, got {type(reasoning)}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# FUZZ TARGET 3: Model Override Parser (MEDIUM PRIORITY)
# ============================================================

def fuzz_parse_model_override(data: bytes) -> None:
    """
    Fuzz target: parse_model_override()

    Attack vectors:
    - ReDoS in tier keyword patterns
    - Unicode edge cases in regex
    - Case sensitivity bypass attempts
    - Whitespace manipulation
    """
    try:
        text = data.decode('utf-8', errors='replace')

        # Split input: first line is hint, rest is content
        lines = text.split('\n', 1)
        hint = lines[0] if lines else None
        content = lines[1] if len(lines) > 1 else ""

        # Test with hint
        result1 = parse_model_override(hint, content)
        if result1 is not None and not isinstance(result1, str):
            raise AssertionError(f"Expected str or None, got {type(result1)}")

        # Test without hint
        result2 = parse_model_override(None, text)
        if result2 is not None and not isinstance(result2, str):
            raise AssertionError(f"Expected str or None, got {type(result2)}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# FUZZ TARGET 4: Input Validation (MEDIUM PRIORITY)
# ============================================================

def fuzz_validate_task(data: bytes) -> None:
    """
    Fuzz target: validate_task()

    Attack vectors:
    - Case sensitivity bypass
    - Null byte injection
    - Unicode normalization
    """
    try:
        text = data.decode('utf-8', errors='replace')
        is_valid, error = validate_task(text)

        if not isinstance(is_valid, bool):
            raise AssertionError(f"is_valid must be bool, got {type(is_valid)}")
        if not isinstance(error, str):
            raise AssertionError(f"error must be str, got {type(error)}")

    except EXPECTED_EXCEPTIONS:
        pass


def fuzz_validate_content(data: bytes) -> None:
    """
    Fuzz target: validate_content()

    Attack vectors:
    - UTF-8 encoding edge cases
    - Boundary at MAX_CONTENT_LENGTH
    - Type confusion
    """
    try:
        text = data.decode('utf-8', errors='replace')
        is_valid, error = validate_content(text)

        if not isinstance(is_valid, bool):
            raise AssertionError(f"is_valid must be bool, got {type(is_valid)}")

    except EXPECTED_EXCEPTIONS:
        pass


def fuzz_validate_file_path(data: bytes) -> None:
    """
    Fuzz target: validate_file_path()

    Security-critical: Path traversal prevention

    Attack vectors:
    - Path traversal (.., ..\\)
    - URL-encoded sequences (%2e%2e)
    - Unicode normalization attacks
    - Null byte injection
    - Tilde expansion edge cases
    """
    try:
        text = data.decode('utf-8', errors='replace')
        is_valid, error = validate_file_path(text)

        if not isinstance(is_valid, bool):
            raise AssertionError(f"is_valid must be bool, got {type(is_valid)}")

        # Security invariant: path traversal MUST be rejected
        if ".." in text and is_valid:
            raise AssertionError(f"Path traversal not rejected: {text[:100]}")

    except EXPECTED_EXCEPTIONS:
        pass


def fuzz_validate_model_hint(data: bytes) -> None:
    """
    Fuzz target: validate_model_hint()

    Attack vectors:
    - Set membership bypass
    - Case sensitivity
    """
    try:
        text = data.decode('utf-8', errors='replace')
        is_valid, error = validate_model_hint(text)

        if not isinstance(is_valid, bool):
            raise AssertionError(f"is_valid must be bool, got {type(is_valid)}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# COMBINED FUZZER
# ============================================================

def fuzz_all_targets(data: bytes) -> None:
    """Combined fuzzer hitting all targets for maximum coverage."""
    fuzz_parse_tool_calls_text(data)
    fuzz_parse_tool_calls_native(data)
    fuzz_detect_code_content(data)
    fuzz_parse_model_override(data)
    fuzz_validate_task(data)
    fuzz_validate_content(data)
    fuzz_validate_file_path(data)
    fuzz_validate_model_hint(data)


def fuzz_parser_targets(data: bytes) -> None:
    """Parser-focused fuzzing (tool calls)."""
    fuzz_parse_tool_calls_text(data)
    fuzz_parse_tool_calls_native(data)


def fuzz_routing_targets(data: bytes) -> None:
    """Routing-focused fuzzing (code detection, model override)."""
    fuzz_detect_code_content(data)
    fuzz_parse_model_override(data)


def fuzz_validation_targets(data: bytes) -> None:
    """Validation-focused fuzzing."""
    fuzz_validate_task(data)
    fuzz_validate_content(data)
    fuzz_validate_file_path(data)
    fuzz_validate_model_hint(data)


# ============================================================
# PERSISTENT MODE
# ============================================================

def fuzz_persistent(fuzz_fn) -> None:
    """
    Persistent mode - 10-100x faster than fork mode.
    Processes multiple inputs per fork.
    """
    if not AFL_AVAILABLE:
        print("Persistent mode requires python-afl", file=sys.stderr)
        return

    while afl.loop(1000):
        data = sys.stdin.buffer.read()
        fuzz_fn(data)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    # Select target based on AFL_TARGET environment variable
    target = os.environ.get('AFL_TARGET', 'all')

    target_map = {
        'all': fuzz_all_targets,
        'parser': fuzz_parser_targets,
        'routing': fuzz_routing_targets,
        'validation': fuzz_validation_targets,
        'tool_calls': fuzz_parse_tool_calls_text,
        'code_detect': fuzz_detect_code_content,
        'file_path': fuzz_validate_file_path,
    }

    fuzz_fn = target_map.get(target, fuzz_all_targets)

    if '--persistent' in sys.argv:
        fuzz_persistent(fuzz_fn)
    else:
        if AFL_AVAILABLE:
            afl.init()

        data = sys.stdin.buffer.read()
        fuzz_fn(data)
