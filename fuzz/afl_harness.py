#!/usr/bin/env python3
"""
AFL++ Coverage-Guided Fuzzing Harness for Delia

High-priority targets identified via code analysis:

PARSER TARGETS (HIGH RISK):
1. parse_tool_calls() - LLM tool call parsing (XML markers, JSON, native format)
2. parse_chain_steps() - JSON chain workflow parsing
3. parse_workflow_definition() - DAG workflow JSON parsing
4. parse_structured_output() - Structured output JSON extraction

ROUTING TARGETS (MEDIUM RISK):
5. detect_code_content() - 20+ regex patterns for code detection
6. parse_model_override() - Regex-based model tier selection
7. detect_chat_task_type() - Chat task routing with regex

VALIDATION TARGETS (CRITICAL):
8. validate_task() - Task type validation
9. validate_content() - Content length/encoding validation
10. validate_file_path() - Path traversal prevention
11. validate_model_hint() - Model hint validation
12. validate_path() - Workspace path security
13. validate_workflow() - Workflow structure validation

QUALITY TARGETS (MEDIUM RISK):
14. validate_response() - Response quality validation

Installation:
    pip install python-afl

Usage (single core):
    mkdir -p fuzz/corpus fuzz/findings
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
    Valid targets: parser, routing, validation, workflow, quality, all
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
    detect_chat_task_type,
)
from delia.validation import (
    validate_task,
    validate_content,
    validate_file_path,
    validate_model_hint,
    VALID_TASKS,
    MAX_CONTENT_LENGTH,
)
from delia.task_chain import parse_chain_steps, ChainStep
from delia.task_workflow import (
    parse_workflow_definition,
    validate_workflow,
    WorkflowDefinition,
    WorkflowNode,
)
from delia.quality import validate_response, QualityScore
from delia.tools.executor import validate_path
from delia.types import Workspace


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
# FUZZ TARGET 2: Chain Steps Parser (HIGH PRIORITY)
# ============================================================

def fuzz_parse_chain_steps(data: bytes) -> None:
    """
    Fuzz target: parse_chain_steps()

    Attack vectors:
    - Malformed JSON array
    - Missing required fields (id, content)
    - Duplicate step IDs
    - Deep nesting in step content
    - Invalid field types
    - Empty arrays
    """
    try:
        text = data.decode('utf-8', errors='replace')
        result = parse_chain_steps(text)

        # Verify invariants
        if not isinstance(result, list):
            raise AssertionError(f"Expected list, got {type(result)}")
        for step in result:
            if not isinstance(step, ChainStep):
                raise AssertionError(f"Expected ChainStep, got {type(step)}")
            if not step.id or not isinstance(step.id, str):
                raise AssertionError(f"Step id must be non-empty str")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# FUZZ TARGET 3: Workflow Definition Parser (HIGH PRIORITY)
# ============================================================

def fuzz_parse_workflow_definition(data: bytes) -> None:
    """
    Fuzz target: parse_workflow_definition()

    Attack vectors:
    - Malformed JSON object
    - Missing entry node
    - Invalid node references (on_success, on_failure, depends_on)
    - Circular dependencies (cycles in DAG)
    - Duplicate node IDs
    - Deep nesting
    - Invalid timeout values
    """
    try:
        text = data.decode('utf-8', errors='replace')
        result = parse_workflow_definition(text)

        # Verify invariants
        if not isinstance(result, WorkflowDefinition):
            raise AssertionError(f"Expected WorkflowDefinition, got {type(result)}")
        if not result.entry:
            raise AssertionError("Workflow must have entry node")

    except EXPECTED_EXCEPTIONS:
        pass


def fuzz_validate_workflow(data: bytes) -> None:
    """
    Fuzz target: validate_workflow() directly

    Attack vectors:
    - Cycle detection bypass
    - Invalid node references
    - Duplicate IDs
    """
    try:
        text = data.decode('utf-8', errors='replace')

        # Parse as JSON first
        try:
            data_dict = json.loads(text)
            if not isinstance(data_dict, dict):
                return
        except json.JSONDecodeError:
            return

        # Create minimal WorkflowDefinition for testing
        nodes = []
        if "nodes" in data_dict and isinstance(data_dict["nodes"], list):
            for node_data in data_dict["nodes"]:
                if isinstance(node_data, dict):
                    try:
                        nodes.append(WorkflowNode(
                            id=str(node_data.get("id", "")),
                            task=str(node_data.get("task", "quick")),
                            content=str(node_data.get("content", "")),
                            depends_on=node_data.get("depends_on"),
                            on_success=node_data.get("on_success"),
                            on_failure=node_data.get("on_failure"),
                        ))
                    except Exception:
                        continue

        definition = WorkflowDefinition(
            name=str(data_dict.get("name", "test")),
            entry=str(data_dict.get("entry", "")),
            nodes=nodes,
        )

        errors = validate_workflow(definition)

        # Verify invariants
        if not isinstance(errors, list):
            raise AssertionError(f"Expected list, got {type(errors)}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# FUZZ TARGET 4: Code Content Detection (HIGH PRIORITY)
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
# FUZZ TARGET 5: Chat Task Type Detection (MEDIUM PRIORITY)
# ============================================================

def fuzz_detect_chat_task_type(data: bytes) -> None:
    """
    Fuzz target: detect_chat_task_type()

    Attack vectors:
    - ReDoS in task type regex patterns
    - Unicode edge cases
    - Very long messages
    - Empty/whitespace-only messages
    """
    try:
        text = data.decode('utf-8', errors='replace')
        task_type, confidence, reasoning = detect_chat_task_type(text)

        # Verify invariants
        if task_type not in ("quick", "coder", "moe"):
            raise AssertionError(f"Invalid task_type: {task_type}")
        if not isinstance(confidence, (int, float)):
            raise AssertionError(f"confidence must be numeric, got {type(confidence)}")
        if confidence < 0 or confidence > 1:
            raise AssertionError(f"confidence out of range: {confidence}")
        if not isinstance(reasoning, str):
            raise AssertionError(f"reasoning must be str, got {type(reasoning)}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# FUZZ TARGET 6: Model Override Parser (MEDIUM PRIORITY)
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
# FUZZ TARGET 7: Input Validation (CRITICAL)
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
# FUZZ TARGET 8: Workspace Path Validation (CRITICAL)
# ============================================================

def fuzz_validate_path_in_workspace(data: bytes) -> None:
    """
    Fuzz target: validate_path() with workspace

    Security-critical: Workspace sandbox escape prevention

    Attack vectors:
    - Path traversal to escape workspace
    - Symlink attacks
    - Unicode path manipulation
    - Null byte injection
    - Absolute path injection
    """
    try:
        text = data.decode('utf-8', errors='replace')

        # Create a test workspace
        workspace = Workspace(root=Path("/tmp/fuzz_workspace"))

        is_valid, error = validate_path(text, workspace)

        if not isinstance(is_valid, bool):
            raise AssertionError(f"is_valid must be bool, got {type(is_valid)}")

        # Security invariant: path traversal MUST be rejected in workspace
        if ".." in text and is_valid:
            raise AssertionError(f"Path traversal in workspace not rejected: {text[:100]}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# FUZZ TARGET 9: Response Quality Validation (MEDIUM)
# ============================================================

def fuzz_validate_response(data: bytes) -> None:
    """
    Fuzz target: validate_response()

    Attack vectors:
    - Repetition detection edge cases
    - Very long responses
    - Unicode edge cases
    - Empty/whitespace responses
    """
    try:
        text = data.decode('utf-8', errors='replace')
        result = validate_response(text, task_type="quick")

        # Verify invariants
        if not isinstance(result, QualityScore):
            raise AssertionError(f"Expected QualityScore, got {type(result)}")
        if not 0 <= result.overall <= 1:
            raise AssertionError(f"overall score out of range: {result.overall}")
        if not 0 <= result.repetition_score <= 1:
            raise AssertionError(f"repetition_score out of range: {result.repetition_score}")
        if not 0 <= result.length_score <= 1:
            raise AssertionError(f"length_score out of range: {result.length_score}")
        if not 0 <= result.coherence_score <= 1:
            raise AssertionError(f"coherence_score out of range: {result.coherence_score}")

    except EXPECTED_EXCEPTIONS:
        pass


# ============================================================
# COMBINED FUZZERS
# ============================================================

def fuzz_all_targets(data: bytes) -> None:
    """Combined fuzzer hitting all targets for maximum coverage."""
    fuzz_parse_tool_calls_text(data)
    fuzz_parse_tool_calls_native(data)
    fuzz_parse_chain_steps(data)
    fuzz_parse_workflow_definition(data)
    fuzz_validate_workflow(data)
    fuzz_detect_code_content(data)
    fuzz_detect_chat_task_type(data)
    fuzz_parse_model_override(data)
    fuzz_validate_task(data)
    fuzz_validate_content(data)
    fuzz_validate_file_path(data)
    fuzz_validate_model_hint(data)
    fuzz_validate_path_in_workspace(data)
    fuzz_validate_response(data)


def fuzz_parser_targets(data: bytes) -> None:
    """Parser-focused fuzzing (tool calls, chains, workflows)."""
    fuzz_parse_tool_calls_text(data)
    fuzz_parse_tool_calls_native(data)
    fuzz_parse_chain_steps(data)
    fuzz_parse_workflow_definition(data)


def fuzz_routing_targets(data: bytes) -> None:
    """Routing-focused fuzzing (code detection, model override, task type)."""
    fuzz_detect_code_content(data)
    fuzz_parse_model_override(data)
    fuzz_detect_chat_task_type(data)


def fuzz_validation_targets(data: bytes) -> None:
    """Validation-focused fuzzing."""
    fuzz_validate_task(data)
    fuzz_validate_content(data)
    fuzz_validate_file_path(data)
    fuzz_validate_model_hint(data)
    fuzz_validate_path_in_workspace(data)


def fuzz_workflow_targets(data: bytes) -> None:
    """Workflow-focused fuzzing (chains and DAGs)."""
    fuzz_parse_chain_steps(data)
    fuzz_parse_workflow_definition(data)
    fuzz_validate_workflow(data)


def fuzz_quality_targets(data: bytes) -> None:
    """Quality validation fuzzing."""
    fuzz_validate_response(data)


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
        'workflow': fuzz_workflow_targets,
        'quality': fuzz_quality_targets,
        # Individual targets
        'tool_calls': fuzz_parse_tool_calls_text,
        'chain': fuzz_parse_chain_steps,
        'workflow_def': fuzz_parse_workflow_definition,
        'code_detect': fuzz_detect_code_content,
        'task_type': fuzz_detect_chat_task_type,
        'file_path': fuzz_validate_file_path,
        'workspace': fuzz_validate_path_in_workspace,
        'response': fuzz_validate_response,
    }

    fuzz_fn = target_map.get(target, fuzz_all_targets)

    if '--persistent' in sys.argv:
        fuzz_persistent(fuzz_fn)
    else:
        if AFL_AVAILABLE:
            afl.init()

        data = sys.stdin.buffer.read()
        fuzz_fn(data)
