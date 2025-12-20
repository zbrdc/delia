#!/usr/bin/env python3
"""
Dataset Builder for FunctionGemma Fine-Tuning

Harvests training examples from:
1. Test files (test_tools.py, test_native_tool_calling.py, etc.)
2. Manual examples (synthetic)
3. tools.openai.json as the single source of truth for allowed tools

Outputs:
- data/train.jsonl - Training set
- data/eval.jsonl - Evaluation set (10% holdout)

Uses FunctionGemma/Gemma chat template format via apply_chat_template.
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict

# Add src to path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
TOOLS_SCHEMA_PATH = SCRIPT_DIR / "tools.openai.json"
SYNTHETIC_DATA_PATH = SCRIPT_DIR / "synthetic_examples.json"
TESTS_DIR = PROJECT_ROOT / "tests"
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_OUTPUT = DATA_DIR / "train.jsonl"
EVAL_OUTPUT = DATA_DIR / "eval.jsonl"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)


@dataclass
class ToolCall:
    """Represents a single tool call."""
    name: str
    arguments: dict[str, Any]
    
    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user", "assistant", "tool"
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # For tool responses
    name: str | None = None  # Tool name for tool responses
    
    def to_dict(self) -> dict:
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": tc.to_dict()
                }
                for i, tc in enumerate(self.tool_calls)
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class TrainingExample:
    """A complete training example."""
    messages: list[ConversationTurn]
    tools: list[dict]  # OpenAI-format tool schemas
    
    def to_dict(self) -> dict:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "tools": self.tools,
        }


def load_tool_schemas() -> tuple[list[dict], dict[str, dict]]:
    """Load tool schemas from tools.openai.json.
    
    Returns:
        Tuple of (list of tool schemas, dict mapping tool name to schema)
    """
    with open(TOOLS_SCHEMA_PATH) as f:
        tools = json.load(f)
    
    # Build lookup map
    tool_map = {}
    for tool in tools:
        name = tool["function"]["name"]
        tool_map[name] = tool
    
    return tools, tool_map


def validate_tool_call(tool_call: ToolCall, tool_map: dict[str, dict]) -> tuple[bool, str]:
    """Validate a tool call against the schema.
    
    Args:
        tool_call: The tool call to validate
        tool_map: Map of tool name to schema
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if tool_call.name not in tool_map:
        return False, f"Unknown tool: {tool_call.name}"
    
    schema = tool_map[tool_call.name]["function"]["parameters"]
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    
    # Check required fields
    for req in required:
        if req not in tool_call.arguments:
            return False, f"Missing required field: {req}"
    
    # Check field types (basic validation)
    for key, value in tool_call.arguments.items():
        if key not in properties:
            # Extra fields are OK for flexibility
            continue
        
        prop_type = properties[key].get("type")
        if prop_type == "string" and not isinstance(value, str):
            return False, f"Field {key} should be string, got {type(value).__name__}"
        elif prop_type == "integer" and not isinstance(value, int):
            return False, f"Field {key} should be integer, got {type(value).__name__}"
        elif prop_type == "boolean" and not isinstance(value, bool):
            return False, f"Field {key} should be boolean, got {type(value).__name__}"
        elif prop_type == "number" and not isinstance(value, (int, float)):
            return False, f"Field {key} should be number, got {type(value).__name__}"
        
        # Check enum constraints
        if "enum" in properties[key]:
            if value not in properties[key]["enum"]:
                return False, f"Field {key} value '{value}' not in enum {properties[key]['enum']}"
    
    return True, ""


def extract_xml_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from XML format.
    
    Format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    """
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    tool_calls = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            tool_calls.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return tool_calls


def extract_native_tool_calls(response: dict) -> list[dict]:
    """Extract tool calls from OpenAI-native format response."""
    tool_calls = []
    
    if "choices" in response:
        for choice in response["choices"]:
            msg = choice.get("message", {})
            for tc in msg.get("tool_calls", []):
                if tc.get("type") == "function":
                    func = tc["function"]
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append({
                        "name": func["name"],
                        "arguments": args
                    })
    
    return tool_calls


def harvest_from_test_files(tool_map: dict[str, dict]) -> list[TrainingExample]:
    """Harvest training examples from test files."""
    examples = []
    
    # Patterns to find in test files
    test_files = [
        TESTS_DIR / "test_tools.py",
        TESTS_DIR / "test_native_tool_calling.py",
        TESTS_DIR / "test_mcp_tools.py",
    ]
    
    for test_file in test_files:
        if not test_file.exists():
            print(f"  Skipping {test_file.name} (not found)")
            continue
        
        content = test_file.read_text()
        print(f"  Processing {test_file.name}...")
        
        # Extract XML format tool calls
        xml_calls = extract_xml_tool_calls(content)
        for tc in xml_calls:
            name = tc.get("name")
            args = tc.get("arguments", {})
            
            if name not in tool_map:
                continue
            
            # Create a simple training example
            tool_call = ToolCall(name=name, arguments=args)
            valid, error = validate_tool_call(tool_call, tool_map)
            
            if not valid:
                print(f"    Skipping invalid: {name} - {error}")
                continue
            
            # Generate a synthetic user prompt
            user_prompt = generate_user_prompt(name, args)
            
            example = TrainingExample(
                messages=[
                    ConversationTurn(role="user", content=user_prompt),
                    ConversationTurn(role="assistant", tool_calls=[tool_call]),
                ],
                tools=[tool_map[name]],
            )
            examples.append(example)
    
    return examples


def generate_user_prompt(tool_name: str, args: dict) -> str:
    """Generate a natural language user prompt for a tool call."""
    prompts = {
        "read_file": lambda a: f"Read the file {a.get('path', 'unknown')}",
        "list_directory": lambda a: f"List files in {a.get('path', '.')}",
        "search_code": lambda a: f"Search for '{a.get('pattern', '')}' in the codebase",
        "web_fetch": lambda a: f"Fetch the content from {a.get('url', '')}",
        "web_search": lambda a: f"Search the web for: {a.get('query', '')}",
        "delegate": lambda a: f"Delegate a {a.get('task', 'task')} task: {a.get('content', '')[:100]}",
        "think": lambda a: f"Think through this problem: {a.get('problem', '')[:100]}",
        "batch": lambda a: f"Run these tasks in parallel: {a.get('tasks', '')[:100]}",
        "health": lambda a: "Check the health status of all backends",
        "models": lambda a: "List all available models",
        "write_file": lambda a: f"Write to file {a.get('path', '')}",
        "shell_exec": lambda a: f"Run command: {a.get('command', '')}",
        "agent": lambda a: f"Use the agent to: {a.get('prompt', '')[:100]}",
        "chain": lambda a: "Execute a task chain",
        "workflow": lambda a: "Run a workflow",
    }
    
    if tool_name in prompts:
        return prompts[tool_name](args)
    
    # Default prompt
    return f"Use the {tool_name} tool"


def generate_synthetic_examples(tools: list[dict], tool_map: dict[str, dict]) -> list[TrainingExample]:
    """Generate synthetic training examples for each tool."""
    examples = []
    
    # Load synthetic data from external JSON
    if not SYNTHETIC_DATA_PATH.exists():
        print(f"  Warning: Synthetic data file not found at {SYNTHETIC_DATA_PATH}")
        return []
        
    try:
        with open(SYNTHETIC_DATA_PATH) as f:
            synthetic_data = json.load(f)
    except Exception as e:
        print(f"  Error loading synthetic data: {e}")
        return []
    
    for item in synthetic_data:
        tool_name = item["tool"]
        if tool_name not in tool_map:
            continue
        
        tool_call = ToolCall(name=tool_name, arguments=item["args"])
        valid, error = validate_tool_call(tool_call, tool_map)
        
        if not valid:
            print(f"  Skipping invalid synthetic: {tool_name} - {error}")
            continue
        
        example = TrainingExample(
            messages=[
                ConversationTurn(role="user", content=item["user"]),
                ConversationTurn(role="assistant", tool_calls=[tool_call]),
            ],
            tools=[tool_map[tool_name]],
        )
        examples.append(example)
    
    return examples


def generate_multi_turn_examples(tool_map: dict[str, dict]) -> list[TrainingExample]:
    """Generate multi-turn conversation examples."""
    examples = []
    
    # Example 1: File exploration
    if "read_file" in tool_map and "search_code" in tool_map:
        example = TrainingExample(
            messages=[
                ConversationTurn(role="user", content="Find all usages of the Config class and show me the main implementation"),
                ConversationTurn(role="assistant", tool_calls=[
                    ToolCall(name="search_code", arguments={"pattern": "class Config", "file_pattern": "*.py"})
                ]),
                ConversationTurn(
                    role="tool",
                    content="src/config.py:15: class Config:\nsrc/config.py:16:    def __init__(self):",
                    tool_call_id="call_0",
                    name="search_code"
                ),
                ConversationTurn(role="assistant", tool_calls=[
                    ToolCall(name="read_file", arguments={"path": "src/config.py", "start_line": 15, "end_line": 50})
                ]),
                ConversationTurn(
                    role="tool",
                    content="15: class Config:\n16:     def __init__(self):\n17:         self.debug = False",
                    tool_call_id="call_0",
                    name="read_file"
                ),
                ConversationTurn(role="assistant", content="I found the Config class in src/config.py. It's a simple configuration class that initializes with debug=False."),
            ],
            tools=[tool_map["read_file"], tool_map["search_code"]],
        )
        examples.append(example)
    
    # Example 2: Code review workflow
    if "delegate" in tool_map:
        example = TrainingExample(
            messages=[
                ConversationTurn(role="user", content="Review this function:\ndef factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"),
                ConversationTurn(role="assistant", tool_calls=[
                    ToolCall(name="delegate", arguments={
                        "task": "review",
                        "content": "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)",
                        "language": "python"
                    })
                ]),
                ConversationTurn(
                    role="tool",
                    content="The factorial function looks correct but could benefit from:\n1. Input validation for negative numbers\n2. Type hints\n3. A docstring",
                    tool_call_id="call_0",
                    name="delegate"
                ),
                ConversationTurn(role="assistant", content="The code review found that while the factorial function is correct, it could be improved with input validation for negative numbers, type hints, and a docstring."),
            ],
            tools=[tool_map["delegate"]],
        )
        examples.append(example)
    
    # Example 3: Health check + model switch
    if "health" in tool_map and "switch_backend" in tool_map:
        example = TrainingExample(
            messages=[
                ConversationTurn(role="user", content="Check which backends are available and switch to the fastest one"),
                ConversationTurn(role="assistant", tool_calls=[
                    ToolCall(name="health", arguments={})
                ]),
                ConversationTurn(
                    role="tool",
                    content='{"status": "healthy", "backends": [{"id": "local", "latency_ms": 50}, {"id": "remote", "latency_ms": 200}]}',
                    tool_call_id="call_0",
                    name="health"
                ),
                ConversationTurn(role="assistant", tool_calls=[
                    ToolCall(name="switch_backend", arguments={"backend_id": "local"})
                ]),
                ConversationTurn(
                    role="tool",
                    content="Switched to local backend successfully",
                    tool_call_id="call_0",
                    name="switch_backend"
                ),
                ConversationTurn(role="assistant", content="I checked the backends and switched to 'local' which has the lowest latency at 50ms compared to 'remote' at 200ms."),
            ],
            tools=[tool_map["health"], tool_map["switch_backend"]],
        )
        examples.append(example)
    
    return examples


def format_for_functiongemma(example: TrainingExample) -> dict:
    """Format a training example for FunctionGemma.
    
    FunctionGemma expects a specific format that will be processed
    by apply_chat_template. The format includes:
    - messages: list of message dicts
    - tools: list of tool schemas in OpenAI format
    
    The template handles the tool call formatting internally.
    """
    return example.to_dict()


def main():
    """Main entry point."""
    print("=" * 60)
    print("FunctionGemma Dataset Builder for Delia")
    print("=" * 60)
    
    # Load tool schemas
    print("\n1. Loading tool schemas...")
    tools, tool_map = load_tool_schemas()
    print(f"   Loaded {len(tools)} tools from tools.openai.json")
    
    # Harvest from test files
    print("\n2. Harvesting from test files...")
    test_examples = harvest_from_test_files(tool_map)
    print(f"   Harvested {len(test_examples)} examples from tests")
    
    # Generate synthetic examples
    print("\n3. Generating synthetic examples...")
    synthetic_examples = generate_synthetic_examples(tools, tool_map)
    print(f"   Generated {len(synthetic_examples)} synthetic examples")
    
    # Generate multi-turn examples
    print("\n4. Generating multi-turn examples...")
    multi_turn_examples = generate_multi_turn_examples(tool_map)
    print(f"   Generated {len(multi_turn_examples)} multi-turn examples")
    
    # Combine all examples
    all_examples = test_examples + synthetic_examples + multi_turn_examples
    print(f"\n   Total examples: {len(all_examples)}")
    
    # Deduplicate
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key = json.dumps(ex.to_dict(), sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_examples.append(ex)
    
    print(f"   After deduplication: {len(unique_examples)}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(unique_examples)
    
    split_idx = int(len(unique_examples) * 0.9)
    train_examples = unique_examples[:split_idx]
    eval_examples = unique_examples[split_idx:]
    
    print(f"\n5. Splitting dataset...")
    print(f"   Train: {len(train_examples)} examples")
    print(f"   Eval:  {len(eval_examples)} examples")
    
    # Write output files
    print("\n6. Writing output files...")
    
    with open(TRAIN_OUTPUT, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(format_for_functiongemma(ex)) + "\n")
    print(f"   Wrote {TRAIN_OUTPUT}")
    
    with open(EVAL_OUTPUT, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(format_for_functiongemma(ex)) + "\n")
    print(f"   Wrote {EVAL_OUTPUT}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {TRAIN_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"  - {EVAL_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"\nTool coverage:")
    tools_used = set()
    for ex in unique_examples:
        for msg in ex.messages:
            for tc in msg.tool_calls:
                tools_used.add(tc.name)
    
    print(f"  - {len(tools_used)}/{len(tools)} tools have training examples")
    print(f"  - Missing: {set(tool_map.keys()) - tools_used}")


if __name__ == "__main__":
    main()
