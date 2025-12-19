#!/usr/bin/env python3
"""
Inference Agent for Fine-Tuned FunctionGemma

A safe executor that:
1. Parses model tool calls (XML format or native)
2. Validates tool name + arguments against tools.openai.json
3. Executes Delia MCP tools
4. Feeds tool results back to the model for multi-step orchestration

Usage:
    python scripts/run_agent.py --prompt "Review this code: def foo(): pass"
    python scripts/run_agent.py --model ./outputs/final/merged --prompt "..."
    python scripts/run_agent.py --interactive
"""

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path for Delia imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Paths
TOOLS_SCHEMA_PATH = SCRIPT_DIR / "tools.openai.json"

# Try to import Delia modules for actual tool execution
try:
    from delia.tools.builtins import get_default_tools
    from delia.tools.registry import ToolRegistry
    DELIA_AVAILABLE = True
except ImportError:
    DELIA_AVAILABLE = False
    print("Note: Delia not available, tool execution will be simulated")


@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class AgentConfig:
    """Agent configuration."""
    max_iterations: int = 10
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    verbose: bool = True


@dataclass
class AgentState:
    """Tracks agent state across iterations."""
    messages: list[dict] = field(default_factory=list)
    tool_calls_made: list[ToolCall] = field(default_factory=list)
    iteration: int = 0
    done: bool = False
    final_response: str = ""


class ToolValidator:
    """Validates tool calls against JSON Schema."""
    
    def __init__(self, tools_path: Path = TOOLS_SCHEMA_PATH):
        with open(tools_path) as f:
            tools = json.load(f)
        
        # Build lookup map
        self.tool_map: dict[str, dict] = {}
        self.tools = tools
        for tool in tools:
            name = tool["function"]["name"]
            self.tool_map[name] = tool
    
    def get_tool_names(self) -> list[str]:
        """Get list of valid tool names."""
        return list(self.tool_map.keys())
    
    def validate(self, tool_call: ToolCall) -> tuple[bool, str]:
        """Validate a tool call against its schema.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if tool_call.name not in self.tool_map:
            return False, f"Unknown tool: {tool_call.name}. Valid tools: {self.get_tool_names()}"
        
        schema = self.tool_map[tool_call.name]["function"]["parameters"]
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        # Check required fields
        for req in required:
            if req not in tool_call.arguments:
                return False, f"Missing required field '{req}' for {tool_call.name}"
        
        # Type validation
        for key, value in tool_call.arguments.items():
            if key not in properties:
                continue  # Extra fields are OK
            
            prop = properties[key]
            prop_type = prop.get("type")
            
            # Type checking
            if prop_type == "string" and not isinstance(value, str):
                return False, f"Field '{key}' should be string, got {type(value).__name__}"
            elif prop_type == "integer" and not isinstance(value, int):
                return False, f"Field '{key}' should be integer, got {type(value).__name__}"
            elif prop_type == "boolean" and not isinstance(value, bool):
                return False, f"Field '{key}' should be boolean, got {type(value).__name__}"
            elif prop_type == "number" and not isinstance(value, (int, float)):
                return False, f"Field '{key}' should be number, got {type(value).__name__}"
            
            # Enum validation
            if "enum" in prop and value not in prop["enum"]:
                return False, f"Field '{key}' value '{value}' not in {prop['enum']}"
            
            # Range validation
            if "minimum" in prop and isinstance(value, (int, float)):
                if value < prop["minimum"]:
                    return False, f"Field '{key}' value {value} below minimum {prop['minimum']}"
            if "maximum" in prop and isinstance(value, (int, float)):
                if value > prop["maximum"]:
                    return False, f"Field '{key}' value {value} above maximum {prop['maximum']}"
        
        return True, ""
    
    def get_schema(self, tool_name: str) -> dict | None:
        """Get schema for a specific tool."""
        if tool_name in self.tool_map:
            return self.tool_map[tool_name]
        return None


class ToolCallParser:
    """Parses tool calls from model output."""
    
    # XML format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    XML_PATTERN = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
    
    # Raw JSON format: {"name": "...", "arguments": {...}}
    JSON_PATTERN = re.compile(r'\{["\']?name["\']?\s*:\s*["\']([^"\']+)["\'].*?["\']?arguments["\']?\s*:\s*(\{[^}]+\})\}', re.DOTALL)
    
    @classmethod
    def parse(cls, text: str) -> list[ToolCall]:
        """Parse tool calls from text.
        
        Supports:
        1. XML format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        2. Raw JSON format
        """
        tool_calls = []
        
        # Try XML format first
        xml_matches = cls.XML_PATTERN.findall(text)
        for match in xml_matches:
            try:
                data = json.loads(match.strip())
                tool_calls.append(ToolCall(
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                    raw=match,
                ))
            except json.JSONDecodeError:
                continue
        
        # If no XML matches, try raw JSON
        if not tool_calls:
            # Look for tool call objects directly
            try:
                # Try to find JSON objects that look like tool calls
                for match in re.finditer(r'\{[^{}]*"name"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}', text, re.DOTALL):
                    try:
                        data = json.loads(match.group())
                        tool_calls.append(ToolCall(
                            name=data.get("name", ""),
                            arguments=data.get("arguments", {}),
                            raw=match.group(),
                        ))
                    except json.JSONDecodeError:
                        continue
            except Exception:
                pass
        
        return tool_calls
    
    @classmethod
    def has_tool_call(cls, text: str) -> bool:
        """Check if text contains any tool calls."""
        return bool(cls.XML_PATTERN.search(text) or '"name"' in text and '"arguments"' in text)


class ToolExecutor:
    """Executes tool calls against Delia or simulates them."""
    
    def __init__(self, validator: ToolValidator, use_delia: bool = True):
        self.validator = validator
        self.use_delia = use_delia and DELIA_AVAILABLE
        
        if self.use_delia:
            self.registry = get_default_tools(allow_write=False, allow_exec=False)
        else:
            self.registry = None
    
    async def execute(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result."""
        # Validate first
        valid, error = self.validator.validate(tool_call)
        if not valid:
            return f"Error: {error}"
        
        if self.use_delia and self.registry:
            return await self._execute_delia(tool_call)
        else:
            return self._simulate(tool_call)
    
    async def _execute_delia(self, tool_call: ToolCall) -> str:
        """Execute using Delia's tool registry."""
        tool_def = self.registry.get(tool_call.name)
        
        if tool_def:
            try:
                result = await tool_def.handler(**tool_call.arguments)
                return str(result)
            except Exception as e:
                return f"Error executing {tool_call.name}: {e}"
        
        # Tool not in local registry - might be an MCP server tool
        return f"Tool '{tool_call.name}' not available in local registry"
    
    def _simulate(self, tool_call: ToolCall) -> str:
        """Simulate tool execution for testing."""
        name = tool_call.name
        args = tool_call.arguments
        
        # Simulated responses for common tools
        simulations = {
            "read_file": lambda a: f"[Simulated] Contents of {a.get('path', 'file')}:\n# Example file content",
            "list_directory": lambda a: f"[Simulated] Contents of {a.get('path', '.')}:\nfile1.py\nfile2.py\nsubdir/",
            "search_code": lambda a: f"[Simulated] Found 3 matches for '{a.get('pattern', '')}'",
            "health": lambda a: '{"status": "healthy", "backends": [{"id": "local", "available": true}]}',
            "models": lambda a: '{"backends": [{"id": "local", "models": {"quick": "gemma:2b", "coder": "qwen2.5:14b"}}]}',
            "delegate": lambda a: f"[Simulated] Completed {a.get('task', 'task')}: Analysis complete.",
            "think": lambda a: f"[Simulated] Thinking about: {a.get('problem', '')[:50]}...\n\nConclusion: This is a thoughtful response.",
        }
        
        if name in simulations:
            return simulations[name](args)
        
        return f"[Simulated] Executed {name} with {json.dumps(args)}"


class Agent:
    """The main agent that orchestrates model + tools."""
    
    def __init__(
        self,
        model,
        tokenizer,
        validator: ToolValidator,
        executor: ToolExecutor,
        config: AgentConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.validator = validator
        self.executor = executor
        self.config = config
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tool_desc = "You are an AI assistant with access to tools. Use tools to help the user.\n\n"
        tool_desc += "Available tools:\n"
        
        for tool in self.validator.tools[:10]:  # Limit to avoid context overflow
            func = tool["function"]
            tool_desc += f"\n### {func['name']}\n{func['description']}\n"
        
        tool_desc += """
To use a tool, respond with:
<tool_call>{"name": "tool_name", "arguments": {"arg1": "value1"}}</tool_call>

After receiving tool results, provide your final answer to the user.
"""
        return tool_desc
    
    def _format_messages(self, state: AgentState) -> str:
        """Format messages for the model."""
        parts = []
        
        # System prompt
        parts.append(f"<start_of_turn>system\n{self._build_system_prompt()}<end_of_turn>")
        
        # Conversation history
        for msg in state.messages:
            role = msg["role"]
            content = msg.get("content", "")
            
            if role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
            elif role == "tool":
                name = msg.get("name", "tool")
                parts.append(f"<start_of_turn>tool\n[{name}] {content}<end_of_turn>")
        
        # Generation prompt
        parts.append("<start_of_turn>model\n")
        
        return "\n".join(parts)
    
    def _generate(self, prompt: str) -> str:
        """Generate model response."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Clean up response
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]
        
        return response.strip()
    
    async def run(self, user_prompt: str) -> str:
        """Run the agent loop."""
        state = AgentState()
        state.messages.append({"role": "user", "content": user_prompt})
        
        while state.iteration < self.config.max_iterations and not state.done:
            state.iteration += 1
            
            if self.config.verbose:
                print(f"\n--- Iteration {state.iteration} ---")
            
            # Generate response
            prompt = self._format_messages(state)
            response = self._generate(prompt)
            
            if self.config.verbose:
                print(f"Model: {response[:200]}...")
            
            # Check for tool calls
            tool_calls = ToolCallParser.parse(response)
            
            if tool_calls:
                # Execute tool calls
                for tc in tool_calls:
                    if self.config.verbose:
                        print(f"Tool call: {tc.name}({json.dumps(tc.arguments)})")
                    
                    result = await self.executor.execute(tc)
                    
                    if self.config.verbose:
                        print(f"Result: {result[:200]}...")
                    
                    # Add to state
                    state.tool_calls_made.append(tc)
                    state.messages.append({
                        "role": "assistant",
                        "content": response,
                    })
                    state.messages.append({
                        "role": "tool",
                        "name": tc.name,
                        "content": result,
                    })
            else:
                # No tool call - this is the final response
                state.done = True
                state.final_response = response
                state.messages.append({
                    "role": "assistant",
                    "content": response,
                })
        
        if not state.done:
            state.final_response = "Max iterations reached without completing the task."
        
        return state.final_response


def load_model(model_path: str):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    
    model.eval()
    return model, tokenizer


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run fine-tuned FunctionGemma agent")
    parser.add_argument("--model", default="google/gemma-2b-it", help="Model path or name")
    parser.add_argument("--prompt", help="Single prompt to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_iterations", type=int, default=10, help="Max tool iterations")
    parser.add_argument("--simulate", action="store_true", help="Simulate tool execution")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FunctionGemma Agent for Delia Tool Orchestration")
    print("=" * 60)
    
    # Load components
    validator = ToolValidator()
    print(f"Loaded {len(validator.tools)} tools from schema")
    
    model, tokenizer = load_model(args.model)
    
    executor = ToolExecutor(validator, use_delia=not args.simulate)
    
    config = AgentConfig(
        max_iterations=args.max_iterations,
        verbose=args.verbose,
    )
    
    agent = Agent(model, tokenizer, validator, executor, config)
    
    if args.prompt:
        # Single prompt mode
        print(f"\nProcessing: {args.prompt}")
        result = await agent.run(args.prompt)
        print(f"\n{'=' * 60}")
        print("Final Response:")
        print(result)
    
    elif args.interactive:
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit", "q"]:
                    break
                if not user_input:
                    continue
                
                result = await agent.run(user_input)
                print(f"\nAssistant: {result}")
            except KeyboardInterrupt:
                break
        print("\nGoodbye!")
    
    else:
        # Demo mode
        print("\nRunning demo...")
        demo_prompt = "What's the health status of the backends?"
        print(f"Prompt: {demo_prompt}")
        result = await agent.run(demo_prompt)
        print(f"\n{'=' * 60}")
        print("Final Response:")
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
