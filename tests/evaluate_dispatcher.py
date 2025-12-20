# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import asyncio
import json
import time
from unittest.mock import MagicMock
from delia.llm import call_llm, init_llm_module
from delia.queue import ModelQueue
from delia.prompts import ModelRole, build_system_prompt

async def benchmark_model_raw(model_path, grammar_path):
    print(f"\n📊 Benchmarking Raw Model: {model_path} with GBNF")
    print("-" * 50)
    
    test_cases = [
        {"name": "Python Fibonacci", "prompt": "Write a python function to calculate fibonacci numbers.", "expected": "call_executor"},
        {"name": "Microservices Arch", "prompt": "Design a high-availability microservices system for a global banking platform.", "expected": "call_planner"},
        {"name": "React Hooks", "prompt": "How do I use useEffect in React?", "expected": "call_executor"},
        {"name": "Data Migration Plan", "prompt": "Create a 6-month migration strategy from Oracle to PostgreSQL.", "expected": "call_planner"}
    ]
    
    success_count = 0
    
    for case in test_cases:
        start_time = time.time()
        
        # Construct raw prompt in the format the model was trained on
        # For FunctionGemma, we include tool descriptions in the system block
        tool_desc = "Available tools:\n### call_planner\nPlan tasks.\n### call_executor\nWrite code."
        full_prompt = f"<start_of_turn>system\n{tool_desc}<end_of_turn>\n<start_of_turn>user\n{case['prompt']}<end_of_turn>\n<start_of_turn>model\n"
        
        # Execute llama-cli directly
        cmd = [
            "/home/dan/llama.cpp/build/bin/llama-cli",
            "-m", model_path,
            "--grammar-file", grammar_path,
            "-p", full_prompt,
            "-n", "128",
            "--quiet"
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        elapsed = time.time() - start_time
        
        response_text = stdout.decode().strip()
        
        tool_name = "none"
        if "<tool_call>" in response_text:
            try:
                call_json = response_text.split("<tool_call>")[1].split("</tool_call>")[0]
                tool_name = json.loads(call_json).get("name", "none")
            except: pass
        
        success = tool_name == case["expected"]
        if success: success_count += 1
        
        status = "✅" if success else "❌"
        print(f"[{status}] {case['name']:<20} | Got: {tool_name:<15} | Expected: {case['expected']:<15} | {elapsed:.2f}s")
        if not success:
            print(f"    Raw Response: {response_text[:100]}...")

    print(f"\n📈 GBNF Score: {success_count}/{len(test_cases)}")
    return success_count

async def run_eval():
    # 1. Test with the GBNF grammar
    model_path = "./models/functiongemma-gguf/functiongemma-delia-f16.gguf"
    grammar_path = "./dispatcher.gbnf"
    
    await benchmark_model_raw(model_path, grammar_path)

if __name__ == "__main__":
    asyncio.run(run_eval())