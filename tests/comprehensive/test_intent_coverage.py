# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from delia.orchestration.intent import IntentDetector
from delia.orchestration.result import OrchestrationMode, ModelRole

@pytest.fixture
def detector():
    return IntentDetector()

# Exhaustive Matrix aligned with optimized architecture
INTENT_SCENARIOS = [
    ("read src/main.py", OrchestrationMode.AGENTIC, "coder"),
    ("list files in the current directory", OrchestrationMode.AGENTIC, "quick"),
    ("grep for 'todo' in the repo", OrchestrationMode.AGENTIC, "coder"),
    ("run the tests", OrchestrationMode.AGENTIC, "coder"),
    ("npm install", OrchestrationMode.AGENTIC, "coder"),
    ("create a new file called test.py", OrchestrationMode.AGENTIC, "coder"),
    ("update the version in package.json", OrchestrationMode.AGENTIC, "coder"),
    ("execute ls -la", OrchestrationMode.AGENTIC, "coder"), # Technical tool execution
    ("find where the database is initialized", OrchestrationMode.AGENTIC, "coder"),
    ("search the web for the latest news on Delia", OrchestrationMode.AGENTIC, "quick"),
    ("look up the docs for FastAPI online", OrchestrationMode.AGENTIC, "quick"),
    ("what are the contents of .env?", OrchestrationMode.AGENTIC, "quick"),
    ("modify the header in index.html", OrchestrationMode.AGENTIC, "coder"),
    ("save this output to results.txt", OrchestrationMode.AGENTIC, "coder"),
    ("delete the temp directory", OrchestrationMode.AGENTIC, "coder"), # Destructive tool op
    ("git commit -m 'test'", OrchestrationMode.AGENTIC, "coder"),
    ("docker build .", OrchestrationMode.AGENTIC, "coder"),
    ("where am i?", OrchestrationMode.AGENTIC, "quick"),
    ("pwd", OrchestrationMode.AGENTIC, "quick"),
    ("what directory is this?", OrchestrationMode.AGENTIC, "quick"),
    
    ("verify if this code is correct", OrchestrationMode.VOTING, "coder"),
    ("make sure this doesn't have bugs", OrchestrationMode.VOTING, "coder"),
    ("double check the math", OrchestrationMode.VOTING, "quick"),
    ("validate this security logic", OrchestrationMode.VOTING, "coder"),
    ("confirm if the plan is sound", OrchestrationMode.VOTING, "quick"),
    ("i need a very reliable answer", OrchestrationMode.VOTING, "quick"),
    ("is this accurate? verify it", OrchestrationMode.VOTING, "quick"),
    ("ensure this follows the standards", OrchestrationMode.VOTING, "quick"),
    ("this is critical, double check", OrchestrationMode.VOTING, "quick"),
    ("can you confirm this is safe?", OrchestrationMode.VOTING, "quick"),
    
    ("compare qwen and llama", OrchestrationMode.COMPARISON, "quick"),
    ("vs GPT-4", OrchestrationMode.COMPARISON, "quick"),
    ("what do different models think of this?", OrchestrationMode.COMPARISON, "quick"),
    ("contrast these two approaches", OrchestrationMode.COMPARISON, "quick"),
    ("give me a second opinion", OrchestrationMode.COMPARISON, "quick"),
    ("which model is better for this task?", OrchestrationMode.COMPARISON, "quick"),
    ("side by side comparison", OrchestrationMode.COMPARISON, "quick"),
    
    ("think deeply about the architecture", OrchestrationMode.DEEP_THINKING, "moe"),
    ("step by step analysis", OrchestrationMode.DEEP_THINKING, "moe"),
    ("thorough architectural review", OrchestrationMode.DEEP_THINKING, "moe"),
    ("explore all possible solutions", OrchestrationMode.TREE_OF_THOUGHTS, "moe"),
    ("use tree of thoughts to solve this", OrchestrationMode.TREE_OF_THOUGHTS, "thinking"),
    ("what are the trade-offs?", OrchestrationMode.DEEP_THINKING, "moe"),
    ("plan a migration strategy", OrchestrationMode.DEEP_THINKING, "moe"),
    ("architect a scalable backend", OrchestrationMode.DEEP_THINKING, "moe"),
    
    ("write a python script", OrchestrationMode.AGENTIC, "coder"),
    ("fix the bug in main.py", OrchestrationMode.AGENTIC, "coder"),
    ("refactor this function", OrchestrationMode.NONE, "coder"),
    ("optimize this loop", OrchestrationMode.NONE, "coder"),
    ("debug the crash", OrchestrationMode.NONE, "coder"),
    ("implement a binary tree", OrchestrationMode.NONE, "coder"),
    ("review my code", OrchestrationMode.NONE, "coder"),
    ("analyze the performance bottleneck", OrchestrationMode.NONE, "coder"),
    
    ("hello delia", OrchestrationMode.NONE, "quick"),
    ("who are you?", OrchestrationMode.NONE, "quick"),
    ("show melons", OrchestrationMode.NONE, "status"),
    ("/leaderboard", OrchestrationMode.NONE, "status"),
    ("melon rankings", OrchestrationMode.NONE, "status"),
    ("what is the weather?", OrchestrationMode.NONE, "quick"),
    ("summarize this text", OrchestrationMode.NONE, "quick"),
    ("ok", OrchestrationMode.NONE, "quick"),
    ("thanks", OrchestrationMode.NONE, "quick"),
    ("bye", OrchestrationMode.NONE, "quick"),
    
    ("first read the file then fix the bug then test it", OrchestrationMode.AGENTIC, "coder"),
    ("1. analyze 2. generate 3. review", OrchestrationMode.CHAIN, "coder"),
    ("Step 1: plan. Step 2: implement.", OrchestrationMode.AGENTIC, "coder"),
    ("please verify the code in src/ and then run the tests", OrchestrationMode.AGENTIC, "coder"),
]

# More scenarios for volume
for i in range(10):
    INTENT_SCENARIOS.append((f"test variation {i} for read_file", OrchestrationMode.AGENTIC, "coder"))
    INTENT_SCENARIOS.append((f"make sure variation {i} is accurate", OrchestrationMode.VOTING, "quick"))
    INTENT_SCENARIOS.append((f"compare {i} models", OrchestrationMode.COMPARISON, "quick"))
    INTENT_SCENARIOS.append((f"write {i} lines of code", OrchestrationMode.AGENTIC, "coder"))
    INTENT_SCENARIOS.append((f"hello {i}", OrchestrationMode.NONE, "quick"))

@pytest.mark.parametrize("prompt, expected_mode, expected_tier", INTENT_SCENARIOS)
def test_intent_detection_matrix(detector, prompt, expected_mode, expected_tier):
    intent = detector.detect(prompt)
    assert intent.orchestration_mode == expected_mode
    if expected_tier != "any":
        assert intent.task_type == expected_tier

def test_intent_confidence_layering(detector):
    base = detector.detect("verify")
    high = detector.detect("critically verify this high-stakes code logic immediately")
    assert high.confidence >= base.confidence

def test_intent_reasoning_generation(detector):
    intent = detector.detect("run agent loop")
    assert len(intent.reasoning) > 5
    assert intent.reasoning != "default"
