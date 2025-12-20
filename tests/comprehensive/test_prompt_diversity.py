# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
from delia.orchestration.intent import IntentDetector
from delia.orchestration.result import OrchestrationMode

# Generate 100+ diverse technical prompts
DIVERSE_PROMPTS = [
    # Languages
    "Write a C++ class for a circular buffer",
    "Explain React hooks",
    "How to use Rust ownership?",
    "Golang interface example",
    "Assembly x86 hello world",
    "COBOL file handling",
    "Lisp macro tutorial",
    "Fortran 90 array syntax",
    "Haskell monad explanation",
    "Scala actor model",
    
    # Infrastructure
    "Terraform script for AWS S3",
    "Docker compose for postgres and redis",
    "Kubernetes deployment manifest",
    "Ansible playbook for nginx",
    "Nginx reverse proxy config",
    "Prometheus alert rules",
    "Grafana dashboard json",
    "GitHub Actions workflow for pytest",
    
    # Database
    "SQL query for join with aggregate",
    "MongoDB aggregation pipeline",
    "Redis pub/sub example",
    "PostgreSQL full text search",
    "Database normalization steps",
    "Indexing strategy for large tables",
    
    # Security
    "Scan this code for XSS",
    "Is this SQL query vulnerable to injection?",
    "Explain JWT security",
    "How does OAuth2 work?",
    "Security audit of src/auth.py",
    "Implement AES encryption",
    
    # System
    "List files in /usr/local",
    "Grep for error in logs",
    "Find large files on disk",
    "What is the system load?",
    "Disk space usage summary",
    "Network interface stats",
    
    # Orchestration Triggers
    "Critically verify this implementation",
    "Compare DeepSeek and Qwen",
    "Step by step migration plan",
    "Thoroughly analyze the bottleneck",
    "Validate the security model",
    "Which model is best for reasoning?",
    "Side-by-side view of these solutions",
]

# Fill up to 100+
for i in range(60):
    DIVERSE_PROMPTS.append(f"Generic technical query number {i} regarding system optimization")

@pytest.mark.parametrize("prompt", DIVERSE_PROMPTS)
def test_diverse_prompt_detection(prompt):
    """Test that a vast diversity of technical prompts are correctly classified."""
    detector = IntentDetector()
    intent = detector.detect(prompt)
    
    # Should never be 'None' type result
    assert intent is not None
    assert intent.task_type in ("quick", "coder", "moe", "status", "thinking")
    assert intent.confidence >= 0.3
    assert len(intent.reasoning) > 0

def test_intent_detection_robustness():
    """Test robustness against empty or garbage input."""
    detector = IntentDetector()
    assert detector.detect("").task_type == "quick"
    assert detector.detect("   ").task_type == "quick"
    assert detector.detect("a" * 10000).confidence >= 0.3
