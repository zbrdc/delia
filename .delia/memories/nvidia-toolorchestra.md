# NVIDIA ToolOrchestra

Source: https://developer.nvidia.com/blog/train-small-orchestration-agents-to-solve-big-problems/

## Core Concept

Small orchestrator model (8B) coordinates specialized tools/models. Outperforms monolithic models while being 2.5x more efficient.

## Key Design Principles

### 1. Multi-Objective RL Training
Balances three competing objectives:
- **Correctness** - accuracy of final answer
- **Efficiency** - cost and latency  
- **User preferences** - marked preferred tools

### 2. Small High-Quality Data > Large Generic Data
- 552 synthetic problems
- 1,296 training prompts
- Outperforms massive datasets through quality focus

### 3. Strategic Tool Selection
Routes to appropriate tool based on task:
- Basic tools (web search, code interpreter)
- Specialized LLMs (coding, math)
- Generalist LLMs (GPT-5, Claude)

Only calls expensive models ~40% of steps, uses cheaper options for rest.

## Results
- 37.1% on Humanity's Last Exam (beats GPT-5 at 35.1%)
- 2.5x more efficient than GPT-5
- 30% of the cost on Tau2-Bench

## Relevance to Delia Playbooks

| ToolOrchestra | Delia Equivalent |
|---------------|------------------|
| 1,296 training prompts | Playbook bullets |
| Multi-objective RL | ACE feedback loop (helpful/harmful) |
| User preferences | Project-specific patterns |
| Small high-quality data | "Smallest high-signal tokens" |

### Key Insight
**Quality over quantity**. 552 well-structured problems beat massive generic datasets.

Playbooks should be:
- **Small** - dozens of bullets, not hundreds
- **Learned** - from actual task outcomes, not seeded generically
- **High-signal** - methodology (HOW), not standards (WHAT)
