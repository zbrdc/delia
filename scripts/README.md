# FunctionGemma Fine-Tuning for Delia Tool Orchestration

Fine-tune Google's FunctionGemma 270M model to call Delia MCP tools using Unsloth LoRA.

## Quick Start

```bash
cd /home/ben/Documents/Projects/delia/scripts

# 1. Install dependencies
pip install -r requirements.txt

# 2. Build training dataset from test examples
python build_dataset.py

# 3. Train with LoRA (single GPU, ~10-30 min)
python train_lora.py

# 4. Test inference
python run_agent.py "List the files in the current directory"
```

## Files

| File | Purpose |
|------|---------|
| `tools.openai.json` | 31 Delia tools in OpenAI function-calling format |
| `build_dataset.py` | Harvests training examples from test files |
| `train_lora.py` | Unsloth LoRA training script |
| `run_agent.py` | Inference harness with schema validation |
| `export_gguf.sh` | Convert to GGUF for llama.cpp deployment |
| `requirements.txt` | Python dependencies |

## Training Pipeline

### 1. Dataset Generation

```bash
python build_dataset.py
```

Creates:
- `data/train.jsonl` - Training examples
- `data/eval.jsonl` - Evaluation examples  
- `data/stats.json` - Dataset statistics

Sources:
- `test_parser.py` - XML tool call format examples
- `test_native_tool_calling.py` - OpenAI native format
- `test_mcp_server.py` - MCP tool usage patterns

### 2. LoRA Training

```bash
# Default settings (good for RTX 3090/4090)
python train_lora.py

# Custom settings
python train_lora.py \
    --epochs 5 \
    --batch-size 4 \
    --lr 3e-4 \
    --lora-rank 32 \
    --max-seq-length 2048
```

LoRA Config:
- Rank: 16 (adjustable)
- Alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- ~0.5M trainable parameters

### 3. Inference

```bash
# Interactive mode
python run_agent.py

# Single query
python run_agent.py "Delegate a code review to the local backend"

# Custom settings
python run_agent.py \
    --model ./models/functiongemma-delia \
    --max-iterations 10 \
    --temperature 0.1
```

## Tool Schema

The model learns to call 31 tools across categories:

**MCP Server Tools (19)**
- `delegate` - Route tasks to LLM backends
- `think` - Deep reasoning with configurable depth
- `batch` - Parallel task execution
- `batch_vote` - Consensus from multiple models
- `chain` - Sequential task pipelines
- `workflow` - Complex multi-step workflows
- `agent` - Autonomous agent execution
- `session_*` - Conversation session management
- `health`, `models`, `queue_status` - System monitoring

**Builtin Agent Tools (12)**
- `read_file`, `write_file`, `delete_file` - File operations
- `list_directory`, `search_code` - Code navigation
- `web_fetch`, `web_search`, `web_news` - Web access
- `shell_exec` - Command execution
- `ask_user` - User interaction

## Chat Template

FunctionGemma uses a specific format:

```
<bos><start_of_turn>user
[Available Tools]
{tool_definitions}

{user_query}<end_of_turn>
<start_of_turn>model
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call><end_of_turn>
<start_of_turn>user
[Tool Result]
{result}<end_of_turn>
<start_of_turn>model
{final_response}<end_of_turn>
```

## GGUF Export (Optional)

For deployment with llama.cpp:

```bash
# Set path to llama.cpp
export LLAMA_CPP_PATH=~/llama.cpp

# Export and quantize
chmod +x export_gguf.sh
./export_gguf.sh
```

Creates:
- `models/gguf/functiongemma-delia-f16.gguf` - Full precision
- `models/gguf/functiongemma-delia-Q4_K_M.gguf` - 4-bit quantized
- `models/gguf/functiongemma-delia-Q8_0.gguf` - 8-bit quantized

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Query     │────▶│  FunctionGemma   │────▶│  Tool Call      │
└─────────────────┘     │  + LoRA Adapter  │     │  (JSON)         │
                        └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  Schema          │◀─────────────┤
                        │  Validation      │              │
                        └────────┬─────────┘              │
                                 │ valid                  │
                        ┌────────▼─────────┐              │
                        │  MCP/Direct      │              │
                        │  Execution       │              │
                        └────────┬─────────┘              │
                                 │                        │
                        ┌────────▼─────────┐              │
                        │  Result          │──────────────┘
                        │  (loop back)     │
                        └──────────────────┘
```

## Hyperparameter Guidance

| GPU VRAM | Batch Size | Seq Length | Notes |
|----------|------------|------------|-------|
| 8 GB     | 1-2        | 1024       | Use gradient checkpointing |
| 12 GB    | 2-4        | 1536       | Default settings |
| 24 GB    | 4-8        | 2048       | Can increase rank to 32 |

## Extending

### Add New Tools

1. Add tool definition to `tools.openai.json`
2. Implement executor in `run_agent.py` (direct or via MCP)
3. Add training examples to test files
4. Rebuild dataset and retrain

### Custom Training Data

Add JSONL files to `data/` with format:
```json
{"text": "<formatted_conversation>", "source": "custom", "tool_calls": ["tool_name"]}
```

## Troubleshooting

**OOM during training**: Reduce `--batch-size` or `--max-seq-length`

**Poor tool selection**: Increase training epochs or add more examples

**Invalid JSON output**: Lower temperature or use constrained decoding

**GGUF export fails**: Ensure llama.cpp is built with: `make clean && make`
