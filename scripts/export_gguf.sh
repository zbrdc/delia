#!/usr/bin/env bash
# export_gguf.sh - Export LoRA-merged model to GGUF format
# Requires: llama.cpp installed (make or cmake build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/functiongemma-delia"
OUTPUT_DIR="${SCRIPT_DIR}/models/gguf"
LLAMA_CPP="${LLAMA_CPP_PATH:-$HOME/llama.cpp}"

# Check prerequisites
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model not found at $MODEL_DIR"
    echo "   Run train_lora.py first"
    exit 1
fi

if [ ! -f "$LLAMA_CPP/convert_hf_to_gguf.py" ]; then
    echo "❌ llama.cpp not found at $LLAMA_CPP"
    echo "   Set LLAMA_CPP_PATH or install: git clone https://github.com/ggerganov/llama.cpp"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "📦 Converting to GGUF..."

# Convert HF model to GGUF (f16 base)
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outfile "$OUTPUT_DIR/functiongemma-delia-f16.gguf" \
    --outtype f16

echo "✅ F16 model: $OUTPUT_DIR/functiongemma-delia-f16.gguf"

# Quantize to Q4_K_M (good balance of speed/quality)
if [ -f "$LLAMA_CPP/llama-quantize" ]; then
    echo "🔧 Quantizing to Q4_K_M..."
    "$LLAMA_CPP/llama-quantize" \
        "$OUTPUT_DIR/functiongemma-delia-f16.gguf" \
        "$OUTPUT_DIR/functiongemma-delia-Q4_K_M.gguf" \
        Q4_K_M
    echo "✅ Q4_K_M model: $OUTPUT_DIR/functiongemma-delia-Q4_K_M.gguf"
    
    echo "🔧 Quantizing to Q8_0..."
    "$LLAMA_CPP/llama-quantize" \
        "$OUTPUT_DIR/functiongemma-delia-f16.gguf" \
        "$OUTPUT_DIR/functiongemma-delia-Q8_0.gguf" \
        Q8_0
    echo "✅ Q8_0 model: $OUTPUT_DIR/functiongemma-delia-Q8_0.gguf"
else
    echo "⚠️  llama-quantize not found, build llama.cpp with: make llama-quantize"
fi

echo ""
echo "📊 Model sizes:"
ls -lh "$OUTPUT_DIR"/*.gguf 2>/dev/null || true

echo ""
echo "🚀 Test with: llama-cli -m $OUTPUT_DIR/functiongemma-delia-Q4_K_M.gguf -p '<prompt>'"
