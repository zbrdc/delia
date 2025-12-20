#!/usr/bin/env python3
"""
Convert Fine-Tuned Model to GGUF for Ollama

This script converts the merged HuggingFace model to GGUF format
that can be imported into Ollama.

Requirements:
    pip install llama-cpp-python transformers

Usage:
    python convert_to_gguf.py --input ./outputs/final/merged --output ./outputs/gguf
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Convert model to GGUF for Ollama")
    parser.add_argument(
        "--input",
        type=str,
        default="./outputs/final/merged",
        help="Path to merged HuggingFace model",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/gguf",
        help="Output directory for GGUF files",
    )
    parser.add_argument(
        "--quantizations",
        type=str,
        default="f16,q4_k_m,q8_0",
        help="Comma-separated quantization types",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        print("\nMake sure you have trained the model first:")
        print("  python train_lora.py --merge")
        sys.exit(1)

    print("=" * 60)
    print("GGUF Conversion for Ollama")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print()

    # Method 1: Try using llama.cpp's convert script directly
    # You need llama.cpp cloned somewhere
    llama_cpp_path = Path.home() / "llama.cpp"
    convert_script = llama_cpp_path / "convert_hf_to_gguf.py"

    if convert_script.exists():
        print("✅ Found llama.cpp conversion script")
        
        for quant in args.quantizations.split(","):
            quant = quant.strip().lower()
            output_file = output_path / f"functiongemma-delia-{quant}.gguf"
            
            print(f"\n📦 Converting to {quant}...")
            
            if quant == "f16":
                # F16 is the base conversion
                cmd = [
                    sys.executable,
                    str(convert_script),
                    str(input_path),
                    "--outfile", str(output_file),
                    "--outtype", "f16",
                ]
            else:
                # First convert to f16, then quantize
                f16_file = output_path / "functiongemma-delia-f16.gguf"
                if not f16_file.exists():
                    print("  First creating F16 base...")
                    cmd = [
                        sys.executable,
                        str(convert_script),
                        str(input_path),
                        "--outfile", str(f16_file),
                        "--outtype", "f16",
                    ]
                    subprocess.run(cmd, check=True)
                
                # Now quantize
                quantize_bin = llama_cpp_path / "build" / "bin" / "llama-quantize"
                if quantize_bin.exists():
                    cmd = [str(quantize_bin), str(f16_file), str(output_file), quant.upper()]
                else:
                    print(f"  ⚠️ llama-quantize not found, skipping {quant}")
                    continue
            
            try:
                subprocess.run(cmd, check=True)
                print(f"  ✅ Created: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed: {e}")
    else:
        print("⚠️ llama.cpp not found at ~/llama.cpp")
        print()
        print("To convert to GGUF, you have two options:")
        print()
        print("OPTION A: Install llama.cpp")
        print("-" * 40)
        print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("  cd ~/llama.cpp && make")
        print("  pip install -r requirements.txt")
        print("  python convert_hf_to_gguf.py", str(input_path), "--outfile", str(output_path / "functiongemma-delia-f16.gguf"))
        print()
        print("OPTION B: Use HuggingFace's gguf library")
        print("-" * 40)
        print("  pip install gguf")
        print("  # Then use transformers to export:")
        print('  from transformers import AutoModelForCausalLM')
        print(f'  model = AutoModelForCausalLM.from_pretrained("{input_path}")')
        print(f'  model.save_pretrained("{output_path}", gguf_file="functiongemma-delia.gguf")')
        print()
        print("OPTION C: Use Ollama's built-in import (easiest!)")
        print("-" * 40)
        print("  Ollama can import safetensors directly in newer versions.")
        print("  See the instructions below.")


    print()
    print("=" * 60)
    print("NEXT STEPS: Import into Ollama")
    print("=" * 60)
    print()
    print("1. Create a Modelfile:")
    print()
    
    modelfile_content = f'''FROM {output_path}/functiongemma-delia-q4_k_m.gguf

# Or import directly from safetensors:
# FROM {input_path}

TEMPLATE """{{{{ if .System }}}}<start_of_turn>developer
{{{{ .System }}}}
<end_of_turn>
{{{{ end }}}}<start_of_turn>user
{{{{ .Prompt }}}}
<end_of_turn>
<start_of_turn>model
"""

PARAMETER stop <end_of_turn>
PARAMETER stop <start_of_turn>
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
'''
    
    modelfile_path = output_path / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    print(f"   Created: {modelfile_path}")
    print()
    print("2. Import into Ollama:")
    print(f"   cd {output_path}")
    print("   ollama create functiongemma-delia -f Modelfile")
    print()
    print("3. Test it:")
    print('   ollama run functiongemma-delia "Check backend health"')
    print()
    print("4. Configure in Delia settings.json:")
    print('   "model_dispatcher": {')
    print('     "name": "functiongemma-delia",')
    print('     "num_ctx": 2048')
    print('   }')
    print()
    print("NOTE: The model name MUST contain 'functiongemma' for Delia")
    print("      to apply the correct prompt formatting!")


if __name__ == "__main__":
    main()
