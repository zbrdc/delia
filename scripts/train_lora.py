#!/usr/bin/env python3
"""
FunctionGemma 270M Fine-Tuning Script with Unsloth

Fine-tunes FunctionGemma for Delia tool orchestration using LoRA.
Designed for single-GPU training on consumer hardware.

Requirements:
    pip install unsloth transformers datasets peft accelerate bitsandbytes

Usage:
    python scripts/train_lora.py
    python scripts/train_lora.py --epochs 3 --batch_size 4
    python scripts/train_lora.py --resume_from ./outputs/checkpoint-500
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass

# Unsloth must be imported first for optimal performance
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Warning: Unsloth not installed. Using standard transformers.")
    UNSLOTH_AVAILABLE = False

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class CausalLMTrainer(Trainer):
    """Custom trainer that computes loss for models that don't return it (e.g., Gemma3)."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return (loss, outputs) if return_outputs else loss

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TRAIN_FILE = DATA_DIR / "train.jsonl"
EVAL_FILE = DATA_DIR / "eval.jsonl"


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for FunctionGemma 270M."""
    
    # Model - FunctionGemma 270M (based on Gemma 3, optimized for function calling)
    # This is ungated and doesn't require license acceptance
    model_name: str = "google/functiongemma-270m-it"
    # Note: For larger models, use "google/gemma-2b-it" (requires license acceptance)
    
    # LoRA configuration
    lora_r: int = 32  # LoRA rank - higher = more capacity, more VRAM
    lora_alpha: int = 64  # LoRA alpha - typically 2x rank
    lora_dropout: float = 0.05  # Dropout for regularization
    lora_target_modules: list = None  # Set in __post_init__
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Sequence length
    max_seq_length: int = 2048  # FunctionGemma works well with 2K
    
    # Hardware optimization
    use_4bit: bool = True  # 4-bit quantization for lower VRAM
    use_gradient_checkpointing: bool = True  # Trade compute for memory
    
    # Output
    output_dir: str = str(OUTPUT_DIR)
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Target attention and MLP layers for Gemma
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


def load_dataset(file_path: Path) -> Dataset:
    """Load JSONL dataset."""
    examples = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    return Dataset.from_list(examples)


def format_conversation(example: dict, tokenizer) -> dict:
    """Format a conversation example for training.
    
    Uses the tokenizer's chat template to format messages.
    For FunctionGemma, this handles tool calls appropriately.
    """
    messages = example.get("messages", [])
    tools = example.get("tools", [])
    
    # Build the conversation text using chat template
    # The template should handle tool calls in the messages
    try:
        # Try using apply_chat_template with tools
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback: manual formatting if template doesn't support tools
        text = format_conversation_manual(messages, tools)
    
    return {"text": text}


def format_conversation_manual(messages: list, tools: list) -> str:
    """Manual conversation formatting fallback.
    
    Format compatible with Gemma instruction-tuned models.
    """
    parts = []
    
    # Add tool definitions at the start if present
    if tools:
        tool_desc = "Available tools:\n"
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = json.dumps(func.get("parameters", {}), indent=2)
            tool_desc += f"\n### {name}\n{desc}\nParameters: {params}\n"
        parts.append(f"<start_of_turn>system\n{tool_desc}<end_of_turn>")
    
    # Format messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        
        if role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        
        elif role == "assistant":
            if tool_calls:
                # Format tool calls
                tc_text = ""
                for tc in tool_calls:
                    func = tc.get("function", tc)
                    name = func.get("name", "")
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    tc_text += f'<tool_call>{{"name": "{name}", "arguments": {json.dumps(args)}}}</tool_call>\n'
                
                if content:
                    parts.append(f"<start_of_turn>model\n{content}\n{tc_text}<end_of_turn>")
                else:
                    parts.append(f"<start_of_turn>model\n{tc_text}<end_of_turn>")
            else:
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        
        elif role == "tool":
            name = msg.get("name", "tool")
            parts.append(f"<start_of_turn>tool\n[{name}] {content}<end_of_turn>")
    
    return "\n".join(parts)


def tokenize_function(examples: dict, tokenizer, max_length: int) -> dict:
    """Tokenize examples for training with proper labels for causal LM."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    # For causal LM, labels are the same as input_ids
    # The trainer will shift them internally
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_model_unsloth(config: TrainingConfig):
    """Load model using Unsloth for optimized training."""
    print("Loading model with Unsloth optimization...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=config.use_4bit,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        random_state=42,
    )
    
    return model, tokenizer


def load_model_standard(config: TrainingConfig):
    """Load model using standard transformers + PEFT."""
    print("Loading model with standard transformers...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        load_in_4bit=config.use_4bit,
    )
    
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Fine-tune FunctionGemma for Delia")
    parser.add_argument("--model", default="google/functiongemma-270m-it", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--resume_from", help="Resume from checkpoint")
    parser.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        output_dir=args.output_dir,
        use_4bit=not args.no_4bit,
    )
    
    print("=" * 60)
    print("FunctionGemma Fine-Tuning for Delia Tool Orchestration")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  LoRA rank: {config.lora_r}")
    print(f"  LoRA alpha: {config.lora_alpha}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  4-bit quantization: {config.use_4bit}")
    print(f"  Output: {config.output_dir}")
    
    # Check for training data
    if not TRAIN_FILE.exists():
        print(f"\nError: Training data not found at {TRAIN_FILE}")
        print("Run 'python scripts/build_dataset.py' first to generate training data.")
        sys.exit(1)
    
    # Load model
    print("\n1. Loading model...")
    if UNSLOTH_AVAILABLE:
        model, tokenizer = load_model_unsloth(config)
    else:
        model, tokenizer = load_model_standard(config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Load datasets
    print("\n2. Loading datasets...")
    train_dataset = load_dataset(TRAIN_FILE)
    print(f"   Train: {len(train_dataset)} examples")
    
    eval_dataset = None
    if EVAL_FILE.exists():
        eval_dataset = load_dataset(EVAL_FILE)
        print(f"   Eval: {len(eval_dataset)} examples")
    
    # Format conversations
    print("\n3. Formatting conversations...")
    train_dataset = train_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: format_conversation(x, tokenizer),
            remove_columns=eval_dataset.column_names,
        )
    
    # Tokenize
    print("\n4. Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
            batched=True,
            remove_columns=["text"],
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit" if UNSLOTH_AVAILABLE else "adamw_torch",
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    # Create trainer (use custom trainer for Gemma3 loss computation)
    trainer = CausalLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n5. Starting training...")
    print("-" * 60)
    
    if args.resume_from:
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()
    
    # Save final model
    print("\n6. Saving model...")
    final_path = Path(config.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    
    # Save adapter
    model.save_pretrained(final_path / "adapter")
    tokenizer.save_pretrained(final_path / "adapter")
    print(f"   Adapter saved to {final_path / 'adapter'}")
    
    # Merge and save full model
    if UNSLOTH_AVAILABLE:
        print("   Merging LoRA weights into base model...")
        model.save_pretrained_merged(
            str(final_path / "merged"),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"   Merged model saved to {final_path / 'merged'}")
    else:
        print("   Merging LoRA weights...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(final_path / "merged")
        tokenizer.save_pretrained(final_path / "merged")
        print(f"   Merged model saved to {final_path / 'merged'}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  - Adapter: {final_path / 'adapter'}")
    print(f"  - Merged:  {final_path / 'merged'}")
    print(f"\nNext steps:")
    print(f"  1. Test with: python scripts/run_agent.py --model {final_path / 'merged'}")
    print(f"  2. Export to GGUF: bash scripts/export_gguf.sh {final_path / 'merged'}")


if __name__ == "__main__":
    main()
