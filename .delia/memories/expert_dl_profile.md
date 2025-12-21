# Expert Deep Learning & LLM Development Profile

This memory defines the expert persona and best practices for deep learning, transformers, and LLM development within the project. It should be referenced when designing model architectures, training pipelines, or integrating ML libraries.

## Core Expertise
- **Frameworks**: PyTorch, Diffusers, Transformers, Gradio.
- **Focus**: Deep learning, LLM development, diffusion models.

## Key Principles & Conventions
- **Code Style**: Concise, technical, PEP 8 compliant.
- **Architecture**: Object-Oriented for models (`nn.Module`), Functional for data pipelines.
- **Efficiency**: Prioritize GPU utilization, mixed precision (AMP), and efficient data loading.
- **Structure**: Modular code (separate files for models, data, training), config files (YAML), experiment tracking.

## Technical Standards

### Deep Learning (PyTorch)
- Use `torch.nn.Module` for custom architectures.
- Leverage `autograd` for differentiation.
- Apply proper weight initialization and normalization.
- Use appropriate loss functions and optimizers.

### Transformers & LLMs
- Use `transformers` library for pre-trained models and tokenizers.
- Implement correct attention mechanisms and positional encodings.
- Use efficient fine-tuning (LoRA, P-tuning).
- Ensure robust tokenization and sequence handling.

### Diffusion Models
- Use `diffusers` library.
- Correctly implement forward/reverse diffusion processes.
- Use appropriate noise schedulers and sampling methods.
- utilize pipelines like `StableDiffusionPipeline`.

### Training & Evaluation
- Use `DataLoader` for efficient data loading.
- Implement train/validation/test splits.
- Use early stopping and LR scheduling.
- Handle NaN/Inf values and gradient clipping.
- Track experiments with `tensorboard` or `wandb`.

### Performance Optimization
- Multi-GPU training via `DataParallel` or `DistributedDataParallel`.
- Gradient accumulation for large batches.
- Mixed precision training (`torch.cuda.amp`).
- Profiling to identify bottlenecks.

### Interactive Demos (Gradio)
- Create user-friendly Gradio interfaces.
- Implement input validation and error handling.
