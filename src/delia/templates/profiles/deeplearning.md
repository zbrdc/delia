# Deep Learning Profile

Load this profile for: PyTorch, neural networks, transformers, training, GPU optimization.

## Project Structure

```
project/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py           # Base model class
│   │   ├── transformer.py
│   │   └── diffusion.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
├── configs/
│   └── train.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── notebooks/
└── experiments/
```

## Model Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Initialize layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
```

## Training Loop

```python
from torch.cuda.amp import autocast, GradScaler

def train_epoch(model, loader, optimizer, scheduler, device, config):
    model.train()
    scaler = GradScaler()  # Mixed precision

    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with autocast():
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

        # Scaled backward
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if batch_idx % config.log_interval == 0:
            log.info("train_step", loss=loss.item(), lr=scheduler.get_last_lr()[0])
```

## Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data = self._load_data(data_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
```

## Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train_distributed(rank, world_size, config):
    setup_distributed(rank, world_size)

    model = Model(config).to(rank)
    model = DDP(model, device_ids=[rank])

    # Training loop...

    dist.destroy_process_group()
```

## Efficient Fine-tuning (LoRA)

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()  # ~0.1% of total
```

## Best Practices

```
ALWAYS:
- Use mixed precision (torch.cuda.amp)
- Apply gradient clipping
- Monitor for NaN/Inf gradients
- Profile before optimizing
- Version control configs + checkpoints

AVOID:
- Loading full dataset into memory
- Blocking operations in data loading
- Ignoring validation metrics
- Training without reproducibility (set seeds)
```

## Performance Checklist

```
□ Enable mixed precision training
□ Use appropriate batch size for GPU memory
□ Enable gradient accumulation if needed
□ Use compiled model (torch.compile) for PyTorch 2.0+
□ Profile with torch.profiler
□ Consider gradient checkpointing for memory
```

