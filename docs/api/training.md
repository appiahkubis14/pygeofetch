# Training Framework

Full-featured training framework for geospatial AI: single-GPU, multi-GPU DDP, FSDP, mixed precision, and resumable checkpoints.

---

## GeoTrainer

The main training class — handles the training loop, logging, callbacks, and validation.

```python
from pygeovision.training.trainer import GeoTrainer
from pygeovision.models import get_model

model = get_model("segformer-b2", num_classes=7, in_channels=4)

trainer = GeoTrainer(
    model=model,
    task="segmentation",         # "segmentation" | "detection" | "classification"
    num_classes=7,
    epochs=100,
    learning_rate=1e-4,
    weight_decay=0.01,
    loss_fn=None,                # Auto-selected from task if None
    device="cuda",               # "cuda" | "cpu" | "mps"
    mixed_precision="bf16",      # "bf16" | "fp16" | "fp32"
    batch_size=16,
    num_workers=8,
    checkpoint_dir="./checkpoints/",
    log_dir="./logs/",
)

# Start training
history = trainer.fit(train_dataloader, val_dataloader)

# Best results
print(f"Best epoch: {history['best_epoch']}")
print(f"Best val_iou: {history['best_metrics']['val_iou']:.4f}")
```

---

## Callbacks

### `EarlyStopping`

```python
from pygeovision.training.callbacks import EarlyStopping

cb = EarlyStopping(
    monitor="val_iou",       # Metric to watch
    patience=15,             # Epochs without improvement before stopping
    mode="max",              # "max" for accuracy, "min" for loss
    min_delta=0.001,         # Minimum improvement to count
    restore_best_weights=True,
)

trainer = GeoTrainer(model=model, callbacks=[cb])
```

### `ModelCheckpoint`

```python
from pygeovision.training.callbacks import ModelCheckpoint

cb = ModelCheckpoint(
    dirpath="./checkpoints/",
    filename="epoch{epoch:03d}_iou{val_iou:.4f}",
    monitor="val_iou",
    mode="max",
    save_top_k=3,            # Keep best 3 checkpoints
    save_last=True,
)
```

### `LearningRateMonitor`

```python
from pygeovision.training.callbacks import LearningRateMonitor

cb = LearningRateMonitor()
# Logs LR to metrics dict at each epoch: metrics["lr"] = current_lr
```

---

## Optimizers

```python
from pygeovision.training.optimizers import build_optimizer

optimizer = build_optimizer(
    model=model,
    name="adamw",          # "adamw" | "sgd" | "lion" | "adam" | "rmsprop"
    lr=1e-4,
    weight_decay=0.01,
    # Layer-wise learning rate decay (recommended for ViT backbones)
    layer_decay=0.75,
)
```

**Recommended settings by model family:**

| Family | Optimizer | LR | Weight Decay |
|--------|-----------|-----|-------------|
| ViT / SegFormer | AdamW | 6e-5 | 0.01 |
| ResNet / U-Net | AdamW | 1e-4 | 0.05 |
| DINOv3 (web) | AdamW | 1e-4 | 0.05 |
| DINOv3 (SAT) | AdamW | 1e-5 | 0.05 |
| Prithvi-EO-2.0 | AdamW | 5e-5 | 0.01 |
| YOLO | SGD | 1e-2 | 5e-4 |

---

## Schedulers

```python
from pygeovision.training.schedulers import build_scheduler

scheduler = build_scheduler(
    optimizer=optimizer,
    name="cosine",          # "cosine" | "step" | "onecycle" | "warmup_cosine"
    epochs=100,
    warmup_epochs=10,       # Linear warmup (recommended for transformers)
    min_lr=1e-6,
)
```

---

## Mixed Precision

```python
from pygeovision.training.mixed_precision import MixedPrecisionManager

mp = MixedPrecisionManager(precision="bf16")  # or "fp16"

for batch in dataloader:
    optimizer.zero_grad()
    with mp.autocast():
        loss = model(batch)
    mp.scale_and_step(loss, optimizer, clip_grad_norm=1.0)
    scheduler.step()
```

---

## Distributed Training

### Data Parallel (DDP) — recommended for most setups

```python
from pygeovision.training.distributed import launch_ddp, setup_distributed, wrap_ddp

def train_fn(rank, world_size):
    setup_distributed(rank=rank, world_size=world_size)
    model   = get_model("segformer-b2", num_classes=7)
    model   = wrap_ddp(model, device_ids=[rank])
    trainer = GeoTrainer(model=model, distributed=True)
    trainer.fit(train_dl, val_dl)

launch_ddp(train_fn, world_size=4)  # 4 GPUs
```

### Fully Sharded (FSDP) — for very large models (>1B params)

```python
from pygeovision.training.distributed import wrap_fsdp

model = wrap_fsdp(model, mixed_precision=True)
```

### Gradient Accumulation — simulate large batches on small GPUs

```python
from pygeovision.training.distributed import GradientAccumulator

acc = GradientAccumulator(steps=8)   # Effective batch = 8 × batch_size

for batch in dataloader:
    loss = model(batch)
    if acc.step(loss, optimizer):    # Returns True when optimizer.step() fires
        scheduler.step()
```

---

## Checkpoint Manager

```python
from pygeovision.training.checkpoint import CheckpointManager

cm = CheckpointManager(
    dirpath="./checkpoints/",
    keep_top_k=3,
    monitor="val_iou",
    mode="max",
)

# Save
cm.save(epoch=50, model=model, optimizer=optimizer,
        metrics={"val_iou": 0.843})

# Load best
state = cm.load_best(model, optimizer)
print(f"Loaded epoch: {state['epoch']}")
print(f"Best metrics: {cm.best_metrics}")

# Resume training
state = cm.load_last(model, optimizer, scheduler)
start_epoch = state["epoch"] + 1
```

---

## Metrics

```python
from pygeovision.training.metrics import (
    MeanIoU, F1Score, Accuracy, Precision, Recall
)

# Compute at validation
metric = MeanIoU(num_classes=7, ignore_index=255)
metric.update(predictions, targets)
iou_per_class = metric.compute()   # Tensor of shape (num_classes,)
mean_iou      = iou_per_class.mean().item()
metric.reset()
```
