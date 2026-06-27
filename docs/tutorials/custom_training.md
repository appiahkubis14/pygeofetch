# Custom Training

Fine-tune any PyGeoVision model on your own geospatial dataset using the GeoTrainer.

---

## Dataset Preparation

Organise your data into the standard structure:

```
my_dataset/
├── images/
│   ├── train/   (GeoTIFF files)
│   ├── val/
│   └── test/
└── labels/
    ├── train/   (Single-band GeoTIFF, uint8 class IDs)
    ├── val/
    └── test/
```

---

## 1. Define the Dataset

```python
import torch
from torch.utils.data import Dataset
import rasterio, numpy as np
from pathlib import Path

class GeoSegDataset(Dataset):
    def __init__(self, image_dir, label_dir, chip_size=512):
        self.images = sorted(Path(image_dir).glob("*.tif"))
        self.labels = sorted(Path(label_dir).glob("*.tif"))
        self.chip_size = chip_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with rasterio.open(self.images[idx]) as src:
            img = src.read().astype(np.float32)
            # Normalise per-band to [0,1]
            for b in range(img.shape[0]):
                img[b] = (img[b] - img[b].min()) / (img[b].max() - img[b].min() + 1e-8)

        with rasterio.open(self.labels[idx]) as src:
            label = src.read(1).astype(np.int64)

        return torch.tensor(img), torch.tensor(label)

train_ds = GeoSegDataset("./data/images/train/", "./data/labels/train/")
val_ds   = GeoSegDataset("./data/images/val/",   "./data/labels/val/")

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4)
val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4)
```

---

## 2. Load Model

```python
from pygeovision.models import get_model

model = get_model(
    "segformer-b2",
    num_classes=7,      # Your number of classes
    in_channels=4,      # Your input band count
    pretrained=True,    # Fine-tune from ImageNet weights
)
```

---

## 3. Configure Training

```python
from pygeovision.training.trainer   import GeoTrainer
from pygeovision.training.callbacks import EarlyStopping, ModelCheckpoint

trainer = GeoTrainer(
    model=model,
    task="segmentation",
    num_classes=7,
    epochs=100,
    learning_rate=6e-5,       # Lower LR for transformer fine-tuning
    weight_decay=0.01,
    mixed_precision="bf16",   # BF16 for modern GPUs
    device="cuda",
    callbacks=[
        EarlyStopping(monitor="val_iou", patience=15, mode="max"),
        ModelCheckpoint("./checkpoints/", monitor="val_iou", mode="max", save_top_k=3),
    ],
)
```

---

## 4. Train

```python
history = trainer.fit(train_dl, val_dl)

print(f"Best epoch:   {history['best_epoch']}")
print(f"Best val_iou: {history['best_metrics']['val_iou']:.4f}")
print(f"Best val_f1:  {history['best_metrics']['val_f1']:.4f}")
```

---

## 5. Fine-Tune a Foundation Model

For the highest accuracy, fine-tune DINOv3 SAT:

```python
from pygeovision.models.foundation.dinov3 import finetune_dinov3

result = finetune_dinov3(
    model_name="dinov3_vitl16_sat",
    task="segmentation",
    num_classes=7,
    epochs=50,
    learning_rate=1e-5,      # Lower LR for SAT-pretrained models
    weight_decay=0.05,
    mixed_precision=True,
    output_dir="./checkpoints/dinov3_finetune/",
)

ft_model   = result["model"]
optimizer  = result["optimizer"]
scheduler  = result["scheduler"]
mp_manager = result["mp_manager"]

# Custom training loop with fine-tuned model
for epoch in range(50):
    ft_model.train()
    for images, labels in train_dl:
        optimizer.zero_grad()
        with mp_manager.autocast():
            outputs = ft_model(images.to("cuda"))
            loss = criterion(outputs, labels.to("cuda"))
        mp_manager.scale_and_step(loss, optimizer)
    scheduler.step()
```

---

## 6. Evaluate

```python
from pygeovision.training.metrics import MeanIoU, F1Score

miou = MeanIoU(num_classes=7)
f1   = F1Score(num_classes=7)

model.eval()
with torch.no_grad():
    for images, labels in val_dl:
        preds = model(images.to("cuda")).argmax(dim=1)
        miou.update(preds.cpu(), labels)
        f1.update(preds.cpu(), labels)

print(f"mIoU: {miou.compute().mean():.4f}")
print(f"F1:   {f1.compute().mean():.4f}")
```

---

## 7. Export for Deployment

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference

ONNXRuntimeInference.from_pytorch(
    model.cpu().eval(),
    "model_finetuned.onnx",
    input_shape=(1, 4, 512, 512),
)
print("Exported to ONNX")
```
