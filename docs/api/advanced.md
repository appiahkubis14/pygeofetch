# Advanced AI

Few-shot learning, multi-task training, AutoML, and advanced model capabilities for specialised geospatial applications.

---

## Few-Shot Learning

Classify new land-cover classes with as few as 5–20 labelled examples using prototypical networks.

```python
from pygeovision.advanced.few_shot import FewShotLearner

learner = FewShotLearner(
    backbone="dinov2-base",   # Feature extractor
    n_way=5,                  # Classes per episode
    k_shot=5,                 # Support examples per class
    device="cuda",
)

# Support set (your labelled examples)
support = {
    "solar_panel":   ["solar1.tif", "solar2.tif", "solar3.tif"],
    "rooftop":       ["roof1.tif",  "roof2.tif",  "roof3.tif"],
    "vegetation":    ["veg1.tif",   "veg2.tif",   "veg3.tif"],
    "road":          ["road1.tif",  "road2.tif",  "road3.tif"],
    "water":         ["water1.tif", "water2.tif", "water3.tif"],
}

learner.fit_support(support)

# Query (classify new images)
result = learner.predict("new_scene.tif")
print(f"Class: {result['class']}  Confidence: {result['confidence']:.3f}")

# Batch predict
results = learner.predict_batch(["a.tif", "b.tif", "c.tif"])

# Fine-tune the backbone on your episodes
history = learner.meta_train(
    dataset_path="./episodes/",
    episodes=1000,
    lr=1e-4,
)
```

---

## Multi-Task Learning

Train one shared backbone with multiple task-specific heads simultaneously.

```python
from pygeovision.advanced.multitask import MultiTaskModel

# Define tasks and heads
mt_model = MultiTaskModel(
    backbone="swin-b",
    tasks={
        "segmentation": {"num_classes": 11, "head": "upernet"},
        "detection":    {"num_classes": 5,  "head": "yolo_head"},
        "regression":   {"output_dim": 1,   "head": "linear"},  # e.g. NDVI
    },
    shared_features=True,
    device="cuda",
)

# Forward pass — returns predictions for all tasks
output = mt_model(image_tensor)
seg_pred    = output["segmentation"]   # (B, 11, H, W)
det_pred    = output["detection"]      # [(boxes, scores, classes)]
reg_pred    = output["regression"]     # (B, 1, H, W)

# Task-weighted loss
loss = mt_model.compute_loss(output, targets,
    task_weights={"segmentation": 1.0, "detection": 0.5, "regression": 0.3})
```

---

## AutoML

Automated hyperparameter optimisation using Optuna.

```python
from pygeovision.advanced.automl import AutoML

automl = AutoML(
    model_family="segformer",
    task="segmentation",
    num_classes=7,
    device="cuda",
    n_trials=50,              # Optuna trials
    timeout_hours=4.0,
    optimise="val_iou",       # Metric to maximise
    direction="maximize",
)

# Run optimisation (finds best lr, weight_decay, batch_size, model variant)
best = automl.optimise(train_dl, val_dl)

print(f"Best trial: #{best['trial_number']}")
print(f"Best val_iou: {best['val_iou']:.4f}")
print(f"Best config: {best['params']}")
# {'lr': 6.3e-5, 'weight_decay': 0.015, 'batch_size': 16, 'variant': 'b2'}

# Get the best model
best_model = automl.get_best_model()
```

### Search Space

The default search space for segmentation:

```python
search_space = {
    "learning_rate":    ("log_float", 1e-5, 1e-3),
    "weight_decay":     ("log_float", 1e-4, 0.1),
    "batch_size":       ("categorical", [8, 16, 32]),
    "model_variant":    ("categorical", ["b0", "b2", "b4", "b5"]),
    "loss_fn":          ("categorical", ["combo", "lovasz", "boundary"]),
    "warmup_epochs":    ("int", 0, 20),
}
```
