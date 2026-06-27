# Monitoring

Production monitoring for deployed geospatial AI models — detect data distribution drift before it silently degrades prediction quality.

---

## Drift Detector

`DriftDetector` computes feature-space statistics on a reference dataset and raises an alert when new images deviate significantly.

```python
from pygeovision.monitoring.drift import DriftDetector

detector = DriftDetector(
    method="psi",          # "psi" (Population Stability Index) | "kl" (KL divergence) | "both"
    n_bins=10,             # Histogram bins for distribution approximation
    threshold_warn=0.1,    # PSI threshold for WARNING
    threshold_critical=0.2 # PSI threshold for CRITICAL alert
)

# 1. Fit on reference data (e.g. training set scenes)
reference_images = ["ref_jan.tif", "ref_feb.tif", "ref_mar.tif", ...]
detector.fit(reference_images)

# 2. Check new data
new_images  = ["new_oct.tif", "new_nov.tif", ...]
report      = detector.check(new_images)

# 3. Interpret results
level = report['data_drift']['drift_level']   # "none" | "warning" | "critical"
psi   = report['data_drift']['psi_score']

print(f"Drift level: {level}  (PSI={psi:.3f})")
print(f"Drifted bands: {report['data_drift']['drifted_bands']}")

# Per-band PSI scores
for band, psi in report['data_drift']['per_band_psi'].items():
    flag = "⚠" if psi > 0.1 else "✓"
    print(f"  {flag} Band {band}: PSI={psi:.3f}")
```

### PSI Interpretation

| PSI Value | Drift Level | Action |
|-----------|-------------|--------|
| < 0.1 | None | No action needed |
| 0.1 – 0.2 | Warning | Monitor closely; consider retraining |
| > 0.2 | Critical | Retrain or recalibrate model immediately |

---

## Performance Tracker

`ModelPerformanceTracker` logs prediction quality metrics over time and detects performance degradation.

```python
from pygeovision.monitoring.tracker import ModelPerformanceTracker

tracker = ModelPerformanceTracker(
    metrics=["val_iou", "val_f1", "val_precision", "val_recall"],
    window=10,              # Rolling window for trend computation
    log_dir="./monitoring/logs/",
)

# Log metrics after each evaluation
tracker.log(epoch=5, metrics={
    "val_iou":       0.843,
    "val_f1":        0.887,
    "val_precision": 0.902,
    "val_recall":    0.873,
})

# Trend analysis
trend = tracker.trend("val_iou")
print(f"Trend direction: {trend['direction']}")   # "improving" | "degrading" | "stable"
print(f"Slope:           {trend['slope']:.5f}")
print(f"Confidence:      {trend['r_squared']:.3f}")

# Rolling statistics
stats = tracker.statistics("val_iou")
print(f"Mean (last 10): {stats['mean']:.4f}")
print(f"Std  (last 10): {stats['std']:.4f}")
print(f"Min:            {stats['min']:.4f}")

# Export log to CSV
tracker.export_csv("./monitoring/metrics_log.csv")
```

---

## Alert System

Configure threshold-based alerts delivered via webhook, email, or Slack.

```python
from pygeovision.monitoring.alerts import AlertManager

manager = AlertManager(
    channels={
        "slack":   {"webhook_url": "https://hooks.slack.com/services/..."},
        "email":   {"smtp_host": "smtp.gmail.com", "to": "team@example.com"},
        "webhook": {"url": "https://myapp.com/alerts"},
    },
    cooldown_minutes=60,   # Minimum time between repeated alerts
)

# Define alert rules
manager.add_rule(
    name="iou_drop",
    metric="val_iou",
    condition="less_than",
    threshold=0.75,
    severity="critical",
    message="Model mIoU dropped below 0.75 — retraining required",
)

manager.add_rule(
    name="drift_warning",
    metric="psi_score",
    condition="greater_than",
    threshold=0.1,
    severity="warning",
    message="Data distribution drift detected",
)

# Check and fire alerts
fired = manager.check({
    "val_iou":   0.72,
    "psi_score": 0.14,
})

for alert in fired:
    print(f"[{alert['severity'].upper()}] {alert['message']}")
```

---

## Full Monitoring Pipeline

Combine drift detection, performance tracking, and alerting in one workflow:

```python
from pygeovision.monitoring.drift   import DriftDetector
from pygeovision.monitoring.tracker import ModelPerformanceTracker
from pygeovision.monitoring.alerts  import AlertManager

class ProductionMonitor:
    def __init__(self, model, reference_images):
        self.drift   = DriftDetector().fit(reference_images)
        self.tracker = ModelPerformanceTracker(["val_iou"])
        self.alerts  = AlertManager(channels={"slack": {...}})

    def check(self, new_images, val_metrics, epoch):
        drift_report = self.drift.check(new_images)
        self.tracker.log(epoch, val_metrics)

        combined = {**val_metrics, "psi_score": drift_report['data_drift']['psi_score']}
        fired    = self.alerts.check(combined)

        return {"drift": drift_report, "performance": val_metrics, "alerts": fired}
```
