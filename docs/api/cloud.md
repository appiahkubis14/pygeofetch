# Cloud Deployment

One-command deployment of PyGeoVision models to AWS SageMaker, Azure ML, and GCP Vertex AI.

---

## AWS SageMaker

```python
from pygeovision.cloud.deploy import AWSDeployer

deployer = AWSDeployer(
    region="us-east-1",
    role_arn="arn:aws:iam::123456789012:role/SageMakerRole",  # Optional
    bucket="my-ml-bucket",
)

result = deployer.deploy(
    model_path="./model.onnx",
    endpoint_name="pygeovision-seg-prod",
    instance_type="ml.g4dn.xlarge",     # GPU inference
    initial_instance_count=1,
    framework="pytorch",
    framework_version="2.1",
    python_version="py310",
)

print(f"Endpoint URL: {result['endpoint_url']}")
print(f"Status:       {result['status']}")
```

### Instance Types

| Instance | GPU | vCPU | RAM | Use Case |
|----------|-----|------|-----|----------|
| `ml.g4dn.xlarge` | T4 16GB | 4 | 16GB | Standard inference |
| `ml.g4dn.2xlarge` | T4 16GB | 8 | 32GB | Higher throughput |
| `ml.g5.xlarge` | A10G 24GB | 4 | 16GB | Large models |
| `ml.g5.2xlarge` | A10G 24GB | 8 | 32GB | Batch inference |
| `ml.inf2.xlarge` | Inferentia2 | 4 | 16GB | Cost-efficient |
| `ml.c5.4xlarge` | — | 16 | 32GB | CPU-only |

### Batch Transform

```python
result = deployer.batch_transform(
    input_s3="s3://my-bucket/input-scenes/",
    output_s3="s3://my-bucket/predictions/",
    model_name="pygeovision-seg-v1",
    instance_type="ml.g4dn.2xlarge",
    instance_count=4,
)
```

### Auto-Scaling

```python
deployer.configure_autoscaling(
    endpoint_name="pygeovision-seg-prod",
    min_capacity=1,
    max_capacity=10,
    target_invocations_per_instance=100,
)
```

### Monitor and Delete

```python
# Check endpoint status
status = deployer.get_status("pygeovision-seg-prod")
print(f"Status: {status['EndpointStatus']}")

# Delete endpoint (stop billing)
deployer.delete("pygeovision-seg-prod")
```

---

## Azure ML

```python
from pygeovision.cloud.deploy import AzureDeployer

deployer = AzureDeployer(
    subscription_id="your-subscription-id",
    resource_group="ml-resources",
    workspace_name="pygeovision-ws",
    location="eastus",
)

result = deployer.deploy(
    model_path="./model.onnx",
    endpoint_name="pygeovision-seg",
    compute_type="Standard_NC6s_v3",   # 1× V100 16GB
    replica_count=2,
)

print(f"Endpoint: {result['endpoint_url']}")
print(f"Key:      {result['api_key']}")
```

### Azure Compute Targets

| Size | GPU | Use Case |
|------|-----|----------|
| `Standard_NC6s_v3` | 1× V100 16GB | Standard |
| `Standard_NC12s_v3` | 2× V100 32GB | High throughput |
| `Standard_ND96asr_v4` | 8× A100 80GB | Foundation models |
| `Standard_NV6` | M60 8GB | Dev/test |

---

## GCP Vertex AI

```python
from pygeovision.cloud.deploy import GCPDeployer

deployer = GCPDeployer(
    project_id="my-gcp-project",
    region="us-central1",
    service_account="ml-sa@my-gcp-project.iam.gserviceaccount.com",
)

result = deployer.deploy(
    model_path="./model.onnx",
    endpoint_name="pygeovision-seg",
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)

print(f"Endpoint ID: {result['endpoint_id']}")
print(f"URL:         {result['endpoint_url']}")
```

---

## Universal Cloud Deployer

Use `CloudDeployer` when you want provider-agnostic code:

```python
from pygeovision.cloud.deploy import CloudDeployer

deployer = CloudDeployer(provider="aws", region="us-east-1")
# or: CloudDeployer(provider="azure", ...)
# or: CloudDeployer(provider="gcp",   ...)

result = deployer.deploy("model.onnx", "my-endpoint")
print(result)
```

---

## CLI Commands

```bash
# AWS
pgv cloud deploy-aws model.onnx my-endpoint \
    --region us-east-1 \
    --instance ml.g4dn.xlarge

# GCP
pgv cloud deploy-gcp model.onnx my-endpoint \
    --project my-gcp-project \
    --region us-central1

# Azure
pgv cloud deploy-azure model.onnx my-endpoint \
    --subscription-id xxxx \
    --resource-group my-rg \
    --workspace my-ws
```

---

## Cost Estimates

Approximate costs for continuous inference endpoints (on-demand pricing):

| Provider | Instance | GPU | $/hr | $/1M chips |
|----------|----------|-----|------|-----------|
| AWS | ml.g4dn.xlarge | T4 | $0.74 | ~$1.85 |
| AWS | ml.g5.xlarge | A10G | $1.41 | ~$3.53 |
| Azure | NC6s_v3 | V100 | $3.06 | ~$7.65 |
| GCP | n1-std-8 + T4 | T4 | $0.90 | ~$2.25 |

> Tip: Use spot/preemptible instances for batch workloads (up to 90% cheaper).
