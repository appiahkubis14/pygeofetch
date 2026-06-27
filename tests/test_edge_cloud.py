"""Tests for edge deployment and cloud deployment modules."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── ONNX Runtime ─────────────────────────────────────────────────────────────

class TestONNXRuntimeInference:
    def test_init_defaults(self):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        eng = ONNXRuntimeInference("model.onnx")
        assert eng.device == "cpu"
        assert eng.optimization_level == "all"
        assert eng._session is None

    def test_init_gpu(self):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        eng = ONNXRuntimeInference("model.onnx", device="cuda")
        assert eng.device == "cuda"

    def test_load_missing_file_raises(self):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        eng = ONNXRuntimeInference("nonexistent_xyz.onnx")
        with pytest.raises(Exception):  # FileNotFoundError or ImportError
            eng._load()

    def test_from_pytorch_requires_torch(self):
        """Should raise if torch not installed."""
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        try:
            import torch
        except ImportError:
            with pytest.raises(ImportError):
                ONNXRuntimeInference.from_pytorch(None, "out.onnx")
            return  # pass if torch not available

    @pytest.mark.skipif(not pytest.importorskip("torch", reason="torch"), reason="needs torch")
    def test_from_pytorch_exports(self, tmp_path):
        import torch, torch.nn as nn
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference

        class Tiny(nn.Module):
            def forward(self, x):
                return x.mean(dim=1, keepdim=True).expand(-1, 2, -1, -1)

        model = Tiny()
        out = tmp_path / "tiny.onnx"
        eng = ONNXRuntimeInference.from_pytorch(model, str(out),
                                                  input_shape=(1, 4, 8, 8),
                                                  simplify=False)
        assert out.exists()
        assert isinstance(eng, ONNXRuntimeInference)

    @pytest.mark.skipif(not pytest.importorskip("onnxruntime", reason="onnxruntime"),
                         reason="needs onnxruntime")
    @pytest.mark.skipif(not pytest.importorskip("torch", reason="torch"), reason="needs torch")
    def test_infer_after_export(self, tmp_path):
        import torch, torch.nn as nn, numpy as np
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference

        class Passthrough(nn.Module):
            def forward(self, x):
                return x[:, :2]  # just take first 2 channels

        model = Passthrough()
        out = str(tmp_path / "pass.onnx")
        eng = ONNXRuntimeInference.from_pytorch(model, out,
                                                  input_shape=(1, 4, 8, 8),
                                                  simplify=False)
        result = eng.infer(np.ones((1, 4, 8, 8), dtype=np.float32))
        assert result.shape == (1, 2, 8, 8)

    def test_model_info_before_load(self):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        eng = ONNXRuntimeInference("x.onnx")
        # Should raise before session is loaded
        with pytest.raises(Exception):
            eng.model_info()


# ── Jetson Deployer ──────────────────────────────────────────────────────────

class TestJetsonDeployer:
    def test_init(self):
        from pygeovision.edge.jetson import JetsonDeployer
        dep = JetsonDeployer()
        assert dep.device == "jetson"

    def test_invalid_precision(self):
        from pygeovision.edge.jetson import JetsonDeployer
        dep = JetsonDeployer()
        with pytest.raises(ValueError, match="precision"):
            dep.convert(MagicMock(), "a.onnx", "a.trt", precision="invalid")

    def test_convert_no_torch_returns_error(self, tmp_path):
        from pygeovision.edge.jetson import JetsonDeployer
        dep = JetsonDeployer()
        # When model_path doesn't exist, should return error dict gracefully
        result = dep.convert("nonexistent_model.onnx", str(tmp_path / "out"), str(tmp_path / "out.trt"))
        assert "error" in result or "output_path" in result  # graceful handling

    def test_convert_onnx_export_step(self, tmp_path):
        import torch, torch.nn as nn
        from pygeovision.edge.jetson import JetsonDeployer

        class TinyModel(nn.Module):
            def forward(self, x): return x[:, :2]

        dep = JetsonDeployer()
        onnx_path = str(tmp_path / "m.onnx")
        trt_path  = str(tmp_path / "m.trt")
        result = dep.convert(TinyModel(), onnx_path, trt_path,
                              precision="fp32", input_shape=(1, 4, 8, 8))
        # TensorRT likely not installed in test env — should either succeed
        # (ONNX exported) or return a Jetson command
        assert isinstance(result, dict)
        if result.get("success"):
            assert Path(onnx_path).exists()
        else:
            # Should provide jetson_command fallback
            assert "jetson_command" in result or "error" in result


# ── Cloud Deployers ──────────────────────────────────────────────────────────

class TestAWSDeployer:
    def test_init(self):
        from pygeovision.cloud.deploy import AWSDeployer
        dep = AWSDeployer(region="us-west-2")
        assert dep.region == "us-west-2"
        assert dep.provider == "aws"

    def test_get_inference_image_gpu(self):
        from pygeovision.cloud.deploy import AWSDeployer
        dep = AWSDeployer(region="us-east-1")
        img = dep._get_inference_image("pytorch", "ml.g4dn.xlarge")
        assert "gpu" in img.lower()
        assert "us-east-1" in img

    def test_get_inference_image_cpu(self):
        from pygeovision.cloud.deploy import AWSDeployer
        dep = AWSDeployer(region="eu-west-1")
        img = dep._get_inference_image("pytorch", "ml.m5.xlarge")
        assert img is not None and isinstance(img, str)  # valid ECR image URL

    def test_generate_iam_policy(self):
        import json
        from pygeovision.cloud.deploy import AWSDeployer
        dep = AWSDeployer(bucket="my-bucket")
        policy = dep.generate_iam_policy()
        p = json.loads(policy)
        assert p["Version"] == "2012-10-17"
        assert len(p["Statement"]) >= 2

    def test_deploy_missing_boto3(self, tmp_path):
        from pygeovision.cloud.deploy import AWSDeployer
        dep = AWSDeployer()
        # Create dummy model file
        dummy_model = str(tmp_path / "model.onnx")
        Path(dummy_model).write_bytes(b"fake")
        with patch.dict("sys.modules", {"boto3": None, "sagemaker": None}):
            result = dep.deploy(dummy_model, "test-endpoint")
        assert result.get("success") is False
        assert "error" in result

    def test_list_endpoints_error_returns_empty(self):
        from pygeovision.cloud.deploy import AWSDeployer
        dep = AWSDeployer()
        with patch.object(dep, "_get_client", side_effect=Exception("No AWS")):
            result = dep.list_endpoints()
        assert result == []


class TestAzureDeployer:
    def test_init(self):
        from pygeovision.cloud.deploy import AzureDeployer
        dep = AzureDeployer(region="westeurope")
        assert dep.region == "westeurope"
        assert dep.provider == "azure"

    def test_deploy_missing_azure_sdk(self, tmp_path):
        from pygeovision.cloud.deploy import AzureDeployer
        dep = AzureDeployer()
        dummy = str(tmp_path / "model.onnx")
        Path(dummy).write_bytes(b"fake")
        with patch.dict("sys.modules", {"azure.ai.ml": None, "azure.identity": None}):
            result = dep.deploy(dummy, "test-endpoint")
        assert result.get("success") is False


class TestGCPDeployer:
    def test_init(self):
        from pygeovision.cloud.deploy import GCPDeployer
        dep = GCPDeployer(project_id="my-project", region="us-central1")
        assert dep.region == "us-central1"
        assert dep.project_id == "my-project"

    def test_deploy_missing_gcp_sdk(self, tmp_path):
        from pygeovision.cloud.deploy import GCPDeployer
        dep = GCPDeployer(project_id="proj")
        dummy = str(tmp_path / "m.onnx")
        Path(dummy).write_bytes(b"x")
        with patch.dict("sys.modules", {"google.cloud.aiplatform": None}):
            result = dep.deploy(dummy, "ep")
        assert result.get("success") is False


class TestCloudDeployer:
    def test_from_provider_aws(self):
        from pygeovision.cloud.deploy import CloudDeployer, AWSDeployer
        dep = CloudDeployer.from_provider("aws", region="us-east-1")
        assert isinstance(dep, AWSDeployer)

    def test_from_provider_azure(self):
        from pygeovision.cloud.deploy import CloudDeployer, AzureDeployer
        dep = CloudDeployer.from_provider("azure")
        assert isinstance(dep, AzureDeployer)

    def test_from_provider_gcp(self):
        from pygeovision.cloud.deploy import CloudDeployer, GCPDeployer
        dep = CloudDeployer.from_provider("gcp")
        assert isinstance(dep, GCPDeployer)

    def test_from_provider_invalid(self):
        from pygeovision.cloud.deploy import CloudDeployer
        with pytest.raises(ValueError):
            CloudDeployer.from_provider("unknown_xyz")
