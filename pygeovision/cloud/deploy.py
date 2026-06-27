"""
Cloud deployment utilities (F4) — AWS SageMaker, Azure ML, GCP Vertex AI.
Deploy PyGeoVision models to all major cloud providers.
"""
from __future__ import annotations
import json, logging, os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class CloudDeployer:
    """Base class for cloud deployment."""

    def __init__(self, provider: str, region: Optional[str] = None) -> None:
        self.provider = provider
        self.region = region

    def deploy(self, model_path: str, endpoint_name: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, endpoint_name: str, input_data: Any) -> Any:
        raise NotImplementedError

    def delete_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        raise NotImplementedError

    def list_endpoints(self) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def from_provider(provider: str, **kwargs) -> "CloudDeployer":
        providers = {"aws": AWSDeployer, "azure": AzureDeployer, "gcp": GCPDeployer}
        if provider not in providers:
            raise ValueError(f"provider must be one of {list(providers)}")
        return providers[provider](**kwargs)


class AWSDeployer(CloudDeployer):
    """Deploy to AWS SageMaker (F4).

    Example::

        deployer = AWSDeployer(region="us-east-1")
        result = deployer.deploy(
            model_path="./models/buildings.onnx",
            endpoint_name="pgv-buildings-v1",
            instance_type="ml.g4dn.xlarge",
        )
        prediction = deployer.predict("pgv-buildings-v1", image_array)
    """

    def __init__(self, region: str = "us-east-1",
                 role_arn: Optional[str] = None,
                 bucket: Optional[str] = None) -> None:
        super().__init__("aws", region)
        self.role_arn = role_arn or os.environ.get("AWS_SAGEMAKER_ROLE_ARN", "")
        self.bucket   = bucket   or os.environ.get("AWS_S3_BUCKET", "")
        self._client = None

    def _get_client(self):
        if self._client: return self._client
        try:
            import boto3
            self._client = boto3.client("sagemaker", region_name=self.region)
            return self._client
        except ImportError:
            raise ImportError("pip install boto3")

    def deploy(
        self,
        model_path: Union[str, Path],
        endpoint_name: str,
        instance_type: str = "ml.g4dn.xlarge",
        framework: str = "onnx",
        min_instances: int = 1,
        max_instances: int = 3,
        auto_scale: bool = True,
    ) -> Dict[str, Any]:
        """Package and deploy a model to AWS SageMaker."""
        try:
            import boto3, sagemaker
            from sagemaker.model import Model

            # Upload model to S3
            s3 = boto3.client("s3", region_name=self.region)
            model_key = f"pygeovision/models/{Path(model_path).name}"
            s3.upload_file(str(model_path), self.bucket, model_key)
            model_s3_uri = f"s3://{self.bucket}/{model_key}"
            logger.info("Model uploaded: %s", model_s3_uri)

            # Create SageMaker model
            sm = sagemaker.Session()
            model = Model(
                model_data=model_s3_uri,
                role=self.role_arn,
                image_uri=self._get_inference_image(framework, instance_type),
                sagemaker_session=sm,
            )

            # Deploy endpoint
            predictor = model.deploy(
                initial_instance_count=min_instances,
                instance_type=instance_type,
                endpoint_name=endpoint_name,
            )

            result = {
                "success": True,
                "endpoint_name": endpoint_name,
                "provider": "aws_sagemaker",
                "instance_type": instance_type,
                "model_s3_uri": model_s3_uri,
                "endpoint_url": f"https://runtime.sagemaker.{self.region}.amazonaws.com/endpoints/{endpoint_name}/invocations",
            }

            if auto_scale and max_instances > min_instances:
                self._setup_autoscaling(endpoint_name, min_instances, max_instances)
                result["autoscaling"] = {"min": min_instances, "max": max_instances}

            return result

        except ImportError:
            return {"success": False, "error": "pip install boto3 sagemaker"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _get_inference_image(self, framework: str, instance_type: str) -> str:
        """Get the appropriate SageMaker inference container URI."""
        gpu = "gpu" if "g" in instance_type or "p" in instance_type else "cpu"
        images = {
            "onnx": f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-inference:2.1-{gpu}-py310",
            "pytorch": f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-inference:2.1-{gpu}-py310",
            "tensorflow": f"763104351884.dkr.ecr.{self.region}.amazonaws.com/tensorflow-inference:2.13-{gpu}",
        }
        return images.get(framework, images["pytorch"])

    def _setup_autoscaling(self, endpoint_name: str, min_cap: int, max_cap: int) -> None:
        try:
            import boto3
            asg = boto3.client("application-autoscaling", region_name=self.region)
            resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
            asg.register_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                MinCapacity=min_cap, MaxCapacity=max_cap,
            )
            asg.put_scaling_policy(
                PolicyName=f"{endpoint_name}-autoscale",
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                PolicyType="TargetTrackingScaling",
                TargetTrackingScalingPolicyConfiguration={
                    "TargetValue": 70.0,
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                    },
                    "ScaleOutCooldown": 60, "ScaleInCooldown": 300,
                },
            )
            logger.info("Auto-scaling configured: %d-%d instances", min_cap, max_cap)
        except Exception as exc:
            logger.warning("Auto-scaling setup failed: %s", exc)

    def predict(self, endpoint_name: str, input_data: Any) -> Any:
        try:
            import boto3, numpy as np, json as _json
            runtime = boto3.client("sagemaker-runtime", region_name=self.region)
            if isinstance(input_data, np.ndarray):
                payload = _json.dumps({"inputs": input_data.tolist()})
            else:
                payload = _json.dumps(input_data)
            resp = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload,
            )
            return _json.loads(resp["Body"].read())
        except ImportError:
            return {"error": "pip install boto3"}

    def delete_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        try:
            import boto3
            sm = boto3.client("sagemaker", region_name=self.region)
            sm.delete_endpoint(EndpointName=endpoint_name)
            return {"success": True, "deleted": endpoint_name}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_endpoints(self) -> List[str]:
        try:
            resp = self._get_client().list_endpoints(StatusEquals="InService")
            return [e["EndpointName"] for e in resp.get("Endpoints", [])]
        except Exception:
            return []

    def generate_iam_policy(self) -> str:
        """Generate a minimal IAM policy JSON for PyGeoVision deployment."""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Effect": "Allow", "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
                 "Resource": [f"arn:aws:s3:::{self.bucket}/*", f"arn:aws:s3:::{self.bucket}"]},
                {"Effect": "Allow", "Action": ["sagemaker:CreateModel", "sagemaker:CreateEndpoint",
                  "sagemaker:CreateEndpointConfig", "sagemaker:DeleteEndpoint",
                  "sagemaker:InvokeEndpoint", "sagemaker:ListEndpoints"],
                 "Resource": "*"},
            ]
        }
        return json.dumps(policy, indent=2)


class AzureDeployer(CloudDeployer):
    """Deploy to Azure Machine Learning (F4).

    Example::

        deployer = AzureDeployer(
            subscription_id="...", resource_group="...", workspace_name="pgv-workspace"
        )
        result = deployer.deploy("model.onnx", "pgv-buildings",
                                  vm_size="Standard_NC4as_T4_v3")
    """

    def __init__(
        self,
        subscription_id: Optional[str] = None,
        resource_group: Optional[str] = None,
        workspace_name: Optional[str] = None,
        region: str = "eastus",
    ) -> None:
        super().__init__("azure", region)
        self.subscription_id = subscription_id or os.environ.get("AZURE_SUBSCRIPTION_ID", "")
        self.resource_group  = resource_group  or os.environ.get("AZURE_RESOURCE_GROUP", "")
        self.workspace_name  = workspace_name  or os.environ.get("AZURE_ML_WORKSPACE", "")

    def deploy(
        self,
        model_path: Union[str, Path],
        endpoint_name: str,
        vm_size: str = "Standard_NC4as_T4_v3",
        instance_count: int = 1,
    ) -> Dict[str, Any]:
        """Deploy model to Azure ML online endpoint."""
        try:
            from azure.ai.ml import MLClient
            from azure.ai.ml.entities import (ManagedOnlineEndpoint, ManagedOnlineDeployment,
                                               Model as AzModel, Environment, CodeConfiguration)
            from azure.identity import DefaultAzureCredential

            ml = MLClient(DefaultAzureCredential(),
                           self.subscription_id, self.resource_group, self.workspace_name)

            # Register model
            model = ml.models.create_or_update(
                AzModel(path=str(model_path), name=endpoint_name,
                         type="custom_model", description="PyGeoVision model")
            )
            logger.info("Azure model registered: %s v%s", model.name, model.version)

            # Create endpoint
            endpoint = ml.online_endpoints.begin_create_or_update(
                ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")
            ).result()

            # Create deployment
            deployment = ml.online_deployments.begin_create_or_update(
                ManagedOnlineDeployment(
                    name="default",
                    endpoint_name=endpoint_name,
                    model=model,
                    instance_type=vm_size,
                    instance_count=instance_count,
                )
            ).result()

            return {
                "success": True,
                "endpoint_name": endpoint_name,
                "provider": "azure_ml",
                "scoring_uri": endpoint.scoring_uri,
                "vm_size": vm_size,
            }
        except ImportError:
            return {"success": False, "error": "pip install azure-ai-ml azure-identity"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def predict(self, endpoint_name: str, input_data: Any) -> Any:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            import json as _json, numpy as np
            ml = MLClient(DefaultAzureCredential(), self.subscription_id,
                           self.resource_group, self.workspace_name)
            payload = {"inputs": input_data.tolist() if hasattr(input_data, "tolist") else input_data}
            result = ml.online_endpoints.invoke(endpoint_name, request_file=_json.dumps(payload))
            return result
        except ImportError:
            return {"error": "pip install azure-ai-ml"}

    def delete_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            ml = MLClient(DefaultAzureCredential(), self.subscription_id,
                           self.resource_group, self.workspace_name)
            ml.online_endpoints.begin_delete(name=endpoint_name).result()
            return {"success": True, "deleted": endpoint_name}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_endpoints(self) -> List[str]:
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            ml = MLClient(DefaultAzureCredential(), self.subscription_id,
                           self.resource_group, self.workspace_name)
            return [e.name for e in ml.online_endpoints.list()]
        except Exception:
            return []


class GCPDeployer(CloudDeployer):
    """Deploy to Google Cloud Vertex AI (F4).

    Example::

        deployer = GCPDeployer(project_id="my-project", region="us-central1")
        result = deployer.deploy("model.onnx", "pgv-buildings",
                                  machine_type="n1-standard-4-nvidia-tesla-t4")
    """

    def __init__(self, project_id: Optional[str] = None, region: str = "us-central1") -> None:
        super().__init__("gcp", region)
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID", "")

    def deploy(
        self,
        model_path: Union[str, Path],
        endpoint_name: str,
        machine_type: str = "n1-standard-4",
        accelerator_type: str = "NVIDIA_TESLA_T4",
        accelerator_count: int = 1,
        gcs_bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deploy model to GCP Vertex AI."""
        try:
            from google.cloud import aiplatform
            from google.cloud import storage

            aiplatform.init(project=self.project_id, location=self.region)

            # Upload to GCS
            bucket = gcs_bucket or os.environ.get("GCS_BUCKET", f"{self.project_id}-pgv")
            gcs_client = storage.Client(project=self.project_id)
            gcs_bucket_obj = gcs_client.bucket(bucket)
            blob = gcs_bucket_obj.blob(f"pygeovision/models/{Path(model_path).name}")
            blob.upload_from_filename(str(model_path))
            model_gcs_uri = f"gs://{bucket}/pygeovision/models/"
            logger.info("Model uploaded: %s", model_gcs_uri)

            # Upload model to Vertex AI
            model = aiplatform.Model.upload(
                display_name=endpoint_name,
                artifact_uri=model_gcs_uri,
                serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.2-1:latest",
            )

            # Create endpoint and deploy
            endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
            model.deploy(
                endpoint=endpoint,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                min_replica_count=1,
                max_replica_count=5,
            )

            return {
                "success": True,
                "endpoint_name": endpoint_name,
                "provider": "gcp_vertex",
                "endpoint_id": endpoint.name,
                "model_gcs_uri": model_gcs_uri,
            }
        except ImportError:
            return {"success": False, "error": "pip install google-cloud-aiplatform google-cloud-storage"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def predict(self, endpoint_name: str, input_data: Any) -> Any:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
            endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
            if not endpoints:
                return {"error": f"Endpoint {endpoint_name!r} not found"}
            endpoint = endpoints[0]
            payload = input_data.tolist() if hasattr(input_data, "tolist") else input_data
            result = endpoint.predict(instances=[payload])
            return result.predictions
        except ImportError:
            return {"error": "pip install google-cloud-aiplatform"}

    def delete_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
            endpoints = aiplatform.Endpoint.list(filter=f"display_name={endpoint_name}")
            for ep in endpoints:
                ep.undeploy_all()
                ep.delete()
            return {"success": True, "deleted": endpoint_name}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def list_endpoints(self) -> List[str]:
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=self.project_id, location=self.region)
            return [e.display_name for e in aiplatform.Endpoint.list()]
        except Exception:
            return []
