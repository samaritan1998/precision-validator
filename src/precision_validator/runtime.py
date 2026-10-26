"""Preflight and immutable runtime fingerprint (never disables driver checks)."""
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
from .common import ROOT, ASSET_ROOT, digest, read, require, object_hash


def runtime_fingerprint(megatron_path):
    import torch
    versions = {name: importlib.metadata.version(name) for name in ("transformer-engine", "fla-core", "triton")}
    versions.update(torch=str(torch.__version__), cuda=torch.version.cuda)
    root = Path(megatron_path)
    require((root / "megatron/core/optimizer").is_dir(), "Megatron checkout missing")
    files = {str(p.relative_to(root)): digest(p) for p in sorted((root / "megatron").rglob("*.py"))}
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    return {"versions": versions, "python": platform.python_version(), "packages_sha256": object_hash(packages),
            "megatron_sha256": object_hash(files), "base_image": os.environ.get("PRECISION_RUNTIME_IMAGE")}


def doctor(profile=None, megatron_path=None, cpu_only=False):
    megatron_path = megatron_path or os.environ.get("MEGATRON_PATH", "/root/Megatron-LM")
    result = runtime_fingerprint(megatron_path)
    require(result["versions"] == read(ASSET_ROOT / "runtime-lock.json")["versions"],
            f"Runtime version drift: {result['versions']}")
    built = ASSET_ROOT / "image-runtime.json"
    if built.is_file():
        require(result == read(built), "Runtime or Megatron differs from image build fingerprint")
    require(os.environ.get("NVIDIA_DISABLE_REQUIRE", "").lower() not in ("true", "1"),
            "NVIDIA_REQUIRE checks must not be disabled")
    if cpu_only:
        return {"status": "runtime_checked", "gpu_checked": False, "runtime": result}
    import torch
    require(torch.cuda.is_available(), "CUDA unavailable; this is an environment failure, not a numerical failure")
    devices = []
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            value = (torch.arange(1024, device="cuda", dtype=torch.float32) + 1).sum().item()
            require(value == 524800, "CUDA arithmetic smoke check failed")
        devices.append({"index": i, "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i))})
    if profile:
        require(len(devices) == profile["world_size"], "Exactly eight visible GPUs are required")
        for row in devices:
            matches = profile["device"] in row["name"] or (profile["device"] == "B200" and row["name"] == "CF-NG-BZZ2-O")
            require(matches and row["capability"] == profile["capability"], f"Wrong GPU: {row}")
    numeric_keys = {"CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                    "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_MODULE_LOADING", "CUDA_MODULE_DATA_LOADING",
                    "TORCHINDUCTOR_COMPILE_THREADS", "PYTORCH_CUDA_ALLOC_CONF"}
    numeric_keys.update(profile["environment"] if profile else ())
    numerical_environment = {k: os.environ.get(k) for k in sorted(numeric_keys)}
    if profile:
        require(all(numerical_environment[k] == v for k, v in profile["environment"].items()),
                "Training worker environment differs from profile")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=30, check=True)
    return {"status": "environment_passed", "gpu_checked": True, "runtime": result,
            "devices": devices, "driver": smi.stdout.splitlines(), "numerical_environment": numerical_environment}
