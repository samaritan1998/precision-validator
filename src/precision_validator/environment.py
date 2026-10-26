"""Non-mutating dependency inventory. This is not numerical acceptance."""
import importlib.metadata
import importlib.util
import os
import platform
import sys

from . import __version__
from .common import ASSET_ROOT


def inspect_environment():
    packages = {}
    for name in ("numpy", "torch", "transformers", "safetensors", "transformer-engine", "fla-core", "triton", "megatron-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "status": "environment_inventory",
        "toolkit_version": __version__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.system(),
        "packages": packages,
        "megatron_path": os.environ.get("MEGATRON_PATH"),
        "slime_importable": importlib.util.find_spec("slime") is not None,
        "bundled_profiles": len(list((ASSET_ROOT / "profiles").glob("*.json"))),
        "gpu_checked": False,
        "note": "Inventory only. Use doctor for the pinned Qwen runtime; run a frozen experiment for numerical acceptance.",
    }
