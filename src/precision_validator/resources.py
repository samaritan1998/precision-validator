"""Bundled adapter templates, independent of the working directory or image."""
from pathlib import Path
import shutil

from . import __version__
from .common import ASSET_ROOT, digest, require, save


EXAMPLES = {
    "torch": {
        "files": ["torch_models.py"],
        "requires": ["torch"],
        "scope": "Tiny CPU MLP/attention, one-step integration check",
    },
    "torch-trajectory": {
        "files": ["training_trajectory.py"],
        "requires": ["torch"],
        "scope": "Three steps, varying inputs, Adam state and process-restart check",
    },
    "hf-causal-lm": {
        "files": ["hf_causal_lm/prepare.py", "hf_causal_lm/hf_causal_lm.py", "hf_causal_lm/inputs.py"],
        "requires": ["torch", "transformers", "safetensors"],
        "scope": "Single-process causal LM/SFT, FP32 storage, optional BF16 autocast",
    },
    "native-parallel": {
        "files": ["native_parallel.py"],
        "requires": ["torch", "CUDA"],
        "scope": "Small distributed PyTorch fixtures; not a production model adapter",
    },
    "megatron-parallel": {
        "files": ["megatron_parallel.py"],
        "requires": ["torch", "CUDA", "Megatron", "Transformer Engine"],
        "scope": "Native GPT/MoE TP/PP/CP/EP/ETP evidence mapping",
    },
    "slime-parallel": {
        "files": ["megatron_parallel.py", "slime_parallel.py"],
        "requires": ["torch", "CUDA", "Megatron training", "Slime"],
        "scope": "Native Slime single-step integration fixture",
    },
    "slime-trajectory": {
        "files": ["megatron_parallel.py", "slime_parallel.py", "slime_trajectory.py"],
        "requires": ["torch", "CUDA", "Megatron training", "Slime"],
        "scope": "Native multi-step/resume, topology and FP32/BF16 storage fixture",
    },
}


def list_examples():
    return [{"adapter": name, **value} for name, value in EXAMPLES.items()]


def export_example(adapter, output):
    """Copy a template and its helpers to a new user-owned directory."""
    require(adapter in EXAMPLES, f"Unknown example adapter: {adapter}")
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), "Example output already exists; choose a new directory")
    source = ASSET_ROOT / "examples"
    files = EXAMPLES[adapter]["files"]
    require(all((source / file).is_file() for file in files), "Installed adapter resources are incomplete")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for file in files:
        target = output / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / file, target)
        hashes[file] = digest(target)
    manifest = {"toolkit_version": __version__, "adapter": adapter, "files": hashes,
                "requires": EXAMPLES[adapter]["requires"], "scope": EXAMPLES[adapter]["scope"]}
    save(output / "template-manifest.json", manifest)
    return {"status": "example_created", "output": str(output), **manifest}
