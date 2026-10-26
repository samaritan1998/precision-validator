"""Small, dependency-free provenance and file utilities."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
ASSET_ROOT = Path(__file__).resolve().parent / "_assets"
STAGES = ("cp1", "cp1_repeat", "cp_target")


class ValidationError(RuntimeError):
    """Evidence, configuration or numerical validation failed."""


def require(condition, message):
    # Unlike assert, these checks survive python -O.
    if not condition:
        raise ValidationError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def source_inventory(source):
    source = Path(source).resolve()
    require((source / "slime/backends/megatron_utils/model.py").is_file(),
            "--source must contain slime/ and slime_plugins/")
    require(not git(source, "status", "--porcelain", "--", "."),
            "Candidate source must be a clean checkout; commit changes before preparing a plan")
    names = git(source, "ls-files", "-z", "--", ".").split("\0")
    files = {name: digest(source / name) for name in names if name and (source / name).is_file()}
    require(files, "No tracked source files")
    return {"commit": git(source, "rev-parse", "HEAD"), "files": files,
            "tree_sha256": object_hash(files)}


def model_inventory(model):
    model = Path(model)
    require((model / "config.json").is_file(), "Model config.json is missing")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        names = {"model.safetensors.index.json", *read(index)["weight_map"].values()}
    else:
        names = {"model.safetensors"}
    names.add("config.json")
    names.update(path.name for path in model.glob("*token*.json"))
    files = {}
    for name in sorted(names):
        require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe model shard path")
        path = model / name
        require(path.is_file(), f"Missing model file: {name}")
        files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return {"files": files, "sha256": object_hash(files)}


def verify_files(root, inventory):
    for name, record in inventory.items():
        path = Path(root) / name
        expected = record if isinstance(record, str) else record["sha256"]
        require(path.is_file() and digest(path) == expected, f"File changed or missing: {path}")


def load_plan(path):
    plan = read(path)
    expected = plan.pop("plan_sha256", None)
    require(expected == object_hash(plan), "Plan checksum mismatch")
    plan["plan_sha256"] = expected
    from .adapters import adapter_for
    adapter_for(plan)
    return plan


def validate_artifact_path(root, relative):
    path = Path(root) / relative
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            f"Unsafe artifact path: {relative}")
    require(not path.is_symlink() and path.resolve().is_relative_to(Path(root).resolve()),
            f"Symlink or escaping artifact: {relative}")
    return path
