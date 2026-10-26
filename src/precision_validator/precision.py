"""Explicit stage precision contracts and native PyTorch execution evidence."""
from .common import require

FP32_STORAGE = {key: "float32" for key in
                ("parameters", "gradients", "optimizer_parameters", "optimizer_state")}


def stage_precision(plan, stage):
    contract = plan["contract"]
    if "precision_by_stage" not in contract:
        return contract["precision"]
    return contract["precision_by_stage"][{"repeat": "reference", "resume": "candidate"}.get(stage, stage)]


def validate_precision(spec):
    contract = spec["contract"]
    stages = contract.get("precision_by_stage")
    if stages is None:
        require(isinstance(contract.get("precision"), dict) and contract["precision"], "Declare precision contract")
        require("precision_comparison" not in contract, "Precision comparison requires per-stage precision")
        return
    require("precision" not in contract, "Do not mix shared and per-stage precision contracts")
    require(isinstance(stages, dict) and set(stages) == {"reference", "candidate"},
            "Declare reference/candidate precision; repeat inherits reference")
    for value in stages.values():
        require(isinstance(value, dict) and set(value) == {*FP32_STORAGE, "compute"}, "Incomplete stage precision")
        require(all(value[k] == v for k, v in FP32_STORAGE.items()),
                "Stage precision v1 requires identical FP32 storage; BF16 parameter storage needs a native master-weight adapter")
        require(value["compute"] in ("float32", "bfloat16_autocast"), "Unsupported compute precision")
    changed = sorted(k for k in stages["reference"] if stages["reference"][k] != stages["candidate"][k])
    comparison = contract.get("precision_comparison", {})
    require(comparison == {"allowed_changes": changed, "initial_weights_alignment": "identical_fp32"},
            "Freeze the exact allowed precision changes and identical_fp32 initial alignment")
    require(all(c["world_size"] == 1 for c in spec["configurations"].values()),
            "Stage precision v1 supports the single-process observer only")


def validate_precision_evidence(row, expected):
    if "compute" not in expected:
        return
    evidence = row.get("precision_evidence", {})
    require(evidence.get("kind") == "native_linear_autocast_v1" and
            evidence.get("device_type") == row["device"]["kind"], "Missing native precision execution evidence")
    require(type(evidence.get("backward_hooks")) is int and evidence["backward_hooks"] > 0 and
            evidence.get("backward_autocast_disabled") is True, "Missing precision backward evidence")
    modules = evidence.get("modules")
    require(isinstance(modules, dict) and modules, "No native Linear precision observations")
    enabled = expected["compute"] == "bfloat16_autocast"
    dtype = "torch.bfloat16" if enabled else "torch.float32"
    for value in modules.values():
        require(type(value.get("calls")) is int and value["calls"] > 0 and
                value.get("autocast_enabled") is enabled and value.get("output_dtype") == dtype,
                "Native compute precision differs from stage contract")


class NativePrecisionObserver:
    """Observe executed Linear outputs and parameter backward hooks, not config tags.

    This verifies the declared autocast path, not the dtype of every internal
    operation. Native autocast deliberately keeps some operations in FP32.
    """
    def __init__(self, model, device_type, expected):
        self.model, self.device_type, self.expected = model, device_type, expected
        self.handles = []
        self.evidence = {"kind": "native_linear_autocast_v1", "device_type": device_type,
                         "modules": {}, "backward_hooks": 0, "backward_autocast_disabled": True}

    def __enter__(self):
        import torch
        require(not torch.is_autocast_enabled(self.device_type), "Enter observer outside autocast")
        enabled = self.expected["compute"] == "bfloat16_autocast"
        expected_dtype = torch.bfloat16 if enabled else torch.float32

        def observe(name, module, args, output):
            require(torch.is_autocast_enabled(self.device_type) is enabled,
                    "Native compute precision differs: autocast was not applied as declared")
            require(isinstance(output, torch.Tensor) and output.dtype == expected_dtype,
                    "Native compute precision differs: Linear output dtype")
            require(not enabled or torch.get_autocast_dtype(self.device_type) == torch.bfloat16,
                    "Autocast dtype differs from contract")
            value = self.evidence["modules"].setdefault(name, {"calls": 0,
                "autocast_enabled": enabled, "output_dtype": str(output.dtype)})
            value["calls"] += 1

        def backward(gradient):
            require(not torch.is_autocast_enabled(self.device_type), "Run backward outside autocast")
            self.evidence["backward_hooks"] += 1
            return gradient

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.handles.append(module.register_forward_hook(
                    lambda mod, args, output, name=name: observe(name, mod, args, output)))
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                self.handles.append(parameter.register_hook(backward))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        if exc[0] is None:
            validate_precision_evidence({"device": {"kind": self.device_type},
                "precision_evidence": self.evidence}, self.expected)
