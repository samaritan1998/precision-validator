"""Prepare a native HF causal-LM experiment for the existing validator image.

Single process, FP32 storage, optional BF16 autocast, causal SFT batches.
No acceptance policy is inferred from the model or from observed differences.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from precision_validator.common import read, require, save
from precision_validator.protocol import prepare
from precision_validator.precision import FP32_STORAGE
from inputs import normalize_batch


def main(args):
    weights = Path(args.model).resolve(strict=True)
    source = Path(args.input_json).resolve(strict=True)
    root = Path(args.output).resolve()
    require(weights.is_dir() and source.is_file(), "Model must be a local directory and input must be a file")
    require(not root.exists(), "Use a new experiment directory")
    require(not root.is_relative_to(weights), "Output cannot be inside the model directory")
    if args.device == "cuda":
        require(args.allowed_device_name and all(n.strip() for n in args.allowed_device_name)
                and args.capability and all(n >= 0 for n in args.capability),
                "CUDA experiments require --allowed-device-name and --capability MAJOR MINOR")
    config = AutoConfig.from_pretrained(weights, local_files_only=True, trust_remote_code=False)
    require(not config.is_encoder_decoder, "This adapter requires a causal language model")
    batch = normalize_batch(read(source), config.vocab_size)
    batch_size, length = len(batch["input_ids"]), len(batch["input_ids"][0])
    max_length = getattr(config, "max_position_embeddings", None)
    require(not isinstance(max_length, int) or length <= max_length, "Input exceeds the native context limit")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", trust_remote_code=False)
        model.tie_weights()
    require(all(p.requires_grad for p in model.parameters()), "Frozen parameters require another adapter")
    parameters = {name: {"shape": list(p.shape), "group": name.rsplit(".", 1)[0]}
                  for name, p in model.named_parameters()}
    reference = {"world_size": 1, "device": args.device, "attention_implementation": args.reference_attention}
    if args.device == "cuda":
        reference.update(allowed_device_names=args.allowed_device_name, capability=args.capability)
    candidate = {**deepcopy(reference), "attention_implementation": args.candidate_attention}
    for cfg in (reference, candidate):
        if cfg["attention_implementation"] == "sdpa":
            cfg["sdpa_backend"] = "MATH"
    worker = Path(__file__).resolve().with_name("hf_causal_lm.py")
    root.mkdir(parents=True, exist_ok=False)
    save(root / "inputs/batch.json", batch)
    spec = {
        "schema_version": 1, "name": args.name or f"{config.model_type}-native-hf-{batch_size}x{length}",
        "scope": args.scope,
        "artifacts": {"code": str(worker), "preparer": str(Path(__file__).resolve()),
                      "input_schema": str(worker.with_name("inputs.py")),
                      "input": str(root / "inputs/batch.json"), "input_provenance": str(source), "weights": str(weights)},
        "parameters": parameters,
        "configurations": {"reference": reference, "candidate": candidate},
        "command": ["{python}", str(worker), "--plan", "{plan}", "--stage", "{stage}"],
        "contract": {
            "seed": args.seed, "gradient_point": "after_reduction_before_clipping",
            "update_point": "optimizer_parameter_after_minus_before", "optimizer_steps": 1,
            "optimizer_initial_state": "fresh",
            "optimizer": {"name": "AdamW", "options": {"lr": 2e-5, "betas": [0.9, 0.999],
                "eps": 1e-8, "weight_decay": 0.01, "foreach": False, "fused": False}},
            "objective": {"normalization": f"native HF causal-LM mean cross-entropy over {batch['supervised_tokens']} nonignored shifted targets",
                "samples": batch_size, "sequence_length": length, "supervised_tokens": batch["supervised_tokens"],
                "sequence_lengths": [sum(row) for row in batch["attention_mask"]],
                "input_semantics": "identical right-padded causal batches; masks index target tokens before causal shift; fresh weights and optimizer"},
            "precision_by_stage": {"reference": {**FP32_STORAGE, "compute": args.reference_compute},
                                   "candidate": {**FP32_STORAGE, "compute": args.candidate_compute}},
            "precision_comparison": {"allowed_changes": ["compute"] if args.reference_compute != args.candidate_compute else [],
                                     "initial_weights_alignment": "identical_fp32"}},
        "observables": {"loss": [], "logits_sample": [batch_size, len(batch["sample_positions"]), len(batch["sample_vocab_ids"])]},
        "required_checks": ["device_matches_configuration", "native_pretrained_load_exact", "native_attention_backend_executed",
                            "complete_real_input_contributed", "no_frozen_parameters", "native_loss_matches_masked_objective"],
        "environment": {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"},
        "policy": None,
    }
    if args.acceptance_json:
        policy_path = Path(args.acceptance_json).resolve(strict=True)
        frozen = read(policy_path)
        require(set(frozen) == {"policy", "acceptance"} and frozen["policy"] is not None and
                frozen["acceptance"] is not None, "Acceptance file must contain global policy and complete strict gates")
        spec.update(frozen)
        spec["artifacts"]["acceptance_policy"] = str(policy_path)
    save(root / "spec.json", spec)
    result = prepare(root / "spec.json", root / "results", timeout=args.timeout,
                     budget_gib=args.disk_budget_gib, retain_raw=True)
    save(root / "preparation.json", {**result, "model_class": type(model).__name__,
        "parameters": len(parameters), "trainable_elements": sum(p.numel() for p in model.parameters()),
        "sequence_length": length, "batch_size": batch_size, "supervised_tokens": batch["supervised_tokens"],
        "policy": spec["policy"], "expected_report_status": "gated" if spec["policy"] is not None else "report_only"})
    print(root / "results/plan.json", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True)
    cli.add_argument("--input-json", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    cli.add_argument("--allowed-device-name", action="append")
    cli.add_argument("--capability", nargs=2, type=int, metavar=("MAJOR", "MINOR"))
    cli.add_argument("--name")
    cli.add_argument("--reference-compute", choices=("float32", "bfloat16_autocast"), default="float32")
    cli.add_argument("--candidate-compute", choices=("float32", "bfloat16_autocast"), default="float32")
    cli.add_argument("--reference-attention", choices=("eager", "sdpa"), default="eager")
    cli.add_argument("--candidate-attention", choices=("eager", "sdpa"), default="sdpa")
    cli.add_argument("--acceptance-json", help="Frozen global policy plus complete parameter/group/observable gates; omitted means report_only")
    cli.add_argument("--scope", choices=("full_model_single_step", "integration_test"), default="full_model_single_step")
    cli.add_argument("--seed", type=int, default=271828)
    cli.add_argument("--timeout", type=int, default=3600)
    cli.add_argument("--disk-budget-gib", type=float, required=True)
    main(cli.parse_args())
