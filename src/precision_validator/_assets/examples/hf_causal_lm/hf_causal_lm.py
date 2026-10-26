"""Native Hugging Face causal-LM adapter for the existing FP32 observer.

No model-family branches, replacement attention, or optimizer formulas.
Eager versus SDPA/math is an implementation comparison, not CP validation.
"""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from precision_validator.common import load_plan, read, require, save, now
from precision_validator.sdk import capture_torch_step
from precision_validator.precision import stage_precision
from inputs import normalize_batch


def worker(plan_path, stage):
    plan = load_plan(plan_path)
    cfg = plan["configurations"][stage]
    require(cfg["world_size"] == 1, "This adapter is a single-process FP32 adapter")
    torch.set_num_threads(4)
    torch.manual_seed(plan["contract"]["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        require(torch.cuda.device_count() == 1, "Stage must own exactly one visible GPU")
        torch.cuda.set_device(0)
        require(list(torch.cuda.get_device_capability(0)) == cfg["capability"], "GPU architecture differs from configuration")
        require(any(name in torch.cuda.get_device_name(0) for name in cfg["allowed_device_names"]),
                "GPU name is outside the configured supported names")
    weights = plan["artifacts"]["weights"]["path"]
    impl = cfg["attention_implementation"]
    started = time.monotonic()
    model, loading = AutoModelForCausalLM.from_pretrained(
        weights, local_files_only=True, dtype=torch.float32,
        attn_implementation=impl, output_loading_info=True,
    )
    require(not any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            "Pretrained model did not load exactly")
    require(model.config._attn_implementation == impl, "Attention implementation changed")
    require(all(p.requires_grad for p in model.parameters()), "Unexpected frozen parameters")
    model = model.to(device).train()
    model.config.use_cache = False
    batch = read(plan["artifacts"]["input"]["path"])
    require(normalize_batch(batch, model.config.vocab_size) == batch, "Input is not canonical")
    tokens = torch.tensor(batch["input_ids"], dtype=torch.long, device=device)
    require(tokens.shape[1] == plan["contract"]["objective"]["sequence_length"], "Input length changed")
    labels = torch.tensor(batch["labels"], dtype=torch.long, device=device)
    attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long, device=device)
    compute = stage_precision(plan, stage)["compute"]
    require(compute != "bfloat16_autocast" or device.type != "cuda" or torch.cuda.is_bf16_supported(),
            "GPU does not support native BF16 autocast")
    optimizer = torch.optim.AdamW(model.parameters(), **{
        **plan["contract"]["optimizer"]["options"],
        "betas": tuple(plan["contract"]["optimizer"]["options"]["betas"]),
    })
    checks = {"device_matches_configuration": True, "native_pretrained_load_exact": True, "native_attention_backend_executed": False,
              "complete_real_input_contributed": False, "no_frozen_parameters": True,
              "native_loss_matches_masked_objective": False}
    calls = {"sdpa": 0}
    original_sdpa = F.scaled_dot_product_attention

    def counted_sdpa(*args, **kwargs):
        calls["sdpa"] += 1
        return original_sdpa(*args, **kwargs)

    def backward():
        F.scaled_dot_product_attention = counted_sdpa
        context = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if impl == "sdpa" else nullcontext()
        try:
            with context:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=compute == "bfloat16_autocast"):
                    result = model(input_ids=tokens, labels=labels, attention_mask=attention_mask, use_cache=False)
                require(result.loss.dtype == torch.float32 and torch.isfinite(result.loss).item(), "Invalid native loss")
                # Independent loss check observes detached outputs; the native loss owns backward.
                expected_loss = F.cross_entropy(result.logits.detach()[:, :-1].float().reshape(-1, model.config.vocab_size),
                                                labels[:, 1:].reshape(-1), ignore_index=-100, reduction="mean")
                checks["native_loss_matches_masked_objective"] = bool(torch.isclose(result.loss.detach(), expected_loss, rtol=1e-6, atol=1e-7).item())
                require(checks["native_loss_matches_masked_objective"], "Native loss differs from frozen masked causal objective")
                observations = {"loss": result.loss.detach(), "logits_sample":
                    result.logits.detach()[:, batch["sample_positions"]][:, :, batch["sample_vocab_ids"]].float().cpu()}
                print("NATIVE_FORWARD_COMPLETE", stage, float(result.loss.detach()), flush=True)
                result.loss.backward()
            checks["native_attention_backend_executed"] = calls["sdpa"] > 0 if impl == "sdpa" else calls["sdpa"] == 0
            require(checks["native_attention_backend_executed"], "Requested backend was not exercised")
            checks["complete_real_input_contributed"] = bool((labels[:, 1:] != -100).sum().item() == batch["supervised_tokens"])
            print("NATIVE_BACKWARD_COMPLETE", stage, "sdpa_calls", calls["sdpa"], flush=True)
            return observations
        finally:
            F.scaled_dot_product_attention = original_sdpa

    print("NATIVE_MODEL_LOADED", stage, type(model).__name__, sum(p.numel() for p in model.parameters()), flush=True)
    row = capture_torch_step(plan_path, stage, model, optimizer, backward, checks=checks)
    require(len(optimizer.state) == len(list(model.parameters())), "Missing optimizer state")
    require(all(float(s["step"]) == 1 and s["exp_avg"].dtype == s["exp_avg_sq"].dtype == torch.float32
                for s in optimizer.state.values()), "AdamW must perform exactly one FP32 update")
    if device.type == "cuda":
        torch.cuda.synchronize()
    save(Path(plan["output"]) / stage / "native-execution.json", {
        "status": "completed", "utc": now(), "model_class": type(model).__name__,
        "attention_implementation": impl, "sdpa_backend": "MATH" if impl == "sdpa" else None,
        "sdpa_calls": calls["sdpa"], "parameters": len(plan["parameters"]),
        "trainable_elements": sum(p.numel() for p in model.parameters()),
        "optimizer_steps": 1, "optimizer": "torch.optim.AdamW", "sequence_length": tokens.shape[1], "batch_size": tokens.shape[0],
        "supervised_tokens": batch["supervised_tokens"], "seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "precision": row["precision"], "precision_evidence": row.get("precision_evidence"),
        "scope": "native_hf_causal_sft_single_step_fp32_storage",
    })
    print("NATIVE_MODEL_STEP_COMPLETE", stage, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--plan", required=True)
    cli.add_argument("--stage", required=True)
    args = cli.parse_args()
    worker(args.plan, args.stage)
