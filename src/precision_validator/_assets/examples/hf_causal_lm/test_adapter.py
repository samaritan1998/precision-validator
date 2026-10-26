"""Independent native SFT checks for FP32 backend and BF16 compute comparisons.

Small randomly initialized Gemma3/Llama models test adapter integration only.
Mixed-precision reports stay report_only without a model acceptance policy.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

# The independent baseline lives in this process, while protocol_runner sets
# this environment in its children. Configure both before importing CUDA code.
if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (None, ':4096:8'):
    raise RuntimeError('Independent HF baseline requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from transformers import AutoModelForCausalLM, Gemma3TextConfig, LlamaConfig

from precision_validator.common import read, require, save, now, digest
from precision_validator.protocol_compare import compare_suite


def strict_test_policy():
    # Integration-test bounds frozen before any training, never production defaults.
    limit = {"relative_l2": .01, "max_abs": 1e-5, "near_zero_rms": 1e-8, "near_zero_max_abs": 1e-7}
    rules = [{"pattern": "*", "gradients": limit, "updates": limit}]
    return {"policy": {f: {"repeat_relative_ceiling": 1e-6, "target_relative_ceiling": .001,
        "noise_multiplier": 2, "noise_slack": .001, "max_abs_ceiling": 1e-5} for f in ("gradients", "updates")},
        "acceptance": {"parameters": rules, "groups": rules,
            "observations": {"loss": limit, "logits_sample": limit}}}


def independent(out, results, stage, batch, device):
    """Ordinary model/optimizer execution with no adapter normalization or SDK."""
    spec = read(results.parent / "spec.json")
    cfg = spec["configurations"]["reference" if stage == "repeat" else stage]
    mode = spec["contract"]["precision_by_stage"]["reference" if stage == "repeat" else stage]["compute"]
    torch.manual_seed(spec["contract"]["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = AutoModelForCausalLM.from_pretrained(out / "weights", dtype=torch.float32,
        attn_implementation=cfg["attention_implementation"], local_files_only=True).to(device).train()
    tokens = torch.tensor(batch["input_ids"], device=device)
    attention = torch.tensor(batch["attention_mask"], device=device)
    labels = torch.tensor(batch["labels"], device=device)
    labels[torch.tensor(batch["loss_mask"], device=device) == 0] = -100
    labels[attention == 0] = -100
    labels[:, 0] = -100
    options = deepcopy(spec["contract"]["optimizer"]["options"])
    options["betas"] = tuple(options["betas"])
    optimizer = torch.optim.AdamW(model.parameters(), **options)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    from contextlib import nullcontext
    backend = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
               if cfg["attention_implementation"] == "sdpa" else nullcontext())
    with backend:
        with torch.autocast(device, dtype=torch.bfloat16, enabled=mode == "bfloat16_autocast"):
            result = model(input_ids=tokens, attention_mask=attention, labels=labels, use_cache=False)
        expected_loss = torch.nn.functional.cross_entropy(result.logits.detach()[:, :-1].float().reshape(-1, 64),
            labels[:, 1:].reshape(-1), reduction="mean", ignore_index=-100)
        torch.testing.assert_close(result.loss.detach(), expected_loss, rtol=1e-6, atol=1e-7)
        result.loss.backward()
    gradients = {n: p.grad.detach().cpu().numpy().copy().ravel() for n, p in model.named_parameters()}
    optimizer.step()
    updates = {n: (p.detach() - before[n]).cpu().numpy().ravel() for n, p in model.named_parameters()}
    receipt = read(results / stage / "rank0.json")
    checked = 0
    for piece in receipt["storage"]["pieces"]:
        expected = (gradients if piece["field"] == "gradients" else updates)[piece["name"]]
        with (results / stage / piece["file"]).open("rb") as stream:
            stream.seek(piece["offset_bytes"])
            actual = np.frombuffer(stream.read((piece["end"] - piece["start"]) * 4), dtype="<f4")
        np.testing.assert_array_equal(actual, expected[piece["start"]:piece["end"]])
        checked += len(actual)
    np.testing.assert_array_equal(receipt["observations"]["loss"]["values"], [float(result.loss.detach())])
    return checked


def main(out, device="cpu", family="gemma", allowed_names=None, capability=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(271828)
    if device == "cuda":
        require(torch.cuda.device_count() == 1, "Test owns exactly one visible GPU")
        torch.cuda.set_device(0)
        require(capability is not None and allowed_names, "Specify target hardware")
    source = Path(__file__).resolve().parent
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, attention_dropout=0.0, tie_word_embeddings=True)
    config = (Gemma3TextConfig(**common, sliding_window=16, sliding_window_pattern=2)
              if family == "gemma" else LlamaConfig(**common))
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager").float().train()
    model.save_pretrained(out / "weights")
    batch = {"input_ids": [list(range(2, 34)), list(range(12, 35)) + [0] * 9],
        "attention_mask": [[1] * 32, [1] * 23 + [0] * 9],
        "labels": [[-100] * 8 + list(range(10, 34)), [-100] * 3 + list(range(15, 35)) + [-100] * 9],
        "loss_mask": [[0] * 8 + [1] * 24, [0] * 3 + [1] * 20 + [0] * 9],
        "sample_positions": [0, 7, 12, 21], "sample_vocab_ids": [0, 2, 31, 63],
        "provenance": {"scope": "tiny_synthetic_sft_integration", "lengths": [32, 23]}}
    batch["labels"][0][11] = 55  # Custom target proves labels are not replaced by input_ids.
    batch["loss_mask"][0][17] = 0
    batch["supervised_tokens"] = 43
    save(out / "input.json", batch)
    save(out / "test-policy.json", strict_test_policy())
    cases = {}
    for name, compute, attention in (("backend", "float32", "sdpa"), ("mixed", "bfloat16_autocast", "eager")):
        experiment = out / name
        command = [sys.executable, str(source / "prepare.py"), "--model", str(out / "weights"),
            "--input-json", str(out / "input.json"), "--output", str(experiment), "--device", device,
            "--disk-budget-gib", "0.1", "--timeout", "300", "--scope", "integration_test",
            "--candidate-compute", compute, "--candidate-attention", attention]
        if name == "backend":
            command.extend(["--acceptance-json", str(out / "test-policy.json")])
        if device == "cuda":
            for value in allowed_names:
                command.extend(["--allowed-device-name", value])
            command.extend(["--capability", *map(str, capability)])
        subprocess.run(command, check=True)
        results = experiment / "results"
        for stage in ("reference", "repeat", "candidate"):
            subprocess.run([sys.executable, "-m", "precision_validator", "run", "--plan", str(results / "plan.json"), "--stage", stage], check=True)
        report = compare_suite(results / "plan.json", retain_raw=True)
        expected_status = "passed" if name == "backend" else "report_only"
        require(report["status"] == expected_status, f"Unexpected {name} result: {report['status']}")
        checked = {stage: independent(out, results, stage, batch, device) for stage in ("reference", "candidate")}
        require(all(n == 2 * report["trainable_elements"] for n in checked.values()), "Independent baseline missed parameters")
        for field in ("gradients", "updates"):
            require(report["comparisons"]["repeat"][field]["max_abs"] == 0, "Repeat changed")
        for stage in ("reference", "repeat", "candidate"):
            row = read(results / stage / "native-execution.json")
            expected_calls = 2 if stage == "candidate" and attention == "sdpa" else 0
            require(row["sdpa_calls"] == expected_calls, "Requested backend not executed")
            require(row["batch_size"] == 2 and row["supervised_tokens"] == 43, "SFT objective changed")
        cases[name] = {"status": report["status"], "independent_native_gradient_and_delta_exact": True,
            "checked_elements": checked, "acceptance_sha256": digest(results / "acceptance.json"),
            "comparisons": {k: {f: {m: v[f][m] for m in ("max_abs", "relative_l2")}
                for f in ("gradients", "updates", "observation/loss", "observation/logits_sample")}
                for k, v in report["comparisons"].items()}}
        rerun = subprocess.run(command, capture_output=True, text=True)
        require(rerun.returncode != 0 and "Use a new experiment directory" in rerun.stderr, "Immutable output reused")
    save(out / "receipt.json", {"status": "passed", "utc": now(), "device": device, "family": family,
        "scope": "tiny_native_sft_adapter_integration_only", "cases": cases,
        "batch_size": 2, "sequence_lengths": [32, 23], "supervised_tokens": 43,
        "mixed_precision_numeric_acceptance": "report_only_no_model_policy", "existing_output_rejected": True})
    print("PACKAGED_HF_SFT_ADAPTER_TEST_PASSED", family, device, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--family", choices=("gemma", "llama"), default="gemma")
    cli.add_argument("--allowed-device-name", action="append")
    cli.add_argument("--capability", nargs=2, type=int)
    args = cli.parse_args()
    main(args.output, args.device, args.family, args.allowed_device_name, args.capability)
