"""Small masked token-training example with three inputs, dropout and process restart.

This exercises generic callbacks, not the HF or distributed Slime adapter.
"""
import argparse
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch import nn

from precision_validator.common import read, save, load_plan, require
from precision_validator.trajectory import PRECISION, STATE_FIELDS
from precision_validator.trajectory_sdk import capture_torch_trajectory


class TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 8)
        self.projection = nn.Linear(8, 8)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(8, 17)

    def forward(self, tokens, split):
        hidden = self.dropout(self.projection(self.embedding(tokens)).tanh())
        flattened = hidden.flatten(0, 1)
        logits = torch.cat([self.head(x) for x in flattened.chunk(split)], dim=0)
        return logits.reshape(*tokens.shape, 17)


def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make(folder, device="cpu"):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    seed_all(9001)
    model = TokenModel()
    steps, artifacts = [], {"code": str(Path(__file__).resolve())}
    for i, (batch, length) in enumerate(((1, 5), (2, 7), (3, 4)), 1):
        tokens = torch.randint(0, 17, (batch, length))
        labels = torch.randint(0, 17, (batch, length))
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[:, 0] = False  # Prompt tokens are excluded from supervision.
        if batch > 1:
            mask[-1, -2:] = False  # Different valid token counts / padding.
        labels[~mask] = -100
        objective = {"normalization": "mean_cross_entropy_over_unmasked_labels",
                     "global_samples": batch, "supervised_units": int(mask.sum())}
        file = folder / f"input-{i}.json"
        save(file, {"token_ids": tokens.tolist(), "labels": labels.tolist(), "loss_mask": mask.tolist()})
        artifacts[f"input_{i}"] = str(file)
        steps.append({"input_artifact": f"input_{i}", "objective": objective, "observables": {"loss": [], "logits": [batch, length, 17]}})
    artifacts["input"] = artifacts["input_1"]
    # These are example-specific CPU/GPU smoke bounds, not production model policy.
    limit = {"relative_l2": 0.01, "max_abs": 2e-5, "near_zero_rms": 1e-7, "near_zero_max_abs": 1e-7}
    local = [{"pattern": "*", "gradients": limit, "updates": limit}]
    state = [{"pattern": "*", **{f: limit for f in STATE_FIELDS}}]
    spec = {"schema_version": 1, "scope": "integration_test", "name": "masked-token-training-multistep-resume",
        "artifacts": artifacts, "parameters": {name: {"shape": list(p.shape), "group": name.rsplit(".", 1)[0]} for name, p in model.named_parameters()},
        "configurations": {"reference": {"world_size": 1, "device": device, "head_splits": 1},
                           "candidate": {"world_size": 1, "device": device, "head_splits": 2}},
        "command": ["{python}", str(Path(__file__).resolve()), "worker", "--plan", "{plan}", "--stage", "{stage}"],
        "contract": {"seed": 9001, "gradient_point": "after_reduction_before_clipping", "update_point": "optimizer_parameter_after_minus_before",
            "optimizer_steps": 3, "optimizer_initial_state": "fresh", "precision": PRECISION,
            "objective": {"normalization": "see_frozen_per_step_objectives"},
            "optimizer": {"name": "AdamW", "options": {"lr": 0.01, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01,
                "foreach": False, "fused": False, "amsgrad": False}}},
        "observables": steps[0]["observables"],
        "required_checks": ["mask_and_sample_count_verified"],
        "environment": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "policy": {f: {"repeat_relative_ceiling": 1e-7, "target_relative_ceiling": 0.01, "noise_multiplier": 2, "noise_slack": 0.01,
                       "max_abs_ceiling": 2e-5} for f in ("gradients", "updates")},
        "acceptance": {"parameters": local, "groups": local, "observations": {"loss": limit, "logits": limit}},
        "trajectory": {"steps": steps, "resume_after_step": 2, "scheduler": {"name": "StepLR", "options": {"step_size": 1, "gamma": 0.5}},
                       "state_acceptance": {"parameters": state, "groups": state}}}
    save(folder / "spec.json", spec)
    return folder / "spec.json"


def worker(plan_path, stage, continuation=False):
    plan = load_plan(plan_path)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_all(plan["contract"]["seed"])
    cfg = plan["configurations"][stage]
    model = TokenModel().to(cfg["device"])
    options = dict(plan["contract"]["optimizer"]["options"])
    options["betas"] = tuple(options["betas"])
    optimizer = torch.optim.AdamW(model.parameters(), **options)
    sched = plan["trajectory"]["scheduler"]
    scheduler = getattr(torch.optim.lr_scheduler, sched["name"])(optimizer, **sched["options"])
    def backward(step, frozen):
        data = read(plan["artifacts"][frozen["input_artifact"]]["path"])
        tokens = torch.tensor(data["token_ids"], device=cfg["device"])
        labels = torch.tensor(data["labels"], device=cfg["device"])
        mask = torch.tensor(data["loss_mask"], dtype=torch.bool, device=cfg["device"])
        require(torch.equal(mask, labels != -100), "Input mask/labels differ")
        objective = {"normalization": "mean_cross_entropy_over_unmasked_labels", "global_samples": len(tokens), "supervised_units": int(mask.sum())}
        require(objective == frozen["objective"], "Observed sample/token count differs")
        # Exercise all three saved RNG streams, including native dropout's torch RNG.
        gain = 0.99 + 0.01 * (random.random() + float(np.random.random()))
        logits = model(tokens, cfg["head_splits"]) * gain
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        loss.backward()
        return {"observations": {"loss": loss, "logits": logits}, "objective": objective,
                "input_sha256": plan["artifacts"][frozen["input_artifact"]]["sha256"], "checks": {"mask_and_sample_count_verified": True}}
    return capture_torch_trajectory(plan_path, stage, model, optimizer, scheduler, backward, resume=continuation)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    create = sub.add_parser("make")
    create.add_argument("--output", required=True)
    create.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run = sub.add_parser("worker")
    run.add_argument("--plan", required=True)
    run.add_argument("--stage", required=True)
    run.add_argument("--phase", choices=("prefix", "continuation"))
    args = cli.parse_args()
    if args.command == "make":
        print(make(args.output, args.device))
    elif args.stage == "resume" and args.phase is None:
        for phase in ("prefix", "continuation"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "worker", "--plan", args.plan,
                            "--stage", args.stage, "--phase", phase], check=True)
    else:
        print(worker(args.plan, args.stage, args.phase == "continuation"))
