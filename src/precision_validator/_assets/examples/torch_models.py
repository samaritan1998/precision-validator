"""Two real tiny CPU training examples, not full-model/GPU certification.

make: materialize frozen synthetic inputs, catalog and experiment specification.
worker: load that input, train one step through the model-independent observer.
"""
import argparse
from pathlib import Path
import torch
from torch import nn
from precision_validator.common import read, save, load_plan
from precision_validator.sdk import capture_torch_step


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(19, 8)
        self.attention = nn.MultiheadAttention(8, 2, dropout=0.0, bias=False, batch_first=True)
        self.norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 3)

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        attention, _ = self.attention(hidden, hidden, hidden, need_weights=False)
        return self.head(self.norm(hidden + attention).mean(dim=1))


def model_for(kind):
    return nn.Sequential(nn.Linear(5, 7), nn.Tanh(), nn.Linear(7, 3)) if kind == "mlp" else TinyAttention()


def make(folder, kind, acceptance=False):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(31415)
    model = model_for(kind)
    data = {"kind": kind, "x": torch.randn(6, 5).tolist() if kind == "mlp" else torch.randint(0, 19, (6, 4)).tolist(),
            "target": torch.randn(6, 3).tolist(), "origin": "synthetic CPU integration example"}
    save(folder / "input.json", data)
    policy = {f: {"repeat_relative_ceiling": 1e-6, "target_relative_ceiling": 2e-4, "noise_multiplier": 2,
                  "noise_slack": 2e-4, "max_abs_ceiling": 1e-5, "rmse_ceiling": 1e-6} for f in ("gradients", "updates")} if acceptance else None
    spec = {"schema_version": 1, "name": f"cpu-{kind}-batch-split", "scope": "integration_test",
        "artifacts": {"code": str(Path(__file__).resolve()), "input": str(folder / "input.json")},
        "parameters": {name: {"shape": list(p.shape), "group": name.rsplit(".", 1)[0]} for name, p in model.named_parameters()},
        "configurations": {"reference": {"world_size": 1, "device": "cpu", "batch_splits": 1},
                           "candidate": {"world_size": 1, "device": "cpu", "batch_splits": 2}},
        "command": ["{python}", str(Path(__file__).resolve()), "worker", "--plan", "{plan}", "--stage", "{stage}"],
        "contract": {"seed": 31415, "gradient_point": "after_reduction_before_clipping",
            "update_point": "optimizer_parameter_after_minus_before", "optimizer_steps": 1, "optimizer_initial_state": "fresh",
            "optimizer": {"name": "AdamW", "options": {"lr": 0.001, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8}},
            "objective": {"normalization": "mean_squared_error_over_all_18_outputs", "samples": 6},
            "precision": {"parameters": "float32", "gradients": "float32", "optimizer_parameters": "float32", "optimizer_state": "float32"}},
        "observables": {"loss": [], "outputs": [6, 3]}, "required_checks": ["full_batch_contributed"],
        "environment": {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}, "policy": policy}
    save(folder / "spec.json", spec)
    return folder / "spec.json"


def worker(plan_path, stage):
    plan = load_plan(plan_path)
    data = read(plan["artifacts"]["input"]["path"])
    torch.set_num_threads(2)
    torch.manual_seed(plan["contract"]["seed"])
    model = model_for(data["kind"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    x = torch.tensor(data["x"], dtype=torch.float32 if data["kind"] == "mlp" else torch.long)
    target = torch.tensor(data["target"], dtype=torch.float32)
    splits = plan["configurations"][stage]["batch_splits"]
    def backward():
        outputs, losses = [], []
        for inputs, labels in zip(x.chunk(splits), target.chunk(splits)):
            output = model(inputs)
            loss = (output - labels).square().sum() / target.numel()
            loss.backward()
            outputs.append(output.detach())
            losses.append(loss.detach())
        return {"loss": torch.stack(losses).sum(), "outputs": torch.cat(outputs)}
    capture_torch_step(plan_path, stage, model, optimizer, backward, checks={"full_batch_contributed": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("make")
    create.add_argument("--output", required=True)
    create.add_argument("--model", choices=("mlp", "attention"), required=True)
    create.add_argument("--example-thresholds", action="store_true")
    run = sub.add_parser("worker")
    run.add_argument("--plan", required=True)
    run.add_argument("--stage", required=True)
    cli = parser.parse_args()
    if cli.command == "make":
        print(make(cli.output, cli.model, cli.example_thresholds))
    else:
        worker(cli.plan, cli.stage)
