"""Fail-closed evidence validation, streaming comparison and small reports."""
from collections import Counter
import math
from pathlib import Path
import numpy as np
from .common import STAGES, digest, read, save, load_plan, require, object_hash, now, validate_artifact_path
from .profiles import topology
from .storage import index_pieces, verify_storage, aligned_chunks
from .metrics import Metrics, evaluate


class Stage:
    def __init__(self, plan, name, verify_raw=True):
        self.plan, self.name = plan, name
        self.folder = Path(plan["output"]) / name
        final = read(self.folder / "complete.json")
        self.completion_sha256 = digest(self.folder / "complete.json")
        require(final["status"] == "stage_complete" and final["stage"] == name and
                final["plan_sha256"] == plan["plan_sha256"], "Missing or mismatched stage completion")
        require(set(final["ranks"]) == {f"rank{i}.json" for i in range(8)}, "Missing or extra ranks")
        require(digest(self.folder / "process.json") == final["process_sha256"] and
                read(self.folder / "process.json")["returncode"] == 0, "Nonzero or missing process exit")
        require(digest(self.folder / "environment.json") == final["environment_sha256"], "Environment proof changed")
        self.environment = read(self.folder / "environment.json")
        require(self.environment["gpu_checked"] and self.environment["status"] == "environment_passed", "GPU preflight missing")
        require(self.environment["runtime"]["versions"] == plan["runtime_lock"]["versions"], "Runtime differs from plan")
        require(all(self.environment["numerical_environment"].get(k) == v for k, v in plan["profile"]["environment"].items()),
                "Worker numerical environment differs from the profile")
        devices = self.environment["devices"]
        require(len(devices) == 8 and {d["index"] for d in devices} == set(range(8)), "Device coverage mismatch")
        for device in devices:
            model = plan["profile"]["device"]
            require(device["capability"] == plan["profile"]["capability"] and
                    (model in device["name"] or (model == "B200" and device["name"] == "CF-NG-BZZ2-O")), "Wrong device type")
        self.ranks, self.initial, self.forward, self.losses = [], {}, {}, []
        self.pieces = {"gradients": [], "updates": []}
        schedules, dp_counts = {}, Counter()
        layout = topology(plan["profile"], name)
        self.raw_files = {}
        for rank in range(8):
            path = self.folder / f"rank{rank}.json"
            require(digest(path) == final["ranks"][path.name], "Rank receipt checksum mismatch")
            row = read(path)
            require(row["rank"] == rank and row["stage"] == name and row["plan_sha256"] == plan["plan_sha256"], "Rank identity mismatch")
            require(row["topology"] == layout, "Topology differs from plan")
            require(row["optimizer_step_executed"] and row["distributed_groups_destroyed"], "Incomplete optimizer/teardown evidence")
            require(row["gradient_stage"] == "after_prepare_grads_before_clipping", "Wrong gradient observation point")
            require(row["sample_sha256"] == plan["input_proof"]["sample_sha256"], "Sample mismatch")
            observation = row["optimizer_observation"]
            require(observation["captured"] and observation["optimizer_step_calls"] == 1 and observation["optimizer_success"], "Native optimizer execution missing")
            require(row["optimizer_dtypes"] and all(all(d[k] == "torch.float32"
                    for k in ("gradient", "master", "exp_avg", "exp_avg_sq")) for d in row["optimizer_dtypes"]), "Optimizer dtype mismatch")
            require(math.isfinite(row["grad_norm"]), "Nonfinite gradient norm")
            require(set(row["entry"]) == {"initialize", "setup_model_and_optimizer", "train_one_step"}, "Native entry coverage mismatch")
            for item in row["entry"].values():
                require(plan["source_inventory"]["files"][item["relative_path"]] == item["sha256"], "Entry source checksum mismatch")
            native = row["native"]
            require(native["optimizer_updated"] and native["grad_sync"]["calls"] == native["grad_sync"]["completed"] > 0,
                    "Native gradient synchronization/update missing")
            require(len(native["files"]) == 2, "Native model/step receipts missing")
            for relative, expected in native["files"].items():
                require(digest(validate_artifact_path(self.folder, relative)) == expected, "Native receipt changed")
            fla = native["fla"]
            require(fla["mode"] == "two-forward" and fla["capability"] == plan["profile"]["capability"], "FLA pin mismatch")
            expected_kernels = {
                "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": ("BV: 32", "num_warps: 2", "num_stages: 2"),
                "chunk_fwd_kernel_o": ("BK: 64", "BV: 64", "num_warps: 4", "num_stages: 3")}
            require({k["kernel"] for k in fla["configs"]} == set(expected_kernels), "FLA kernel coverage mismatch")
            for k in fla["configs"]:
                require(k["calls"] > 0 and all(v in k["config"] and v in k["best_config"] for v in expected_kernels[k["kernel"]]),
                        "FLA kernel config did not execute")
            for param, initial in row["initial_parameters"].items():
                require(initial["dtype"] == "torch.bfloat16", "Initial dtype mismatch")
                require(len(initial["sha256"]) == 64, "Invalid initial parameter digest")
                require(self.initial.setdefault(param, initial) == initial, "Initial DP replicas disagree")
            schedule = row["schedule"]
            dp = schedule["dp_rank"]
            count = plan["sample_copies_per_step"] // layout["dp"]
            require(dp in range(layout["dp"]) and schedule["microbatches"] == count, "DP contribution mismatch")
            require(len(schedule["indices"]) == len(set(schedule["indices"])) == count, "Duplicate/missing DP sample copy")
            require(sorted(i for batch in schedule["batches"] for i in batch) == list(range(count)) and
                    all(len(batch) == 1 for batch in schedule["batches"]), "Microbatch schedule mismatch")
            require(schedule["supervised_tokens"] == plan["input_proof"]["supervised_tokens"], "Supervised token count mismatch")
            require(schedules.setdefault(dp, schedule["indices"]) == schedule["indices"], "DP replicas use different copies")
            dp_counts[dp] += 1
            if verify_raw:
                verify_storage(self.folder, row["storage"])
            for piece in row["storage"]["pieces"]:
                require(piece["field"] in self.pieces, "Unexpected tensor field")
                self.pieces[piece["field"]].append(piece)
            for relative, info in row["storage"]["files"].items():
                require(relative not in self.raw_files, "Raw file is owned by multiple ranks")
                self.raw_files[relative] = info
            for layer, sample in row["forward_samples"].items():
                require(len(sample["values"]) == len(sample["positions"]), "Forward sample length mismatch")
                for position, values in zip(sample["positions"], sample["values"]):
                    require(len(values) == len(sample["features"]), "Forward feature count mismatch")
                    for feature, value in zip(sample["features"], values):
                        key = (int(layer), position, feature)
                        require(key not in self.forward and math.isfinite(value), "Duplicate/nonfinite forward sample")
                        self.forward[key] = value
            if row["losses"]:
                require("loss" in row["losses"] and all(math.isfinite(v) for v in row["losses"].values()), "Missing/nonfinite loss")
                self.losses.append(row["losses"]["loss"])
            self.ranks.append(row)
        require(dp_counts == Counter({i: 8 // layout["dp"] for i in range(layout["dp"])}), "DP rank multiplicity mismatch")
        require(sorted(i for indices in schedules.values() for i in indices) == list(range(plan["sample_copies_per_step"])), "Global sample contributions differ")
        require(self.losses, "Native loss was not reported")
        require(max(self.losses) - min(self.losses) <= 1e-6, "Native loss replicas disagree")
        expected = plan["profile"]["trainable_elements"]
        self.indices = {field: index_pieces(rows, expected) for field, rows in self.pieces.items()}
        require(self.indices["gradients"].keys() == self.indices["updates"].keys() == self.initial.keys(), "Initial/gradient/update parameter sets differ")
        require(sum(math.prod(p["shape"]) for p in self.initial.values()) == expected, "Initial weight coverage mismatch")
        for field in self.indices.values():
            require(all(rows[0]["shape"] == self.initial[name]["shape"] for name, rows in field.items()), "Initial weight shape mismatch")
        length = plan["profile"]["sequence_length"]
        hidden = plan["profile"]["hidden_size"]
        # Same sampling definition as torch.linspace(...).round() in the adapter.
        positions = np.rint(np.linspace(0, length - 1, 64)).astype(int).tolist()
        features = np.rint(np.linspace(0, hidden - 1, 32)).astype(int).tolist()
        keys = {(layer, p, f) for layer in range(1, plan["profile"]["layers"] + 1) for p in positions for f in features}
        require(self.forward.keys() == keys, "Incomplete forward sample coverage")

    def timing(self):
        return {"ranks": [{"rank": r["rank"], **r["timings"], "peak_allocated_bytes": r["peak_allocated_bytes"],
                          "peak_reserved_bytes": r["peak_reserved_bytes"]} for r in self.ranks],
                "scope": "Instrumented single step; includes compilation, observation and I/O; not production throughput"}


def compare_pair(reference, actual):
    require(reference.initial == actual.initial, "Initial model weights differ (including DP replicas)")
    require(reference.environment["runtime"] == actual.environment["runtime"], "Runtime/build changed between stages")
    require(reference.environment["numerical_environment"] == actual.environment["numerical_environment"],
            "Numerical environment changed between stages")
    report = {}
    for field in ("gradients", "updates"):
        total, per_parameter = Metrics(), {}
        for name, x, y in aligned_chunks(reference.folder, reference.indices[field], actual.folder, actual.indices[field]):
            part = Metrics().add(x, y)
            total.merge(part)
            per_parameter.setdefault(name, Metrics()).merge(part)
        row = total.result()
        require(row["reference2"] > 0 and row["actual2"] > 0, f"All-zero {field}; cannot accept a vacuous step")
        row["top_parameters"] = [{"name": name, **value.result()} for name, value in sorted(
            per_parameter.items(), key=lambda item: item[1].diff2, reverse=True)[:20]]
        by_layer = {}
        for name, value in per_parameter.items():
            parts = name.split(".")
            layer = ".".join(parts[:4]) if ".decoder.layers." in name else name.rsplit(".", 1)[0]
            by_layer.setdefault(layer, Metrics()).merge(value)
        row["layers"] = {name: value.result() for name, value in by_layer.items()}
        report[field] = row
    keys = sorted(reference.forward)
    require(reference.forward.keys() == actual.forward.keys(), "Forward sample alignment differs")
    report["sampled_hidden"] = Metrics().add([actual.forward[k] for k in keys], [reference.forward[k] for k in keys]).result()
    report["loss"] = Metrics().add([sum(actual.losses) / len(actual.losses)],
                                  [sum(reference.losses) / len(reference.losses)]).result()
    return report


def markdown(report):
    lines = [f"# 精度验收：{report['profile']}", "", f"状态：**{report['status']}**",
             f"代码：{report['commit']}", "", "所有完整梯度在归约后、裁剪前采集；更新为实际 FP32 主参数差值。",
             "loss 和采样 hidden 报告绝对/相对误差，当前策略不为它们声明未校准的数值阈值。", "",
             "| 对照 | 指标 | 最大绝对误差 | 绝对 L2 | RMSE | 相对 L2 | 余弦 |", "|---|---|---:|---:|---:|---:|---:|"]
    for pair, fields in report["comparisons"].items():
        for field, values in fields.items():
            numbers = [values[k] for k in ("max_abs", "absolute_l2", "rmse", "relative_l2", "cosine")]
            lines.append(f"| {pair} | {field} | " + " | ".join("未定义（参考为零）" if v is None else f"{v:.8g}" for v in numbers) + " |")
    lines += ["", "## 验收条件", ""]
    for field, gate in report["gates"].items():
        lines.append(f"- {field}：{'通过' if gate['passed'] else '未通过'}，相对 L2 上限 {gate['relative_limit']}。")
        lines.extend(f"  - {reason}" for reason in gate["reasons"])
    lines += ["", "完整分层误差、误差最大的参数、逐 rank 耗时和显存峰值见 acceptance.json。",
              "本结果仅覆盖指定模型、卡型、代码、输入和单步拓扑；不代表其他模型、卡型、训练时长或恢复训练通过。", ""]
    return "\n".join(lines)


def cleanup_passed(plan, report):
    require(report["status"] == "passed" and report["plan_sha256"] == plan["plan_sha256"], "Only the accepted plan can be cleaned")
    require(not Path(plan["output"]).is_symlink(), "Unsafe cleanup root")
    root = Path(plan["output"]).resolve()
    receipt_path = root / "cleanup.json"
    previous = read(receipt_path) if receipt_path.exists() else {"plan_sha256": plan["plan_sha256"], "removed": []}
    require(previous["plan_sha256"] == plan["plan_sha256"], "Cleanup receipt belongs to a different plan")
    removed = list(previous["removed"])
    before_bytes = sum(row["bytes"] for row in removed)
    # The report lists exact owned raw files; never glob/delete directories.
    for stage, files in report["raw_artifacts"].items():
        require(stage in STAGES, "Unexpected cleanup stage")
        require(digest(root / stage / "complete.json") == report["stage_completions"][stage],
                "Stage receipt changed before cleanup")
        for relative, evidence in files.items():
            path = validate_artifact_path(root / stage, relative)
            require(path.suffix == ".f32" and path.name.startswith(("gradients.rank", "updates.rank")), "Not a raw verification shard")
            if not path.exists():
                # Idempotency only after a persisted receipt proves prior deletion.
                key = str(path.relative_to(root))
                if previous.get("pending", {}).get("path") == key:
                    removed.append(previous["pending"])
                require(key in {p["path"] for p in removed}, "Raw file unexpectedly missing")
                continue
            require(path.stat().st_size == evidence["bytes"] and digest(path) == evidence["sha256"], "Raw artifact changed before cleanup")
            pending = {"path": str(path.relative_to(root)), **evidence}
            save(receipt_path, {"status": "in_progress", "plan_sha256": plan["plan_sha256"],
                 "removed": removed, "pending": pending, "utc": now()})
            path.unlink()
            removed.append(pending)
            save(receipt_path, {"status": "in_progress", "plan_sha256": plan["plan_sha256"], "removed": removed, "utc": now()})
    save(receipt_path, {"status": "complete", "plan_sha256": plan["plan_sha256"], "removed": removed, "utc": now()})
    return sum(row["bytes"] for row in removed) - before_bytes


def compare_suite(plan_path, retain_raw=False):
    plan = load_plan(plan_path)
    require(Path(plan_path).resolve().parent == Path(plan["output"]), "Plan location changed")
    stages = {name: Stage(plan, name) for name in STAGES}
    comparisons = {name: compare_pair(stages["cp1"], stages[name]) for name in STAGES[1:]}
    gates = {field: evaluate(comparisons["cp1_repeat"][field], comparisons["cp_target"][field], plan["policy"][field])
             for field in ("gradients", "updates")}
    report = {"schema_version": 1, "status": "passed" if all(g["passed"] for g in gates.values()) else "numerical_failure",
        "profile": plan["profile"]["name"], "commit": plan["source_inventory"]["commit"], "plan_sha256": plan["plan_sha256"],
        "comparisons": comparisons, "gates": gates, "scope": "one_unique_real_256k_sample_one_native_adam_step",
        "sample_copies_per_step": plan["sample_copies_per_step"], "unique_samples": 1,
        "trainable_elements": plan["profile"]["trainable_elements"], "finished_utc": now(),
        "policy_sha256": object_hash(plan["policy"]), "sample_sha256": plan["input_proof"]["sample_sha256"],
        "stage_completions": {name: stage.completion_sha256 for name, stage in stages.items()},
        "timings": {name: stage.timing() for name, stage in stages.items()},
        "raw_artifacts": {name: stage.raw_files for name, stage in stages.items()},
        "raw_retained": True}
    root = Path(plan["output"])
    save(root / "acceptance.json", report)
    (root / "report.md").write_text(markdown(report))
    if report["status"] == "passed" and plan["cleanup_on_pass"] and not retain_raw:
        report["raw_bytes_removed"] = cleanup_passed(plan, report)
        report["raw_retained"] = False
        save(root / "acceptance.json", report)
    return report


def compare_regression(reference_plan, candidate_plan):
    reference, candidate = load_plan(reference_plan), load_plan(candidate_plan)
    # Both suites must retain their raw vectors. They may differ only in source.
    for field in ("profile", "model_inventory", "input_proof", "sample_copies_per_step", "seed",
                  "learning_rate", "policy", "runtime_lock", "toolkit_files"):
        require(reference[field] == candidate[field], f"Regression plans differ in {field}")
    old = {name: Stage(reference, name) for name in STAGES}
    new = {name: Stage(candidate, name) for name in STAGES}
    repeat = compare_pair(old["cp1"], old["cp1_repeat"])
    # A same-topology comparison does not replace either suite's own CP checks.
    suite_gates = {}
    for label, suite, noise in (("reference", old, repeat), ("candidate", new, compare_pair(new["cp1"], new["cp1_repeat"]))):
        target = compare_pair(suite["cp1"], suite["cp_target"])
        suite_gates[label] = {field: evaluate(noise[field], target[field], reference["policy"][field])
                              for field in ("gradients", "updates")}
    comparisons = {name: compare_pair(old[name], new[name]) for name in ("cp1", "cp_target")}
    gates = {f"{name}/{field}": evaluate(repeat[field], values[field], reference["policy"][field])
             for name, values in comparisons.items() for field in ("gradients", "updates")}
    passed = all(g["passed"] for g in gates.values()) and all(g["passed"] for suite in suite_gates.values() for g in suite.values())
    report = {"schema_version": 1, "status": "passed" if passed else "numerical_failure",
        "profile": candidate["profile"]["name"], "commit": candidate["source_inventory"]["commit"],
        "reference_commit": reference["source_inventory"]["commit"], "scope": "known_good_vs_candidate_same_topologies",
        "plan_sha256": candidate["plan_sha256"], "reference_plan_sha256": reference["plan_sha256"],
        "comparisons": comparisons, "gates": gates, "suite_gates": suite_gates, "finished_utc": now()}
    save(Path(candidate["output"]) / "regression.json", report)
    (Path(candidate["output"]) / "regression.md").write_text(markdown(report))
    return report
