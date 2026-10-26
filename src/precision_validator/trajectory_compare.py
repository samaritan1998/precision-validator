"""Check every training step and full optimizer state; never average away a bad step."""
from fnmatch import fnmatchcase
import hashlib
import math
from pathlib import Path

from .common import read, save, require, digest, object_hash, load_plan, now, validate_artifact_path
from .trajectory import STAGES, FIELDS, STATE_FIELDS, PRECISION, identity
from .trajectory_sdk import cursor_for
from .storage import verify_storage, index_pieces, aligned_chunks, CHUNK_ELEMENTS
from .metrics import Metrics
from .acceptance import group_for, evaluate_limit
from .protocol_compare import gates_for


def piece_digest(folder, piece):
    h = hashlib.sha256()
    with validate_artifact_path(folder, piece["file"]).open("rb") as stream:
        stream.seek(piece["offset_bytes"])
        remaining = piece["elements"] * 4
        while remaining:
            chunk = stream.read(min(remaining, CHUNK_ELEMENTS * 4))
            require(chunk, "Truncated state tensor")
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


class Stage:
    def __init__(self, plan, name):
        self.plan, self.name = plan, name
        self.folder = Path(plan["output"]) / name
        final = read(self.folder / "complete.json")
        require(final["status"] == "stage_complete" and final["stage"] == name and final["plan_sha256"] == plan["plan_sha256"], "Completion mismatch")
        require(digest(self.folder / "process.json") == final["process_sha256"] and
                read(self.folder / "process.json")["returncode"] == 0, "Trajectory process failed")
        require(set(final["ranks"]) == {"rank0.json"} and digest(self.folder / "rank0.json") == final["ranks"]["rank0.json"], "Trajectory receipt mismatch")
        self.row = row = read(self.folder / "rank0.json")
        require(all(row.get(k) == v for k, v in identity(plan, name).items()), "Trajectory identity differs")
        count = plan["contract"]["optimizer_steps"]
        require(row["execution"] == {"optimizer_steps": count, "scheduler_steps": count, "teardown_completed": True}, "Incomplete native trajectory")
        require(row["precision"] == PRECISION and row["environment"] == plan["environment"], "Trajectory precision/environment mismatch")
        config = plan["configurations"][name]
        require(row["device"]["kind"] == config["device"] and
                (not config.get("device_name") or config["device_name"] in row["device"]["name"]), "Unexpected trajectory device")
        require(row["initial"]["data_cursor"] == cursor_for(plan, 0) and not row["initial"]["optimizer"]["state"], "Trajectory did not start fresh")
        require(row["step_receipts"].keys() == {f"steps/{i:06d}/receipt.json" for i in range(1, count + 1)}, "Missing/extra step receipt")
        self.steps, self.raw_files = [], {}
        previous = row["initial"]
        for number, frozen in enumerate(plan["trajectory"]["steps"], 1):
            relative = f"steps/{number:06d}/receipt.json"
            path = validate_artifact_path(self.folder, relative)
            require(digest(path) == row["step_receipts"][relative], "Step checksum differs")
            step = read(path)
            require(step["step"] == number and step["stage"] == name and step["plan_sha256"] == plan["plan_sha256"], "Step identity/order differs")
            require(step["input_sha256"] == plan["artifacts"][frozen["input_artifact"]]["sha256"] and
                    step["objective"] == frozen["objective"], "Step input/objective differs")
            require(all(step["checks"].get(k) is True for k in plan["required_checks"]), "Missing native step check")
            require(step["optimizer_step_counters"] == {n: number for n in plan["parameters"]}, "Optimizer step counter differs")
            state = step["state"]
            require(state["data_cursor"] == cursor_for(plan, number) and state["scheduler"]["last_epoch"] == number, "Scheduler/cursor progression differs")
            require(step["lr_used"] == [g["lr"] for g in previous["optimizer"]["param_groups"]], "Actual learning rate differs from previous scheduler state")
            require(step["observations"].keys() == frozen["observables"].keys(), "Missing/extra observations")
            for key, shape in frozen["observables"].items():
                value = step["observations"][key]
                require(value["shape"] == shape and len(value["values"]) == math.prod(shape) and
                        all(math.isfinite(v) for v in value["values"]), "Nonfinite/incomplete observation")
            verify_storage(path.parent, step["storage"])
            require({p["field"] for p in step["storage"]["pieces"]} == set(FIELDS), "Missing/extra trajectory tensor fields")
            indices = {field: index_pieces([p for p in step["storage"]["pieces"] if p["field"] == field], plan["trainable_elements"]) for field in FIELDS}
            require(state["optimizer"]["state"].keys() == {str(i) for i in range(len(plan["parameters"]))}, "Missing/extra optimizer states")
            for field, index in indices.items():
                require(index.keys() == plan["parameters"].keys(), "Incomplete state/parameter coverage")
                for i, (key, info) in enumerate(plan["parameters"].items()):
                    pieces = index[key]
                    require(len(pieces) == 1 and pieces[0]["shape"] == info["shape"], "Trajectory helper requires complete canonical parameters")
                    if field in STATE_FIELDS:
                        expected = state["model"][key] if field == "parameters" else state["optimizer"]["state"][str(i)][field]
                        require(expected == {"shape": info["shape"], "dtype": "torch.float32", "sha256": piece_digest(path.parent, pieces[0])},
                                "Raw optimizer/parameter state differs from checkpointable state")
            for file, value in step["storage"]["files"].items():
                self.raw_files[f"steps/{number:06d}/{file}"] = value
            self.steps.append({"folder": path.parent, "row": step, "indices": indices})
            previous = state
        if name == "resume":
            checkpoint = read(self.folder / "checkpoint.json")
            evidence = row["resume"]
            boundary = plan["trajectory"]["resume_after_step"]
            require(checkpoint["plan_sha256"] == plan["plan_sha256"] and checkpoint["stage"] == name and checkpoint["initial"] == row["initial"], "Resume checkpoint identity differs")
            require(digest(self.folder / "checkpoint.pt") == checkpoint["sha256"] == evidence["checkpoint_sha256"] and
                    (self.folder / "checkpoint.pt").stat().st_size == checkpoint["bytes"], "Resume checkpoint checksum differs")
            require(checkpoint["cursor"] == cursor_for(plan, boundary) and checkpoint["state"] == self.steps[boundary - 1]["row"]["state"] == evidence["restored_state"], "Resume state/cursor differs from saved prefix")
            require(checkpoint["producer_pid"] == evidence["producer_pid"] and evidence["producer_pid"] != evidence["consumer_pid"], "Fresh resume process not observed")
        else:
            require(row["resume"] is None, "Unexpected checkpoint reuse")


def compare_step(plan, left, right):
    result = {}
    for field in FIELDS:
        total, params, groups = Metrics(), {}, {}
        for name, actual, baseline in aligned_chunks(left["folder"], left["indices"][field], right["folder"], right["indices"][field]):
            part = Metrics().add(actual, baseline)
            total.merge(part)
            params.setdefault(name, Metrics()).merge(part)
        value = total.result()
        if field in ("gradients", "updates"):
            require(value["reference2"] > 0 and value["actual2"] > 0, f"All-zero {field} in trajectory")
        for name, metric in params.items():
            groups.setdefault(group_for(name, plan["parameters"][name]), Metrics()).merge(metric)
        result[field] = {**value, "parameters": {k: v.result() for k, v in params.items()}, "groups": {k: v.result() for k, v in groups.items()}}
    for name, ref in left["row"]["observations"].items():
        result["observation/" + name] = Metrics().add(right["row"]["observations"][name]["values"], ref["values"]).result()
    return result


def exact_gate(equal, message):
    return {"passed": equal, "relative_limit": 0, "reasons": [] if equal else [message]}


def compare_suite(plan_path, retain_raw=True):
    plan = load_plan(plan_path)
    require(Path(plan_path).resolve().parent == Path(plan["output"]), "Plan moved")
    stages = {n: Stage(plan, n) for n in STAGES}
    reference = stages["reference"]
    require(all(s.row["initial"] == reference.row["initial"] for s in stages.values()), "Initial weights/optimizer/scheduler/RNG differ")
    require(all(s.row["runtime"] == reference.row["runtime"] for s in stages.values()), "Trajectory runtimes differ")
    comparisons, gates = {}, {}
    for i in range(plan["contract"]["optimizer_steps"]):
        current = {name: compare_step(plan, reference.steps[i], stages[name].steps[i]) for name in ("repeat", "candidate")}
        current["resume"] = compare_step(plan, stages["candidate"].steps[i], stages["resume"].steps[i])
        prefix = f"step/{i + 1}"
        comparisons[prefix] = current
        numerical = gates_for(current, plan["policy"], plan["acceptance"])
        for category, rules in plan["trajectory"]["state_acceptance"].items():
            for rule_index, rule in enumerate(rules):
                for field in STATE_FIELDS:
                    for name, value in current["candidate"][field][category].items():
                        if fnmatchcase(name, rule["pattern"]):
                            numerical[f"{category}/{field}/{name}/rule{rule_index}"] = evaluate_limit(current["repeat"][field][category][name], value, rule[field])
        for name in ("repeat", "candidate", "resume"):
            left = stages["candidate"].steps[i]["row"] if name == "resume" else reference.steps[i]["row"]
            right = stages[name].steps[i]["row"]
            for key in ("lr_used", "optimizer_step_counters"):
                numerical[f"{name}/{key}"] = exact_gate(left[key] == right[key], f"{key} differs")
            for key in ("scheduler", "data_cursor"):
                numerical[f"{name}/{key}"] = exact_gate(left["state"][key] == right["state"][key], f"{key} differs")
            numerical[f"{name}/optimizer_groups"] = exact_gate(left["state"]["optimizer"]["param_groups"] == right["state"]["optimizer"]["param_groups"], "Optimizer group options differ")
            auxiliary = lambda row: {k: v for k, v in row["state"]["model"].items() if k not in plan["parameters"]}
            numerical[f"{name}/buffers_and_frozen_parameters"] = exact_gate(auxiliary(left) == auxiliary(right), "Buffers/frozen parameters differ")
        # Resume compares the candidate to its own uninterrupted execution exactly.
        # This policy is frozen by this protocol; it cannot be widened after a run.
        numerical["resume/checkpointable_state"] = exact_gate(stages["candidate"].steps[i]["row"]["state"] == stages["resume"].steps[i]["row"]["state"], "Resumed state/RNG/buffers diverged")
        for field, values in current["resume"].items():
            numerical["resume/" + field] = exact_gate(values["max_abs"] == 0, "Resumed values differ from uninterrupted candidate")
        gates.update({f"{prefix}/{name}": value for name, value in numerical.items()})
    report = {"schema_version": 3, "adapter": plan["adapter"], "name": plan["name"], "scope": "single_process_fp32_multi_step_and_fresh_process_resume",
        "plan_sha256": plan["plan_sha256"], "status": "passed" if all(g["passed"] for g in gates.values()) else "numerical_failure",
        "finished_utc": now(), "comparisons": comparisons, "gates": gates, "contract": plan["contract"], "trajectory": plan["trajectory"],
        "coverage": {"steps_per_stage": plan["contract"]["optimizer_steps"], "stages": list(STAGES), "parameters": len(plan["parameters"]),
            "distinct_inputs": len({plan["artifacts"][s["input_artifact"]]["sha256"] for s in plan["trajectory"]["steps"]}),
            "fields": list(FIELDS), "all_declared_observations_gated": True, "resume_new_process": True, "resume_exact": True},
        "raw_retained": True, "stage_completions": {n: digest(s.folder / "complete.json") for n, s in stages.items()},
        "raw_artifacts": {n: s.raw_files for n, s in stages.items()}}
    save(Path(plan["output"]) / "acceptance.json", report)
    lines = [f"# 多步精度验证：{plan['name']}", "", f"状态：{report['status']}。范围：单进程 FP32 原生 Adam/AdamW；不代表分布式、大模型或混合精度验收。", "",
        "每一步均检查完整梯度、实际更新、参数值和 Adam 一阶/二阶状态；局部门限和前向观测量门限参与判定。",
        "恢复分支在新进程中加载 checkpoint，并与候选配置的连续执行逐步精确比较。", "",
        "| 步骤 | 对照 | 指标 | 最大绝对误差 | 绝对 L2 | RMSE | 相对 L2 |", "|---|---|---|---:|---:|---:|---:|"]
    for step, pairs in comparisons.items():
        for pair, fields in pairs.items():
            for field, values in fields.items():
                numbers = [values[k] for k in ("max_abs", "absolute_l2", "rmse", "relative_l2")]
                lines.append(f"| {step} | {pair} | {field} | " + " | ".join("未定义" if v is None else f"{v:.8g}" for v in numbers) + " |")
    lines.extend(["", "未通过项：", *[f"- {key}: {'; '.join(g['reasons'])}" for key, g in gates.items() if not g["passed"]], ""])
    (Path(plan["output"]) / "report.md").write_text("\n".join(lines))
    return report


def cleanup_passed(*args):
    require(False, "Trajectory v1 retains owned raw data and checkpoints; automatic cleanup is not implemented")


def compare_regression(*args):
    require(False, "Trajectory cross-version regression is not implemented; tensor-protocol-v1 still supports regress")
