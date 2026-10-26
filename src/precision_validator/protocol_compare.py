"""Compare canonical tensors. No model names, token lengths or GPU counts."""
from pathlib import Path
import math
from .common import read, save, require, digest, load_plan, object_hash, now, validate_artifact_path
from .protocol import STAGES, FIELDS, identity
from .storage import verify_storage, index_pieces, aligned_chunks
from .metrics import Metrics, evaluate
from .acceptance import strict_gates, group_for
from .parallel import validate_rank_groups
from .precision import stage_precision, validate_precision_evidence


class Stage:
    def __init__(self, plan, name):
        self.plan, self.name, self.folder = plan, name, Path(plan["output"]) / name
        final = read(self.folder / "complete.json")
        require(final["status"] == "stage_complete" and final["stage"] == name and final["plan_sha256"] == plan["plan_sha256"], "Stage completion mismatch")
        self.completion_sha256 = digest(self.folder / "complete.json")
        require(digest(self.folder / "process.json") == final["process_sha256"] and read(self.folder / "process.json")["returncode"] == 0, "Process did not exit successfully")
        count = plan["configurations"][name]["world_size"]
        require(set(final["ranks"]) == {f"rank{i}.json" for i in range(count)}, "Missing or extra rank completion")
        self.initial, self.auxiliary, self.observations, self.raw_files, self.ranks = {}, {}, {}, {}, []
        pieces = {field: [] for field in FIELDS}
        runtimes = set()
        for rank in range(count):
            path = self.folder / f"rank{rank}.json"
            require(digest(path) == final["ranks"][path.name], "Rank receipt checksum mismatch")
            row = read(path)
            require(all(row.get(k) == v for k, v in identity(plan, name, rank).items()), "Rank input/configuration/contract mismatch")
            execution = row["execution"]
            require(all(execution.get(k) is True for k in ("backward_completed", "gradients_reduced", "optimizer_succeeded", "teardown_completed")) and
                    type(execution.get("optimizer_steps")) is int and execution["optimizer_steps"] == 1, "Incomplete native execution")
            expected_precision = stage_precision(plan, name)
            require(row["precision"] == expected_precision, "Precision contract mismatch")
            validate_precision_evidence(row, expected_precision)
            require(row["environment"] == plan["environment"], "Numerical environment mismatch")
            require(row["device"]["kind"] == plan["configurations"][name]["device"], "Device kind differs from configuration")
            expected_device = plan["configurations"][name].get("device_name")
            require(not expected_device or expected_device in row["device"]["name"], "Device name differs from configuration")
            require(all(row["checks"].get(k) is True for k in plan["required_checks"]), "Required adapter check missing or failed")
            runtimes.add(object_hash(row["runtime"]))
            for param, initial in row["initial_parameters"].items():
                require(param in plan["parameters"] and initial["shape"] == plan["parameters"][param]["shape"], "Initial catalog mismatch")
                require(isinstance(initial.get("dtype"), str) and len(initial.get("sha256", "")) == 64, "Missing initial digest/dtype")
                require(self.initial.setdefault(param, initial) == initial, "Initial parameter replicas disagree")
            for key, value in row["initial_auxiliary"].items():
                require(self.auxiliary.setdefault(key, value) == value, "Initial auxiliary state replicas disagree")
            if row["storage"]["files"]:
                verify_storage(self.folder, row["storage"])
            else:
                require(not row["storage"]["pieces"] and row["storage"]["format"] == "little_endian_float32", "Unowned rank has unexpected tensor pieces")
            for piece in row["storage"]["pieces"]:
                require(piece["field"] in FIELDS, "Unknown vector field")
                pieces[piece["field"]].append(piece)
            for file, evidence in row["storage"]["files"].items():
                require(file not in self.raw_files, "Multiple raw-file owners")
                self.raw_files[file] = evidence
            for key, observation in row["observations"].items():
                require(key in plan["observables"] and key not in self.observations, "Observation ownership is not unique")
                shape = plan["observables"][key]
                require(observation["shape"] == shape and len(observation["values"]) == math.prod(shape) and
                        all(math.isfinite(v) for v in observation["values"]), "Observation coverage/nonfinite failure")
                self.observations[key] = observation
            self.ranks.append(row)
        require(len(runtimes) == 1, "Worker runtimes differ")
        self.runtime = runtimes.pop()
        self.indices = {field: index_pieces(rows, plan["trainable_elements"]) for field, rows in pieces.items()}
        require(self.initial.keys() == plan["parameters"].keys(), "Missing initial parameter coverage")
        for field, index in self.indices.items():
            require(index.keys() == plan["parameters"].keys(), f"Missing/extra {field} parameters")
            require(all(rows[0]["shape"] == plan["parameters"][key]["shape"] for key, rows in index.items()), "Canonical parameter shape mismatch")
        require(self.observations.keys() == plan["observables"].keys(), "Incomplete observation coverage")
        validate_rank_groups(plan["configurations"][name], self.ranks)


def compare_pair(reference, candidate):
    require(reference.initial == candidate.initial and reference.auxiliary == candidate.auxiliary, "Initial model/buffer state differs")
    require(reference.runtime == candidate.runtime, "Runtime changed between comparisons")
    result = {}
    for field in FIELDS:
        total, params, groups = Metrics(), {}, {}
        for name, actual, baseline in aligned_chunks(reference.folder, reference.indices[field], candidate.folder, candidate.indices[field]):
            part = Metrics().add(actual, baseline)
            total.merge(part)
            params.setdefault(name, Metrics()).merge(part)
        value = total.result()
        require(value["reference2"] > 0 and value["actual2"] > 0, f"All-zero {field}; no nonvacuous training step")
        value["top_parameters"] = [{"name": key, **metric.result()} for key, metric in sorted(params.items(), key=lambda item: item[1].diff2, reverse=True)[:20]]
        value["parameters"] = {key: metric.result() for key, metric in params.items()}
        for name, metric in params.items():
            group = group_for(name, reference.plan["parameters"][name])
            groups.setdefault(group, Metrics()).merge(metric)
        value["groups"] = {key: metric.result() for key, metric in groups.items()}
        result[field] = value
    for name in reference.observations:
        result["observation/" + name] = Metrics().add(candidate.observations[name]["values"], reference.observations[name]["values"]).result()
    return result


def gates_for(comparisons, policy, acceptance=None):
    gates = {} if policy is None else {field: evaluate(comparisons["repeat"][field], comparisons["candidate"][field], policy[field]) for field in FIELDS}
    gates.update(strict_gates(comparisons, acceptance))
    return gates


def markdown(report):
    lines = [f"# 精度对照：{report['name']}", "", f"状态：**{report['status']}**", f"范围：{report['scope']}",
        "", "完整梯度为归约后、裁剪前的值；更新为优化器实际参数的单步差值。原始比较数据以 FP32 存储，指标以 FP64 累积。",
        "未配置验收阈值时仅输出 report_only。验收覆盖以 acceptance_coverage 为准；旧策略只约束全模型梯度和更新。", "",
        "| 对照 | 指标 | 最大绝对误差 | 绝对 L2 | RMSE | 相对 L2 | 余弦 |", "|---|---|---:|---:|---:|---:|---:|"]
    for pair, fields in report["comparisons"].items():
        for field, values in fields.items():
            numbers = [values[k] for k in ("max_abs", "absolute_l2", "rmse", "relative_l2", "cosine")]
            lines.append(f"| {pair} | {field} | " + " | ".join("未定义" if v is None else f"{v:.8g}" for v in numbers) + " |")
    lines += ["", "## 判定", ""]
    coverage = report.get("acceptance_coverage", {})
    lines += [f"局部参数、分组及全部已声明观测量门限：{'已启用' if coverage.get('strict') else '未启用'}。",
              f"声明拓扑对照：{report.get('parallel_validation') or '未声明'}。", ""]
    for name, gate in report["gates"].items():
        lines.append(f"- {name}：{'通过' if gate['passed'] else '未通过'}；相对误差上限 {gate['relative_limit']}。")
        lines.extend(f"  - {reason}" for reason in gate["reasons"])
    lines += ["", "逐参数、分组误差和运行环境见 acceptance.json。结论限于计划中声明的模型、输入、训练入口与单步对照。",
              "适配器负责正确采集原生张量及其全局坐标；报告系统检查协议、覆盖和数值，不能证明适配器所声明的语义必然正确。", ""]
    return "\n".join(lines)


def cleanup_passed(plan, report):
    # Reuse exact-file, checksum-checked deletion after replacing only stage names.
    require(report["status"] == "passed" and report["plan_sha256"] == plan["plan_sha256"], "Only the accepted plan can be cleaned")
    root = Path(plan["output"])
    require(not root.is_symlink(), "Unsafe cleanup root")
    receipt_path = root / "cleanup.json"
    receipt = read(receipt_path) if receipt_path.exists() else {"plan_sha256": plan["plan_sha256"], "removed": []}
    require(receipt["plan_sha256"] == plan["plan_sha256"], "Cleanup plan mismatch")
    removed, newly_removed = list(receipt["removed"]), 0
    for stage, files in report["raw_artifacts"].items():
        require(stage in STAGES and digest(root / stage / "complete.json") == report["stage_completions"][stage], "Changed stage evidence")
        for name, expected in files.items():
            path = validate_artifact_path(root / stage, name)
            require(path.suffix == ".f32" and path.name.startswith(("gradients.rank", "updates.rank")), "Not owned comparison data")
            key = str(path.relative_to(root))
            if not path.exists():
                if receipt.get("pending", {}).get("path") == key and key not in {p["path"] for p in removed}:
                    removed.append(receipt["pending"])
                require(key in {p["path"] for p in removed}, "Unexplained missing raw artifact")
                continue
            require(path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"], "Changed raw artifact")
            pending = {"path": key, **expected}
            save(receipt_path, {"plan_sha256": plan["plan_sha256"], "removed": removed, "pending": pending})
            path.unlink()
            removed.append(pending)
            newly_removed += pending["bytes"]
            save(receipt_path, {"plan_sha256": plan["plan_sha256"], "removed": removed})
    save(receipt_path, {"status": "complete", "plan_sha256": plan["plan_sha256"], "removed": removed})
    return newly_removed


def compare_suite(plan_path, retain_raw=False):
    plan = load_plan(plan_path)
    require(Path(plan_path).resolve().parent == Path(plan["output"]), "Plan moved")
    stages = {name: Stage(plan, name) for name in STAGES}
    comparisons = {name: compare_pair(stages["reference"], stages[name]) for name in STAGES[1:]}
    gates = gates_for(comparisons, plan["policy"], plan.get("acceptance"))
    status = "report_only" if not gates else ("passed" if all(g["passed"] for g in gates.values()) else "numerical_failure")
    report = {"schema_version": 2, "name": plan["name"], "status": status, "scope": plan["scope"],
        "adapter": plan["adapter"], "plan_sha256": plan["plan_sha256"], "comparisons": comparisons, "gates": gates,
        "policy": plan["policy"], "contract": plan["contract"], "trainable_elements": plan["trainable_elements"],
        "configurations": plan["configurations"], "artifacts": plan["artifacts"], "finished_utc": now(),
        "stage_completions": {name: s.completion_sha256 for name, s in stages.items()},
        "raw_artifacts": {name: s.raw_files for name, s in stages.items()}, "raw_retained": True,
        "workers": {name: [{"rank": r["rank"], "device": r["device"], "runtime": r["runtime"], "timings": r["timings"]} for r in s.ranks] for name, s in stages.items()}}
    report["acceptance"] = plan.get("acceptance")
    report["acceptance_coverage"] = {"strict": plan.get("acceptance") is not None,
        "parameters": len(plan["parameters"]) if plan.get("acceptance") else 0,
        "groups": len(comparisons["candidate"]["gradients"]["groups"]) if plan.get("acceptance") else 0,
        "observations": sorted(plan["observables"]) if plan.get("acceptance") else []}
    report["parallel_validation"] = plan.get("parallel_validation")
    root = Path(plan["output"])
    save(root / "acceptance.json", report)
    (root / "report.md").write_text(markdown(report))
    if status == "passed" and plan["cleanup_on_pass"] and not retain_raw:
        report["raw_bytes_removed"] = cleanup_passed(plan, report)
        report["raw_retained"] = False
        save(root / "acceptance.json", report)
    return report


def compare_regression(reference_path, candidate_path):
    a, b = load_plan(reference_path), load_plan(candidate_path)
    for field in ("adapter", "schema_version", "parameters", "configurations", "contract", "observables", "required_checks", "environment", "policy", "toolkit_files"):
        require(a[field] == b[field], f"Regression contract differs: {field}")
    for field in ("acceptance", "parallel_adapter", "parallel_validation"):
        require(a.get(field) == b.get(field), f"Regression contract differs: {field}")
    require(set(a["artifacts"]) == set(b["artifacts"]), "Artifact roles differ")
    require(all(a["artifacts"][k]["sha256"] == b["artifacts"][k]["sha256"] for k in a["artifacts"] if k != "code"), "Non-code regression input changed")
    left, right = ({name: Stage(plan, name) for name in STAGES} for plan in (a, b))
    noise = compare_pair(left["reference"], left["repeat"])
    suite_gates = {key: gates_for({name: compare_pair(suite["reference"], suite[name]) for name in STAGES[1:]}, a["policy"], a.get("acceptance"))
                   for key, suite in (("reference", left), ("candidate", right))}
    comparisons = {name: compare_pair(left[name], right[name]) for name in ("reference", "candidate")}
    gates = {f"{stage}/{field}": gate for stage, values in comparisons.items()
             for field, gate in gates_for({"repeat": noise, "candidate": values}, a["policy"], a.get("acceptance")).items()}
    passed = all(g["passed"] for g in gates.values()) and all(g["passed"] for group in suite_gates.values() for g in group.values())
    report = {"schema_version": 2, "name": b["name"], "scope": "known_good_vs_candidate_same_configurations", "plan_sha256": b["plan_sha256"],
        "reference_plan_sha256": a["plan_sha256"], "status": "report_only" if not gates else ("passed" if passed else "numerical_failure"),
        "comparisons": comparisons, "gates": gates, "suite_gates": suite_gates,
        "acceptance_coverage": {"strict": a.get("acceptance") is not None},
        "parallel_validation": a.get("parallel_validation")}
    save(Path(b["output"]) / "regression.json", report)
    (Path(b["output"]) / "regression.md").write_text(markdown(report))
    return report
