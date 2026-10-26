"""Run real CPU/Gloo training across native TP, PP and EP partitions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from precision_validator.common import ASSET_ROOT, read, save, now, require
from precision_validator.protocol import prepare
from precision_validator.protocol_compare import compare_suite


def main(root, device="cpu", kinds=("tp", "pp", "ep")):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for kind in kinds:
        folder = root / kind
        subprocess.run([sys.executable, str(ASSET_ROOT / "examples/native_parallel.py"), "make", "--kind", kind,
                        "--output", str(folder), "--device", device], check=True)
        prepare(folder / "spec.json", folder / "results", timeout=300, budget_gib=0.1, retain_raw=True)
        for stage in ("reference", "repeat", "candidate"):
            subprocess.run([sys.executable, "-m", "precision_validator", "run", "--plan", str(folder / "results/plan.json"),
                            "--stage", stage], check=True)
        report = compare_suite(folder / "results/plan.json", retain_raw=True)
        require(report["status"] == "passed", f"Native {kind} comparison failed: {report['gates']}")
        evidence[kind] = {"status": report["status"], "scope": f"tiny_{device}_native_partition_only", "elements": report["trainable_elements"],
            "parallel_validation": report["parallel_validation"], "acceptance_coverage": report["acceptance_coverage"],
            "metrics": {pair: {field: {k: values[k] for k in ("max_abs", "relative_l2")}
                for field,values in data.items()} for pair,data in report["comparisons"].items()}}
        print("NATIVE_PARALLEL_CASE_COMPLETE", kind, flush=True)
    save(root / "receipt.json", {"status": "passed", "utc": now(), "gpu_tested": device == "cuda",
        "cases": evidence, "scope": f"{device} native TP1/TP2, PP1/PP2, fixed-route EP1/EP2 integration; not Slime full-model acceptance"})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", required=True)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    cli.add_argument("--kinds", nargs="+", choices=("tp", "pp", "ep"), default=["tp", "pp", "ep"])
    args = cli.parse_args()
    main(args.output, args.device, args.kinds)
