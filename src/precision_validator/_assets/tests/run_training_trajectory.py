"""Run an actual native masked-token training trajectory and process restart."""
import argparse
from pathlib import Path
import subprocess
import sys

from precision_validator.common import ASSET_ROOT, save, require, now, digest
from precision_validator.trajectory import prepare, STAGES
from precision_validator.trajectory_compare import compare_suite


def run(output, device="cpu"):
    output = Path(output).resolve()
    subprocess.run([sys.executable, str(ASSET_ROOT / "examples/training_trajectory.py"), "make", "--output", str(output), "--device", device], check=True)
    prepare(output / "spec.json", output / "results", timeout=300, budget_gib=0.1)
    for stage in STAGES:
        subprocess.run([sys.executable, "-m", "precision_validator", "run", "--plan", str(output / "results/plan.json"), "--stage", stage], check=True)
    report = compare_suite(output / "results/plan.json")
    require(report["status"] == "passed", f"Trajectory failed: {[k for k,g in report['gates'].items() if not g['passed']]}")
    receipt = {"status": report["status"], "utc": now(), "device": device, "scope": report["scope"], "coverage": report["coverage"],
        "acceptance_sha256": digest(output / "results/acceptance.json"),
        "max_abs_by_pair": {pair: {field: max(step[pair][field]["max_abs"] for step in report["comparisons"].values())
            for field in next(iter(report["comparisons"].values()))[pair]} for pair in ("repeat", "candidate", "resume")}}
    save(output / "receipt.json", receipt)
    print(receipt)
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", required=True)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = cli.parse_args()
    run(args.output, args.device)
