"""Install the actual wheel away from source and exercise it in a CPU runtime.

The caller provides an environment containing Torch/Megatron when --native is
used. Nothing here downloads or installs those training dependencies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import venv


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--wheel", required=True)
    cli.add_argument("--evidence", required=True)
    cli.add_argument("--tests", required=True)
    cli.add_argument("--native", action="store_true")
    args = cli.parse_args()
    wheel = Path(args.wheel).resolve()
    wheel_bytes = wheel.read_bytes()
    wheel_sha256 = hashlib.sha256(wheel_bytes).hexdigest()
    evidence = Path(args.evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2")
    env.pop("PYTHONPATH", None)
    checks = []
    with tempfile.TemporaryDirectory(prefix="precision-installed-wheel-") as temporary:
        root = Path(temporary)
        input_wheel = root / wheel.name
        input_wheel.write_bytes(wheel_bytes)
        installed = root / "venv"
        venv.create(installed, with_pip=True, system_site_packages=True)
        # A nested venv inherits the base interpreter's packages, not its
        # caller venv's NumPy/Torch. Reuse the explicitly supplied runtime
        # without installing or downloading training dependencies. The child
        # venv remains first and the byte/import checks below reject shadowing.
        child_site = installed / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
        inherited = [str(Path(p).resolve()) for p in site.getsitepackages() if Path(p).is_dir()]
        (child_site / 'provided-runtime.pth').write_text(
            ''.join(f'import site; site.addsitedir({p!r})\n' for p in inherited))
        python = str(installed / "bin/python")
        command = str(installed / "bin/precisionctl")

        def run(name, argv, timeout=240):
            with (evidence / f"{name}.log").open("w") as log:
                result = subprocess.run(argv, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            checks.append({"name": name, "returncode": result.returncode})
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {evidence / (name + '.log')}")

        try:
            run("install", [python, "-m", "pip", "install", "--no-deps", "--no-index", str(input_wheel)])
            code = (
                "import json, pathlib, precision_validator; "
                "p=pathlib.Path(precision_validator.__file__).resolve(); "
                f"assert p.is_relative_to(pathlib.Path({str(installed)!r})), str(p); "
                "print(json.dumps({'version':precision_validator.__version__,'module':str(p)}))"
            )
            run("installed-import", [python, "-c", code])
            verify = '''
import hashlib, json, pathlib, zipfile, precision_validator
root = pathlib.Path(precision_validator.__file__).resolve().parent.parent
records = {}
with zipfile.ZipFile(WHEEL) as archive:
    for name in archive.namelist():
        if name.startswith("precision_validator/"):
            data = (root / name).read_bytes()
            assert data == archive.read(name), name
            records[name] = hashlib.sha256(data).hexdigest()
print(json.dumps(records, indent=2))
'''.replace("WHEEL", repr(str(input_wheel)))
            run("installed-files", [python, "-c", verify])
            run("environment", [command, "env"])
            run("packaging-tests", [python, "-m", "unittest", "discover", "-s", str(Path(args.tests).resolve()), "-v"])
            run("selftest", [command, "selftest"] + (["--require-all"] if args.native else []), timeout=600)
            if args.native:
                run("init", [command, "init", "--adapter", "torch", "--output", str(root / "template")])
                run("input", [python, str(root / "template/torch_models.py"), "make", "--model", "mlp",
                              "--output", str(root / "input"), "--example-thresholds"])
                # Use the public Python API for prepare/capture/compare as well as the CLI checks above.
                script = '''
import json
from precision_validator.api import prepare_generic, run_stage, compare, status
prepare_generic("input/spec.json", "results", timeout_seconds=120, disk_budget_gib=.1, retain_raw=True)
for stage in ("reference", "repeat", "candidate"):
    run_stage("results/plan.json", stage)
report = compare("results/plan.json", retain_raw=True)
assert report["status"] == "passed", report["status"]
assert status("results/plan.json")["acceptance"] == "passed"
print(json.dumps(report, allow_nan=False))
'''
                run("exported-example-python-api", [python, "-c", script])
            result = "passed"
        except BaseException:
            result = "failed"
            raise
        finally:
            summary = {"status": result, "wheel": wheel.name,
                       "wheel_sha256": wheel_sha256,
                       "native_cpu_runtime": args.native, "gpu_tested": False, "checks": checks}
            (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
