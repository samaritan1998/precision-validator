"""Validate release evidence against the exact wheel; never inherit old results.

This only certifies the supplied coverage manifest. It cannot infer that an
untested model or topology works from the name of a test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--wheel', type=Path, required=True)
    cli.add_argument('--cpu-evidence', type=Path, nargs='+', required=True)
    cli.add_argument('--gpu-evidence', type=Path, nargs='*', default=[])
    cli.add_argument('--required-coverage', type=Path, required=True,
                     help='Frozen JSON mapping device families to required case names')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    wheel_sha = sha(args.wheel)
    with zipfile.ZipFile(args.wheel) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        version = next(line.split(': ', 1)[1] for line in metadata.splitlines() if line.startswith('Version: '))
        files = {n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()
                 if n.startswith('precision_validator/')}
    records = []
    full_cpu = False
    for folder in args.cpu_evidence:
        summary = read(folder / 'summary.json')
        check(summary['status'] == 'passed' and summary['wheel_sha256'] == wheel_sha, 'CPU wheel/checks differ')
        check(all(row['returncode'] == 0 for row in summary['checks']), 'CPU check failed')
        check(read(folder / 'installed-files.log') == files, 'Installed package bytes differ')
        selftest = json.loads((folder / 'selftest.log').read_text().splitlines()[-1])
        full_cpu |= summary['native_cpu_runtime'] and selftest['skipped'] == 0 and selftest['status'] == 'selftest_passed'
        records.append(dict(kind='cpu', summary_sha256=sha(folder / 'summary.json'),
                            selftest=selftest, native=summary['native_cpu_runtime']))
    check(full_cpu, 'Full CPU dependency coverage is missing')
    required = read(args.required_coverage)
    check(isinstance(required, dict) and required and all(isinstance(v, list) and v for v in required.values()),
          'Required GPU coverage must be a nonempty mapping of device families to cases')
    actual = {key:set() for key in required}
    for folder in args.gpu_evidence:
        result = read(folder / 'qualification.json')
        check(result['wheel_sha256'] == wheel_sha and result['status'] == 'passed', 'GPU wheel/checks differ')
        # Exact device-family labels must be supplied by the recorded GPU runner.
        family = result['device_family']
        check(family in actual, 'Unrequested hardware cannot replace required coverage')
        for name, case in result['cases'].items():
            report = folder / name / 'acceptance.json'
            check(case['status'] == 'passed' and sha(report) == case['report_sha256'], 'GPU report changed/failed')
            check(read(report)['status'] == 'passed', 'GPU numerical acceptance failed')
            actual[family].add(name)
        records.append(dict(kind='gpu', device_family=family, device=result['device'],
                            qualification_sha256=sha(folder / 'qualification.json'), cases=sorted(result['cases'])))
    missing = {key:sorted(set(names) - actual[key]) for key,names in required.items() if set(names) - actual[key]}
    record = dict(status='qualified' if not missing else 'incomplete', version=version,
        wheel=args.wheel.name, wheel_sha256=wheel_sha, evidence=records,
        coverage_policy_sha256=sha(args.required_coverage), missing_gpu_coverage=missing,
        scope='Only the exact supplied wheel, hardware and cases; no implied production-model coverage')
    check(not args.output.exists(), 'Evidence output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    if missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
