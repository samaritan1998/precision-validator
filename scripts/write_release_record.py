"""Validate release evidence against the exact wheel; never inherit old results.

This only certifies the supplied coverage manifest. It cannot infer that an
untested model or topology works from the name of a test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


CORE_CPU_CHECKS = frozenset({
    'install', 'installed-import', 'installed-files', 'environment',
    'packaging-tests', 'selftest',
})
NATIVE_CPU_CHECKS = frozenset({'init', 'input', 'exported-example-python-api'})


def validate_cpu_checks(summary, selftest):
    """A successful exit or empty check list does not prove test coverage."""
    native = summary.get('native_cpu_runtime')
    check(type(native) is bool, 'CPU runtime scope must be boolean')
    rows = summary.get('checks')
    check(isinstance(rows, list) and rows, 'CPU check receipts are missing')
    names = []
    for row in rows:
        check(isinstance(row, dict) and isinstance(row.get('name'), str),
              'CPU check name is missing')
        check(type(row.get('returncode')) is int and row['returncode'] == 0,
              'CPU check failed or has an invalid exit code')
        names.append(row['name'])
    check(len(set(names)) == len(names), 'Duplicate CPU check receipt')
    required = CORE_CPU_CHECKS | (NATIVE_CPU_CHECKS if native else frozenset())
    check(required <= set(names), f'Missing required CPU checks: {sorted(required - set(names))}')
    fields = ('tests_run', 'failures', 'errors', 'skipped')
    check(isinstance(selftest, dict) and
          all(type(selftest.get(k)) is int and selftest[k] >= 0 for k in fields),
          'CPU selftest counts are missing or invalid')
    check(selftest['tests_run'] > selftest['skipped'], 'CPU selftest executed no tests')
    check(selftest['failures'] == selftest['errors'] == 0, 'CPU selftest contains failures')
    expected = 'selftest_partial' if selftest['skipped'] else 'selftest_passed'
    check(selftest.get('status') == expected, 'CPU selftest status disagrees with counts')
    check(not native or selftest['skipped'] == 0, 'Full CPU runtime has skipped tests')
    return native


def build_record(wheel, cpu_evidence, gpu_evidence, required_coverage):
    wheel_sha = sha(wheel)
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(next(n for n in archive.namelist() if n.endswith('.dist-info/METADATA'))).decode()
        version = next(line.split(': ', 1)[1] for line in metadata.splitlines() if line.startswith('Version: '))
        files = {n:hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()
                 if n.startswith('precision_validator/')}
    records = []
    full_cpu = False
    for folder in cpu_evidence:
        summary = read(folder / 'summary.json')
        check(summary['status'] == 'passed' and summary['wheel_sha256'] == wheel_sha, 'CPU wheel/checks differ')
        check(read(folder / 'installed-files.log') == files, 'Installed package bytes differ')
        selftest = json.loads((folder / 'selftest.log').read_text().splitlines()[-1])
        full_cpu |= validate_cpu_checks(summary, selftest)
        # unittest exits zero when discovery finds no tests. Check its actual
        # run count as well as the recorded process return code.
        packaging = (folder / 'packaging-tests.log').read_text()
        run = re.search(r'^Ran (\d+) tests? in ', packaging, re.MULTILINE)
        outcome = re.search(r'^OK(?: \(skipped=(\d+)\))?\s*$', packaging, re.MULTILINE)
        check(run is not None and outcome is not None and int(run[1]) > int(outcome[1] or 0),
              'CPU packaging suite executed no successful tests')
        check(not summary['native_cpu_runtime'] or not outcome[1],
              'Full CPU packaging suite has skipped tests')
        records.append(dict(kind='cpu', summary_sha256=sha(folder / 'summary.json'),
                            selftest=selftest, native=summary['native_cpu_runtime']))
    check(full_cpu, 'Full CPU dependency coverage is missing')
    required = read(required_coverage)
    check(isinstance(required, dict) and required and all(isinstance(v, list) and v for v in required.values()),
          'Required GPU coverage must be a nonempty mapping of device families to cases')
    actual = {key:set() for key in required}
    numerical_failures = []
    execution_failures = []
    seen = set()
    for folder in gpu_evidence:
        result = read(folder / 'qualification.json')
        check(result['wheel_sha256'] == wheel_sha, 'GPU wheel differs')
        check(result['status'] in ('passed', 'failed', 'running'), 'Unknown GPU matrix outcome')
        # Exact device-family labels must be supplied by the recorded GPU runner.
        family = result['device_family']
        check(family in actual, 'Unrequested hardware cannot replace required coverage')
        for name, case in result['cases'].items():
            check(name in required[family], 'Unrequested case cannot replace required coverage')
            check((family, name) not in seen, 'Duplicate case evidence; select one immutable execution')
            seen.add((family, name))
            if case['status'] == 'execution_failure':
                execution_failures.append(dict(device_family=family, case=name))
                continue
            check(case['status'] in ('passed', 'numerical_failure'), 'Unsupported case outcome')
            report = folder / name / 'acceptance.json'
            check(sha(report) == case['report_sha256'], 'GPU report changed')
            check(read(report)['status'] == case['status'], 'GPU report outcome differs')
            if case['status'] == 'numerical_failure':
                numerical_failures.append(dict(device_family=family, case=name,
                                               report_sha256=case['report_sha256']))
            actual[family].add(name)
        expected = 'passed' if all(c['status'] == 'passed' for c in result['cases'].values()) else 'failed'
        # An interrupted suite can retain immutable completed cases. Only
        # hash-matched final reports count; absent cases remain missing. A
        # separate recovery can supply the rest without rerunning valid cases.
        if result['status'] != 'running':
            check(result['status'] == expected, 'GPU aggregate outcome differs')
        records.append(dict(kind='gpu', device_family=family, device=result['device'],
                            qualification_sha256=sha(folder / 'qualification.json'),
                            matrix_status_at_snapshot=result['status'], cases=sorted(result['cases'])))
    missing = {key:sorted(set(names) - actual[key]) for key,names in required.items() if set(names) - actual[key]}
    status = 'incomplete' if missing or execution_failures else 'numerical_failure' if numerical_failures else 'qualified'
    return dict(status=status, version=version, wheel=wheel.name, wheel_sha256=wheel_sha, evidence=records,
        coverage_policy_sha256=sha(required_coverage), missing_gpu_coverage=missing,
        numerical_failures=numerical_failures, execution_failures=execution_failures,
        scope='Only the exact supplied wheel, hardware and cases; numerical failures remain failures; no implied production-model coverage')


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--wheel', type=Path, required=True)
    cli.add_argument('--cpu-evidence', type=Path, nargs='+', required=True)
    cli.add_argument('--gpu-evidence', type=Path, nargs='*', default=[])
    cli.add_argument('--required-coverage', type=Path, required=True,
                     help='Frozen JSON mapping device families to required case names')
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    check(not args.output.exists(), 'Evidence output already exists')
    record = build_record(args.wheel, args.cpu_evidence, args.gpu_evidence, args.required_coverage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    if record['status'] != 'qualified':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
