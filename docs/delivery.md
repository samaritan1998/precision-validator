# Install and reproduce the release candidate

The distribution is `0.3.0rc3`. Its package tests and stated adapter coverage
have executed; the recorded model comparisons still include numerical
failures. Read [release status](release-status.md) before using an example as
evidence for a production training configuration.

## Install the tested wheel

Download the wheel and `SHA256SUMS` from the
[GitHub release](https://github.com/samaritan1998/precision-validator/releases/tag/v0.3.0rc3-verified).
Keep the distribution's original filename. From a directory containing the
wheel and its `SHA256SUMS`:

```bash
sha256sum -c --ignore-missing SHA256SUMS
python -m pip install ./precision_validator-0.3.0rc3-py3-none-any.whl
precisionctl --version
precisionctl env
precisionctl selftest
```

The tested wheel SHA-256 is
`0a7b0084f21228b3edc54e39faeb76d0a0a873b2740cf6a90f8f7572a0f05081`.
The base install needs NumPy. Optional training tests explicitly report skips
when Torch or Megatron is absent. A successful core self-test does not certify
GPU execution. Use the existing validated training runtime for GPU adapters.

## Source and verification

The source checkout contains the Python package, Slime bridge, examples, CPU
tests, GPU runners, GitHub Actions workflow, Apache-2.0 license, documentation,
and sanitized validation summaries. Proprietary model weights, private
training data, raw model tensors and cluster credentials are excluded.

From the source checkout, verify the exact wheel using its original name:

```bash
python scripts/verify_wheel.py \
  --wheel /path/to/precision_validator-0.3.0rc3-py3-none-any.whl \
  --tests tests --evidence ./test-results/installed
```

`--native` requires an existing full Torch/Megatron runtime and rejects skipped
tests. The verifier installs into a temporary environment, runs away from the
source directory, and checks installed package bytes. The summary preserves
the wheel hash, executed checks and failure status.

The source archive can also be installed with `python -m pip install
./precision_validator-0.3.0rc3.tar.gz`. Its release tooling and documentation
may include follow-up audits. The 69 runtime package files are checked against
the frozen wheel; source/archive hashes are recorded separately. A new build's
ZIP hash can differ because of metadata or timestamps, so do not silently
substitute its identity for the tested wheel.

## Publication state

The public repository is
[`samaritan1998/precision-validator`](https://github.com/samaritan1998/precision-validator).
The `v0.3.0rc3-verified` release keeps the exact tested wheel and a separately
identified source snapshot. The source tag includes later release-tooling,
test and documentation audits; its 69 runtime package files still match the
wheel byte for byte. `SHA256SUMS` and the delivery manifest identify the wheel,
source distribution, source archive and Git bundle separately.

This is a GitHub prerelease. No PyPI publication or stable model-parity
qualification is claimed. The [GitHub CPU workflow](https://github.com/samaritan1998/precision-validator/actions/workflows/ci.yml)
checks clean public installations. It does not allocate GPUs or replace the
full native-runtime and GPU evidence in this repository.

`numerical_failure` reports remain failed experiments. Resolving those
model-side differences requires new frozen comparisons; they are not removed
or converted to passes by packaging the library.
