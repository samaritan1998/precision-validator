# Install and reproduce the release candidate

The distribution is `0.3.0rc3`. Its package tests and stated adapter coverage
have executed; the recorded model comparisons still include numerical
failures. Read [release status](release-status.md) before using an example as
evidence for a production training configuration.

## Install the tested wheel

Keep the distribution's original filename. From a directory containing the
wheel and its `SHA256SUMS`:

```bash
sha256sum -c SHA256SUMS
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

The prepared checkout has no GitHub remote configured. Neither GitHub nor
PyPI publication is claimed. A repository destination and authenticated write
access are needed before publishing. `SHA256SUMS` and the delivery manifest
identify the exact wheel, source archive and Git bundle independently of the
future repository URL.

`numerical_failure` reports remain failed experiments. Resolving those
model-side differences requires new frozen comparisons; they are not removed
or converted to passes by packaging the library.
