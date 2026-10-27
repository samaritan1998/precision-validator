# Install and reproduce version 0.3.0

Version `0.3.0` is the validator software release. Its model reports can return
`passed`, `report_only`, `numerical_failure`, or an execution/evidence error.
A model numerical failure is not converted into a pass by releasing the tool.
See [software verification](0.3.0-software-release.md) and
[known model findings](moe-diagnostic.md).

## Install the tested wheel

Download `precision_validator-0.3.0-py3-none-any.whl` and `SHA256SUMS` from the
[GitHub release](https://github.com/samaritan1998/precision-validator/releases/tag/v0.3.0).
Preserve the original filename:

```bash
sha256sum -c --ignore-missing SHA256SUMS
python -m pip install ./precision_validator-0.3.0-py3-none-any.whl
precisionctl --version
precisionctl env
precisionctl selftest
```

The canonical wheel SHA-256 is
`c5fefbf9e861e7bc08a41dc6a0ed15fc9a31743ea261f430239f9cd8badc2be8`.
The base installation needs NumPy. Training adapters use the existing validated
Torch/Slime/Megatron runtime. Optional training tests explicitly report skips
when dependencies are absent; those skips do not certify GPU coverage.
No PyPI publication is claimed. A similarly named PyPI package is not this
verified distribution.

## Verify a clean installation

From the matching source checkout:

```bash
python scripts/verify_wheel.py \
  --wheel /path/to/precision_validator-0.3.0-py3-none-any.whl \
  --tests tests --evidence ./test-results/installed
```

`--native` requires the full existing Torch/Megatron CPU runtime and rejects
skipped tests. The verifier installs away from the source directory and checks
all installed package bytes against the supplied wheel. GPU runs have separate
hardware, input, topology, gate and process evidence.

The source distribution, tagged archive and Git bundle are identified
separately in the release manifest. A wheel rebuilt from source may have a
different ZIP hash because of metadata or timestamps. Its runtime package
files can be compared with the canonical wheel, but it is not silently treated
as the identical GPU-tested artifact.

## Publication and scope

[Source](https://github.com/samaritan1998/precision-validator) includes the
package, Slime bridge, adapters, CPU tests, GPU runners, independent auditors,
documentation and sanitized evidence under Apache-2.0. Proprietary weights,
training data, raw model tensors and cluster credentials are excluded.

The older [rc3 release](https://github.com/samaritan1998/precision-validator/releases/tag/v0.3.0rc3-verified)
remains available with its original hashes and model findings. Its historical
executions are not claimed as executions of version 0.3.0.

`scripts/write_release_record.py` retains its strict numerical matrix check:
a matrix with numerical failures still returns `numerical_failure` and a
nonzero exit. The separate software verification record checks the validator's
predeclared outcomes and independent raw-vector calculations. It does not
certify arbitrary models, all topology combinations, or training convergence.
