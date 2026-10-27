# Precision Validator

[中文说明](README.zh-CN.md) · [Architecture and adapters](docs/architecture.md) · [Tested runtimes and storage cost](docs/compatibility.md)

[GitHub repository](https://github.com/samaritan1998/precision-validator) · [Releases](https://github.com/samaritan1998/precision-validator/releases) · [CPU CI](https://github.com/samaritan1998/precision-validator/actions/workflows/ci.yml)

Compare complete training gradients, actual optimizer updates, and declared
forward observations across implementations, precision settings, and parallel
topologies. The default installation requires only NumPy; training adapters use
your existing training environment.

**Software release:** `0.3.0`. The validator records both numerical passes and
numerical failures. Its software verification checks capture, mapping, metrics,
acceptance decisions and supported integrations against independent evidence.
A software release does not certify that every tested model configuration is
numerically equivalent. See the [0.3.0 verification record](docs/0.3.0-software-release.md),
[model findings](docs/moe-diagnostic.md), and [validation scope](docs/validation.md).

## Install the verified distribution

The release assets include the exact GPU-tested wheel and `SHA256SUMS`.
Download them from the [verified release](https://github.com/samaritan1998/precision-validator/releases/tag/v0.3.0),
then follow the [hash verification and installation instructions](docs/delivery.md).
No PyPI publication is claimed. An unrelated package with the same name is not
this distribution.

## Install from source

Linux and Python 3.10 or newer are required.

```bash
python -m pip install .
precisionctl --version
precisionctl env
precisionctl selftest
```

For a development checkout, use `python -m pip install -e '.[dev]'`.
Optional extras `torch` and `hf` are available for clean example environments;
do not use them to overwrite a working distributed training stack.
Use a release tag for a reproducible checkout; `main` may contain later
documentation or tooling changes.

## Slime: add validation options to the training command

First prepare an isolated source copy:

```bash
python -m precision_validator.slime_check install-bridge \
  --source /path/to/slime \
  --destination /path/to/slime-with-precision-check
export SLIME_ROOT=/path/to/slime-with-precision-check
```

All Ray training workers must have the same installed wheel and source copy.
The installer checks recognized source boundaries and fails on incompatible
versions. Installing the wheel alone does not patch an existing framework.

Keep your model, input, loss, optimizer, batch, and candidate topology arguments:

```bash
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 1 \
  --precision-check-reference tp=1,etp=1 \
  --precision-check-dir /shared/checks/tp-comparison-001
```

This executes three independent bounded training runs from the same initial
weights: reference, reference repeat, and the original candidate configuration.
It compares corresponding optimizer steps and exits. Unspecified topology axes
keep their original values; DP is derived from the allocated GPU count.
The driver reuses the allocation sequentially. It does not enumerate a matrix.

| Original candidate | Reference override | Compared reference |
|---|---|---|
| TP1 / PP2 / CP4 | `cp=1` | TP1 / PP2 / CP1 |
| TP2 / PP2 / CP1 / ETP2 | `tp=1,etp=1` | TP1 / PP2 / CP1 / ETP1 |
| TP1 / PP2 / CP1 | `pp=1` | TP1 / PP1 / CP1 |
| TP1 / PP2 / CP1 / EP2 | `ep=1` | TP1 / PP2 / CP1 / EP1 |
| TP2 / PP2 / CP2 / ETP2 | `cp=1` | TP2 / PP2 / CP1 / ETP2 |

The model and backend must support each topology and both sides must fit memory.
Examples are configuration illustrations, not GPU acceptance claims.
[Chinese Slime guide](docs/slime-training-check.zh-CN.md).

Current Slime scope: synchronous text SFT, fresh HF initial weights, native
distributed Adam, and one model chunk. Intermediate-checkpoint sampling,
continue-training-after-validation, RL, multimodal training, and virtual
pipeline are not enabled. Adam state types, finiteness, and counters are checked;
this entry does not compare all state-vector values. Use the trajectory adapters
for their separately documented state and resume scope.

## Run a portable CPU example

Requires an existing PyTorch installation; no model download is needed.

```bash
precisionctl init --adapter torch --output ./example
python example/torch_models.py make --model mlp \
  --output ./input --example-thresholds
precisionctl prepare-generic --spec ./input/spec.json \
  --output ./results --disk-budget-gib 0.1 --retain-raw
precisionctl run --plan ./results/plan.json --stage reference
precisionctl run --plan ./results/plan.json --stage repeat
precisionctl run --plan ./results/plan.json --stage candidate
precisionctl compare --plan ./results/plan.json --retain-raw
```

The three stages can also be scheduled independently on disjoint resources.
The example's tolerances apply only to its synthetic fixture.

## Understand a result

- `passed`: declared gates passed and required evidence was complete.
- `report_only`: metrics were generated without an acceptance policy.
- `numerical_failure`: complete evidence exceeded at least one declared gate.
- `execution_failure` or `incomplete_evidence`: no precision-pass conclusion.

Reports include maximum absolute error, absolute L2, RMSE, relative L2,
per-parameter and grouped metrics. Strict acceptance covers gradients, updates,
and declared forward observables, including near-zero reference handling.
Choose tolerances before execution. Same-configuration consistency alone does
not exclude a bug shared by both sides; use known-good regression evidence too.

The Slime entry accepts `--precision-check-policy /path/to/policy.json` with
`policy` and `acceptance` keys. Without it, the result is `report_only`.
Full evidence is large: three stages, N steps, and P trainable elements require
approximately `24 * P * N` bytes for FP32 gradients and updates, plus scratch.

## Build and verify

```bash
python -m pip install build
python -m build
python scripts/verify_wheel.py --wheel dist/precision_validator-0.3.0-py3-none-any.whl \
  --tests tests --evidence ./test-results/installed
```

`--native` additionally requires the full existing Torch/Megatron CPU runtime,
requires zero skipped self-tests, and runs the exported training example.
GPU tests require a separately provisioned compatible Slime/Megatron environment.
The project does not submit cluster jobs or distribute proprietary weights.

## License and contributions

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE),
[contribution guide](CONTRIBUTING.md), and [security notes](SECURITY.md).
