# Release status

`0.3.0rc3` is a reproducible release candidate. The package and its evidence
protocol are verified, but the recorded model comparisons are not all
numerical passes. The repository keeps those failures in the release evidence;
they are not hidden by changing a gate or replacing a full vector with a
sample.

The exact wheel is
`precision_validator-0.3.0rc3-py3-none-any.whl`, SHA-256
`0a7b0084f21228b3edc54e39faeb76d0a0a873b2740cf6a90f8f7572a0f05081`.

The source and verified distribution are published through
[`samaritan1998/precision-validator`](https://github.com/samaritan1998/precision-validator).
See [delivery instructions](delivery.md) for the release tag, hash verification
and installation. Public availability does not change any numerical verdict.

The package checks passed across Python 3.10 through 3.13. The complete Python
3.12 training runtime passed the native CPU self-tests and the generated
Torch/Slime integration fixtures. B200, H800 and A800 GPU evidence contains 30
complete matrix cases: 24 passed and six BF16 MoE cases exceeded their frozen
local gates. Independent checks reproduced all 34,560 matrix metrics and all
6,309 recorded Adam counter values.

The real Qwen 256K TP1/PP2/CP1 versus TP2/PP2/CP2 comparison also completed
with all raw tensors and 65 forward observations. Its report is
`numerical_failure`: the reference-repeat gradient/update relative L2 values
are 0.7519% and 4.5315%, while the candidate values are 82.3066% and
87.3942%. The candidate hidden layer 64 relative L2 is 5.28986% against the
frozen 2% observation gate; loss relative L2 is 0.225959%. The full report
remains private with the raw vectors, while the sanitized summary is in
[`validation.md`](validation.md) and
[`evidence/0.3.0rc3-progress.json`](evidence/0.3.0rc3-progress.json).

Therefore this repository is suitable as a reviewed validator release
candidate and diagnostic baseline. It is not a certificate that the tested
MoE or Qwen candidate topology is numerically equivalent, and it should not be
promoted to a stable model-parity release until those model-side differences
are fixed or explicitly accepted as the intended behavior.

## Follow-up delivery audit

The repository's release tooling now rejects empty, incomplete or duplicate
CPU check lists, nonzero or invalid exit codes, empty test discovery, failed
self-tests, and skipped tests claimed as complete native-runtime coverage.
Before this fix, an empty CPU check list could incorrectly satisfy the release
record's success condition. Eight regression tests cover these cases. The
model comparison code, policies, and all 69 files in the tested wheel are
unchanged; this is a source-tooling fix, not a new GPU execution.

The final audit installed the exact wheel in an isolated environment inside
the full Python 3.12 training runtime. All 44 repository tests and 140 bundled
self-tests passed with zero skips. The exported three-stage training example
also passed. This used CPU only, with no GPU allocated. The
[final native evidence](evidence/0.3.0rc3-final-native.json) records the tested
source commit, test and verifier hashes, installed-wheel identity, runtime
versions, and log hashes.

Earlier Python 3.10, 3.11 and NumPy-only 3.13 installed-wheel checks passed
with explicit optional skips. The earlier host Torch check skipped 25
Megatron-dependent tests; the final native run supplies the missing coverage.
The stricter release record was regenerated using this zero-skip CPU evidence
and the existing exact-wheel GPU evidence. It preserved all six numerical
failures, with no missing or failed executions. No numerical gate or GPU
runtime code changed.

A separate read-only Qwen audit reconstructed 305 complete small parameters
(679,424 elements per field) and all 65 forward observations directly from
the saved evidence, without importing the validator. All 5,400 recalculated
metric values matched. It read about 15.6 MiB of selected raw tensors, not the
entire 601 GiB dataset, so it is a subset diagnostic. Candidate forward
differences are already present in layer 1, and layer 36 is the first hidden
observation to exceed its frozen gate. This narrows the investigation but
does not identify the responsible operation.

The wheel's installable filename is
`precision_validator-0.3.0rc3-py3-none-any.whl`. Preserve that filename when
downloading or copying it; a name ending in `-final.whl` is not a valid wheel
filename. See [delivery instructions](delivery.md) and the
[follow-up evidence](evidence/0.3.0rc3-followup-audit.json).

## Capture invariance and first-step MoE replay

Additional B200 tests of the same wheel found exact native-state agreement with
capture disabled, disabled again and enabled, over three optimizer steps for
EP2, combined TP2/ETP2/PP2/EP2 and CPU optimizer offload. The independent
recorder checks gradients, weights and full Adam states without canonical
mapping. All three cases passed in the completed run; no runtime package file
changed. [Invariance evidence](evidence/0.3.0rc3-observer-invariance.json).

The EP2 investigation also captured complete first-step expert linear tensors.
Inputs, outputs, weights and upstream gradients match, while native weight
gradients differ by about 0.23%–0.24%. A CPU replay of the fixture's BF16
microbatch weight-gradient rounding reproduces all but two reference and three
candidate gradient elements out of 49,152. The complete FP64 calculation is
identical between stages. This localizes an earlier model-side EP2 difference;
it does not resolve the production Qwen result or rewrite existing failures.
[Diagnostic and reproduction](moe-diagnostic.md).

A separately frozen control enables native FP32 weight-gradient accumulation
in every stage. EP2 now passes all three steps on B200, H800 and A800; on B200,
maximum local gradient relative L2 is `8.58743e-8`. The combined TP2/ETP2/PP2/EP2
case still fails on B200 on step two (maximum local gradient relative L2
8.57245%). All four complete reports passed independent metric audits. These
new configuration results are kept
[separately](evidence/0.3.0rc3-moe-fp32-wgrad.json); the six historical failures
and production Qwen result remain unchanged.
