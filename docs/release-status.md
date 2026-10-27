# Release status

`0.3.0rc3` is a reproducible release candidate. The package and its evidence
protocol are verified, but the recorded model comparisons are not all
numerical passes. The repository keeps those failures in the release evidence;
they are not hidden by changing a gate or replacing a full vector with a
sample.

The exact wheel is
`precision_validator-0.3.0rc3-py3-none-any.whl`, SHA-256
`0a7b0084f21228b3edc54e39faeb76d0a0a873b2740cf6a90f8f7572a0f05081`.

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
