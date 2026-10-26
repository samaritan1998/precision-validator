# Investigating the BF16 EP2 numerical failure

The generated two-layer MoE fixture returns `numerical_failure` under the frozen
5% local relative-L2 ceiling. This result remains a model numerical failure.
The validator does not change model kernels or tolerances to make it pass.

A separate B200 diagnostic used the exact `0.3.0rc3` wheel. Read-only forward
hooks captured both native routers' inputs, weights, probabilities and expert
choices. All three stages completed three optimizer updates with eight masked,
variable-length samples per step. The diagnostic's complete reported step
results are identical to the original uninstrumented B200 EP2 run. An independent
NumPy reconstruction also verified all 1,824 reported error metrics.

## What the trace establishes

The reference repeat has identical router inputs, weights, probabilities and
expert choices at every captured call. The EP2 candidate has identical router
inputs and outputs on the first step. Small differences appear after updates.
On step three, one of the 248 real tokens changes its second selected expert
in the second layer:

| Observation at sample 7, token 13 | EP1 reference | EP2 candidate |
|---|---|---|
| Selected experts, zero-based | 0, 2 | 0, 3 |
| Probability of expert 0 | 0.5227313042 | 0.5227553248 |
| Probability of the other selected expert | 0.4772686660 | 0.4772446752 |
| Expert 2 minus expert 3 score, independent FP64 replay | +0.0003834572 | -0.0000767335 |

This is a supervised token, not padding. The router already uses FP32 scoring.
The independent score replay multiplies each captured input by its captured
router weight; it is diagnostic evidence, not the runtime's original score
tensor. The observed change in expert choice is from the actual native router.

For the affected third-step expert-3 `linear_fc1.weight_v` gradient, relative
L2 error is 12.5962%, maximum absolute error is `1.3268210023e-4`, absolute L2
is `6.3246108240e-4`, and RMSE is `1.3975547506e-5`. Expert 2 also exceeds the
local gradient ceiling. This is consistent with routing amplifying earlier
small differences, and establishes a real forward-computation divergence before
gradient collection. It does not identify the first low-level operation that
introduced the difference, prove that routing explains all error, or resolve
the separate TP/PP/EP combined case.

These are generated-fixture results. They do not establish that a production
MoE model fails. The [release validation matrix](validation.md) retains both
BF16 MoE failures on all three devices.

## Reproduce the trace

Use the same compatible Slime/Megatron runtime and isolated Slime bridge copy as
the main GPU matrix. Install the verified wheel on every worker. Reserve eight
GPUs; the three EP2 stages execute concurrently on disjoint pairs.

From the repository root, create a separate diagnostic runner:

```bash
mkdir -p test-results/router-diagnostic
cp tests/gpu_slime_matrix.py test-results/router-diagnostic/gpu_slime_matrix.py
patch -d test-results/router-diagnostic -p0 < tests/moe_router_trace.patch
python test-results/router-diagnostic/gpu_slime_matrix.py launch \
  --source /path/to/slime-with-precision-check \
  --wheel dist/precision_validator-0.3.0rc3-py3-none-any.whl \
  --output /shared/checks/router-diagnostic --cases ep_bf16
```

The launch exits nonzero if numerical acceptance fails. Inspect
`latest-execution.json`, then the `ep_bf16/acceptance.json` in its execution
directory. A nonzero exit alone does not distinguish numerical failure from
incomplete execution. For a complete case, run both independent audits:

```bash
python scripts/audit_small_report.py \
  --results /shared/checks/router-diagnostic/execution-ID/ep_bf16 \
  --output ./test-results/router-metrics.json
python scripts/audit_router_trace.py \
  --results /shared/checks/router-diagnostic/execution-ID/ep_bf16 \
  --output ./test-results/router-trace.json
```

The trace analyzer is specific to this two-router, three-step fixture. It
excludes padded tokens, aligns calls by sample and rank, and preserves the
hashes of every input trace. It is not a generic production-model adapter.
