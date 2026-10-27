# Investigating the BF16 MoE numerical failures

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
the separate TP/PP/EP combined case by itself.

## TP2 / ETP2 / PP2 / EP2 combination

A second B200 diagnostic traced the combined case. It reconstructed router
observations by global layer, sample and token position, joining the TP sequence
fragments and PP layer partitions independently of the gradient mapper.
Reference repeat observations again match exactly. The instrumented run's
complete three-step report is identical to the original combined run, and an
independent calculation verified all 1,824 error metrics.

On the second step, layer 2 selects experts `[1, 2]` in the reference and
`[0, 1]` in the candidate for sample 0, token 5. This is a supervised token.
The same step exceeds the local gradient gate: expert-0 `linear_fc2.weight`
has relative L2 error 8.56594%, maximum absolute error `1.2494867133e-4`,
absolute L2 `4.9816596912e-4`, and RMSE `1.1008016716e-5`.
Steps one and three pass the frozen numerical gates.

The third step also has a changed route, at sample 7, token 2. That position
is in the masked prompt. A changed route alone is not an acceptance verdict;
its effect depends on the supervised objective and subsequent computation.

Both router diagnostics establish actual forward differences upstream of the
gradient collector. They support routing sensitivity as an amplification
mechanism. The later expert-linear replay below localizes an earlier EP2
difference; it does not establish the same origin for the combined or production
Qwen case. No historical model implementation or acceptance gate was changed.

These are generated-fixture results. They do not establish that a production
MoE model fails. The [release validation matrix](validation.md) retains both
BF16 MoE failures on all three devices.

## First-step expert weight gradients

A separate B200 trace captured every first-step expert linear input, output,
weight and upstream gradient. It also read the native optimizer's actual
parameter ranges and gradients, without the validator's canonical mapper.
All 16 linears have exactly matching inputs, outputs, weights and upstream
gradients after row alignment. Nevertheless, their native weight gradients
differ between EP1 and EP2 by 0.2281%–0.2445% relative L2. The reference repeat
is identical. The instrumented run's complete three-step report matches the
earlier run, and an independent audit verified all 1,824 report metrics.

The fixture explicitly disables `gradient_accumulation_fusion`. In the tested
native Megatron implementation, that path returns the BF16 result of
`grad_output.T @ input`; the DDP buffer subsequently accumulates it in FP32.
Changing EP changes which tokens enter each microbatch expert GEMM. Therefore
FP32 gradient storage and reduction alone do not prevent earlier BF16 rounding.

An independent CPU replay provides two checks:

| Replay | Result |
|---|---|
| Complete `dY.T @ X` in FP64 with the true supervised-token denominator | Exactly equal between EP1 and EP2 for all 16 linears |
| Per-microbatch BF16 rounding, followed by rank-local FP32 accumulation and reduction | Reproduces 49,150 of 49,152 reference gradient elements and 49,149 of 49,152 candidate elements exactly |
| Per-microbatch FP32 rounding and accumulation versus the FP64 calculation | Maximum relative L2 below `6.8e-8` across these linears |

The remaining rounding-replay discrepancies are at most `1.09e-9` on the
reference and `2.78e-7` on the candidate. CPU FP64 is not the original CUDA
accumulator, so exact rounding ties can differ. This trace identifies an
actual source of first-step EP2 weight-gradient differences, before the later
route change. It does not prove that every model-side difference has this cause.
See the [complete sanitized replay evidence](evidence/0.3.0rc3-moe-linear.json).

## Does capture change training?

The exact rc3 wheel also passed a separate B200 test with capture disabled,
disabled again, and enabled, always using the same topology, weights and
eight masked samples. Each run executes three native optimizer steps.
EP2, TP2/ETP2/PP2/EP2, and TP2 with native CPU optimizer offload all match
exactly. An independent recorder compares rank-local gradients, model weights,
optimizer parameters, full Adam state tensors, counters and step results.
It does not use canonical parameter mapping. All 2,316 tensor-state snapshots
match both the disabled repeat and the enabled run, alongside scalar metadata.

Both sides use the same read-only independent recorder; this is an invariance
check for these generated fixtures and runtime, not a universal guarantee of
zero instrumentation effects. An initial attempt could not serialize a
class-valued CPU optimizer option. The recorder was fixed, and all three cases
passed in a fresh run. No installed library file changed.
See [invariance evidence](evidence/0.3.0rc3-observer-invariance.json) and
`tests/gpu_observer_invariance.py` for reproduction.

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

For the combined case, copy a fresh `tests/gpu_slime_matrix.py` to a different
diagnostic directory and apply `tests/moe_combined_router_trace.patch` instead.
Run with `--cases combined_bf16` and a new output directory. Each stage uses
eight GPUs, so the launcher reuses that allocation sequentially. Audit its
completed `combined_bf16` directory using `audit_small_report.py` and
`audit_combined_router_trace.py` with the same `--results`/`--output` arguments.

The expert-linear patch targets the original release-tag fixture. This avoids
mixing later test-helper additions into the recorded trace source:

```bash
mkdir -p test-results/linear-diagnostic
git show v0.3.0rc3-verified:tests/gpu_slime_matrix.py > test-results/linear-diagnostic/gpu_slime_matrix.py
patch -d test-results/linear-diagnostic -p0 < tests/moe_linear_trace.patch
python test-results/linear-diagnostic/gpu_slime_matrix.py launch \
  --source /path/to/slime-with-precision-check \
  --wheel dist/precision_validator-0.3.0rc3-py3-none-any.whl \
  --output /shared/checks/linear-diagnostic --cases ep_bf16
```

For the completed `ep_bf16` case, run `scripts/audit_linear_trace.py` and
`scripts/audit_linear_rounding.py`, each with `--results CASE_DIRECTORY` and
a distinct `--output NEW_JSON_FILE`. Both use NumPy only. The replay does not
replace native training gradients or change the failed acceptance verdict.
