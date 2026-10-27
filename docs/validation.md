# Release qualification

The `0.3.0rc3` evidence run is complete. The package and evidence protocol are
verified, but it is not a numerically qualified model-parity release: required
GPU cases and the real Qwen candidate contain preserved numerical failures. It
adds a required final-step check for native Adam parameter-group counters.
The current rc3 results and the historical rc2 results are recorded separately;
an older artifact's results do not certify the new artifact.
Each positive result must identify the exact wheel hash, source/runtime identity,
model, input, topology, precision, and gate policy. Package/import tests, negative
tests, GPU integration, and production-model acceptance are separate claims.

## Current rc3 candidate

Wheel SHA-256:
`0a7b0084f21228b3edc54e39faeb76d0a0a873b2740cf6a90f8f7572a0f05081`.

- Python 3.10, 3.11, 3.12 and 3.13 installed-wheel checks passed.
- The final source tooling and exact wheel passed 44 packaging/bridge tests
  and 140 self-tests in the full CPU runtime, with zero skipped tests. The
  exported three-stage training example also passed. NumPy-only runs
  explicitly report optional skips. See the
  [final native CPU evidence](evidence/0.3.0rc3-final-native.json).
- Source-distribution rebuilds produce the same 69 package files as the tested
  wheel; installation and execution from that rebuilt distribution passed.
- All 30 B200, H800 and A800 native rank-loop comparisons completed: 24 numerical
  passes and six BF16 MoE numerical failures. The case-by-case outcomes match
  the matrix below. No result from rc2 substitutes for these new executions.
- An independent raw-fragment calculation verified all 34,560 reported metrics.
  All 6,309 actual Adam counter values across 810 rank-step receipts matched
  their captured step, including the last update.
- B200, H800 and A800 completed the full Ray entry, native
  precision/offload/resume, and HF adapter integration tests. All five extended
  integration tests passed on each device family, within their scopes below.
- The real Qwen 256K TP1/PP2/CP1 versus TP2/PP2/CP2 comparison completed with
  all 26,895,998,464 gradient and update elements and 65 forward observations.
  Its complete report is `numerical_failure`: reference-repeat gradient/update
  relative L2 is 0.7518898%/4.5315453%, while candidate relative L2 is
  82.30657%/87.39424%. Of 3,055 gates, 2,680 failed; 27 forward hidden
  observations exceeded their frozen gates. Candidate hidden layer 64 has
  5.289861% relative L2 and maximum absolute error 1.6875 against a 2%/
  0.25 gate. Candidate loss relative L2 is 0.225959% and its maximum absolute
  error is 0.0007754334. The raw vectors and full report remain private; the
  sanitized summary is in the [release status](release-status.md).
- Package qualification evidence is complete, while model-parity qualification
  remains false. The [public source and verified assets](delivery.md) preserve
  both the executed coverage and the numerical failures.

The [rc3 evidence snapshot](evidence/0.3.0rc3-progress.json) is kept separately.
Two counter-receipt counterexamples failed against the old rc2 reader and pass
against rc3. The native observer now reads both PyTorch parameter counters and
fused Adam parameter-group counters, checks freshness and advancement on the
final captured step, and records actual values. A missing counter no longer
counts as checked evidence.

## Completed rc2 evidence, before the counter fix

| Required release evidence | Status |
|---|---|
| Built wheel install away from source | Passed; all 69 installed package files matched the tested wheel |
| Source distribution | Passed; rebuilt package bytes match the tested wheel and installed checks pass |
| NumPy-only CPU contracts | Python 3.10, 3.11 and 3.13 passed, optional-runtime skips explicitly recorded |
| Full training-runtime CPU contracts and negative cases | 30 packaging/bridge tests and 138 self-tests passed, zero skips, Python 3.12 |
| Real Slime Ray driver, bounded steps, three independent stages | Three steps per stage passed on B200, H800 and A800; counter limitation above applies |
| GPU TP / CP / PP / EP / combined integration | All 30 three-step cases executed; 24 numerical passes and 6 numerical failures, detailed below |
| BF16 storage and native CPU optimizer offload integration | B200/H800/A800 storage/trajectory/resume checks passed |
| B200 / H800 / A800 candidate-wheel regression | Complete for the matrix and small full Ray fixture; superseded by the counter fix |
| Real 256K single-sample full-gradient and update comparison | Complete; rc3 report is numerical_failure with full raw coverage |
| Public source, distribution, and documentation audit | Complete for the release candidate; model-parity qualification remains false |

The wheel under test has SHA-256
`4810e067ecc60900c7157739d545ba14db8c769c403df47efc9790f2394da5a8`.
The [machine-readable snapshot](evidence/0.3.0rc2-progress.json) records the
completed numerical matrix and independent metric audit. It is a progress
record, not a release certificate.

## Native Slime rank-loop matrix: rc3 and historical rc2

Each case executes reference, repeat, and candidate from identical generated
weights, with eight variable-length masked examples and three real optimizer
updates per stage. The test calls Slime's native training loop and optimizer;
this scope does not include its Ray driver. All three devices used Python
3.12.3, PyTorch 2.11.0+cu129, and CUDA 12.9. Other runtime/source identities
are retained in the experiment manifests.

| Candidate change from the all-ones reference | B200 | H800 | A800 |
|---|---|---|---|
| TP2 / ETP2, FP32 | passed | passed | passed |
| PP2, FP32 | passed | passed | passed |
| EP2, FP32 | passed | passed | passed |
| TP2 / ETP2, BF16 | passed | passed | passed |
| EP2, BF16 | numerical_failure | numerical_failure | numerical_failure |
| CP2, BF16 | passed | passed | passed |
| CP4, BF16 | passed | passed | passed |
| TP2 / ETP2 / CP2, BF16 | passed | passed | passed |
| TP2 / ETP2 / PP2 / EP2, BF16 | numerical_failure | numerical_failure | numerical_failure |
| TP2 / ETP2 with native CPU optimizer offload, BF16 | passed | passed | passed |

The BF16 MoE cases exceed the frozen local gates. In the EP2 case, a third-step
expert gradient has about 12.6% relative L2 error against a 5% local ceiling.
For that H800 parameter, maximum absolute error is `1.3268210023e-4`, absolute
L2 is `6.3246108240e-4`, and RMSE is `1.3975547506e-5`.
They remain numerical failures; no tolerance was changed after execution.
A separate B200 EP2 trace found a real native router expert-choice change at
step three. A TP/PP/EP trace also found a real expert-choice change on its
failing second step. Both complete reports match their uninstrumented runs;
another 1,824 metrics per diagnostic were independently verified. See the
[diagnostic and reproduction steps](moe-diagnostic.md). The first operation
introducing all combined-case differences remains unresolved. A later complete
first-step EP2 expert-linear trace found identical input/output/upstream tensors
but differing native weight gradients. Modeled per-microbatch BF16 rounding
reproduces almost every native gradient element; FP64 complete-vector replay
is identical between EP1 and EP2. This identifies an earlier EP2 error source,
without changing any historical gate or verdict.
These findings do not establish that another MoE model or production
configuration fails.

A separate B200 capture-invariance test passed for EP2, TP2/ETP2/PP2/EP2 and
native CPU optimizer offload. Three steps with capture disabled, disabled again
and enabled have exactly matching native gradients, weights, full Adam state
tensors and counters. The independent recorder compares physical rank-local
states without the canonical mapper. All 2,316 tensor-state snapshots match
in both comparisons. This is additional generated-fixture coverage, not a
production Qwen result or a numerical pass for the differing topologies.
See [its evidence](evidence/0.3.0rc3-observer-invariance.json).

An independent NumPy implementation reconstructs every canonical element from
the saved rank fragments. Across all 30 reports, all 34,560 global and
per-parameter maximum-absolute, absolute-L2, RMSE, and relative-L2 metrics
matched independently for each version. This verifies metric arithmetic and fragment coverage, including
the reported failures. It is not a second proof of the framework's semantic
parameter mapping. Numerical acceptance of a model and correct operation of
the validator are separate claims.

The extended suite has five distinct scopes. The full Ray fixture performs
three actual SFT updates in each of three independent runs. The native storage
fixture compares FP32 reference parameters with BF16 candidate parameters over
three steps and a resumed trajectory. The offload fixture checks native CPU
optimizer offload, TP changes, optimizer state vectors and checkpoint resume.
The generated Llama and Gemma HF fixtures exercise masked SFT and two backends:
their FP32 runs use strict gates, while BF16 runs remain `report_only` and have
their raw outputs checked independently. A successful adapter test does not
turn those BF16 report-only runs into numerical passes.

## Reproduce the GPU checks

Use an existing compatible Slime/Megatron runtime, install the built wheel on
every worker, and prepare an isolated Slime copy with the bridge installer.
The repository includes `tests/gpu_slime_matrix.py`,
`tests/gpu_slime_ray.py`, and `tests/gpu_extended.py`. Their `--help` output
documents the paths and resources they require. Generated fixtures need no
pretrained weights or external training dataset.

The full Ray fixture uses the current upstream Slime requirements, including
`straw-queue` 0.1.5 in this environment. Its `--load` and `--hf-checkpoint`
both refer to the same generated HF checkpoint. The library does not install
Slime's runtime dependencies or replace its training algorithms.

Historical results remain historical. They are not included in a new wheel's
passed count. A configuration example is not an executed test. A skipped test
is not a pass. Missing runtime evidence keeps qualification incomplete.

The initial release does not claim arbitrary models, all topology combinations,
long-run convergence, or semantic correctness from numerical agreement alone.
Checkpoint-resume trajectory adapters and the Slime bounded-run entry expose
different evidence scopes; consult the specific report's coverage fields.

## Machine-readable matrix qualification

`scripts/write_release_record.py` checks the exact wheel, installed package
bytes, complete CPU coverage, report hashes and required GPU cases. It writes
`numerical_failure` when complete cases exceed their frozen gates, and
`incomplete` for missing or failed executions. Neither status qualifies a
release; the command exits nonzero. The
[current matrix record](evidence/0.3.0rc3-matrix-record.json) retains all six
numerical failures and no missing cases.

Completed, hash-matched cases from an interrupted suite may be combined with
nonoverlapping recovery cases. The record preserves each source snapshot hash
and matrix status; unfinished cases never count. Five additional CPU tests
cover failures, incomplete evidence, changed reports and duplicate cases.
This matrix record does not include production-model qualification or certify
arbitrary input adapters.

The follow-up release-tooling audit also requires named CPU check receipts,
nonempty executed test counts and consistent failure/skip counts. Successful
empty discovery cannot certify the wheel. Eight new negative tests and the
re-evaluation of the existing matrix are recorded in
[release status](release-status.md#follow-up-delivery-audit). Runtime package
bytes and frozen numerical policies did not change.
