# Release qualification

The `0.3.0rc3` candidate is under validation. It is not a final qualified release.
It adds a required final-step check for native Adam parameter-group counters.
The current rc3 results and the historical rc2 results are recorded separately;
an older artifact's results do not certify the new artifact.
Each positive result must identify the exact wheel hash, source/runtime identity,
model, input, topology, precision, and gate policy. Package/import tests, negative
tests, GPU integration, and production-model acceptance are separate claims.

## Current rc3 candidate

Wheel SHA-256:
`0a7b0084f21228b3edc54e39faeb76d0a0a873b2740cf6a90f8f7572a0f05081`.

- Python 3.10, 3.11, 3.12 and 3.13 installed-wheel checks passed.
- The full CPU runtime passed 31 packaging/bridge tests and 140 self-tests,
  with zero skipped tests. NumPy-only runs explicitly report optional skips.
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
- A new three-stage Qwen 256K comparison is running with rc3, unchanged inputs,
  model computation and frozen numerical gates. There is no numerical verdict yet.
- Final production-model qualification and GitHub publication are incomplete.

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
| Real 256K single-sample full-gradient and update comparison | Interrupted; incomplete rc2 runs were superseded by fresh rc3 runs, with no numerical conclusion |
| Public source, distribution, and documentation audit | In progress |

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
step three. Its complete reported step results match the uninstrumented run;
another 1,824 metrics were independently verified. See the
[diagnostic and reproduction steps](moe-diagnostic.md). The first operation
introducing the difference and the separate combined case remain unresolved.
These findings do not establish that another MoE model or production
configuration fails.

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
