# Release qualification

The `0.3.0rc2` candidate is under validation. It is not a final qualified release.
Each positive result must identify the exact wheel hash, source/runtime identity,
model, input, topology, precision, and gate policy. Package/import tests, negative
tests, GPU integration, and production-model acceptance are separate claims.

| Required release evidence | Status |
|---|---|
| Built wheel and sdist install away from source | Pending |
| NumPy-only CPU contracts with explicit optional-runtime skips | Pending |
| Full training-runtime CPU contracts and negative cases, zero skips | Pending |
| Real Slime Ray driver, bounded steps, three independent stages | Pending |
| GPU TP / CP / PP / EP / combined integration | Pending |
| BF16 storage and native CPU optimizer offload integration | Pending |
| B200 / H800 / A800 candidate-wheel regression | Pending |
| Real 256K single-sample full-gradient and update comparison | Pending |
| Public source, distribution, and documentation audit | In progress |

Historical results remain historical. They are not included in a new wheel's
passed count. A configuration example is not an executed test. A skipped test
is not a pass. Missing runtime evidence keeps qualification incomplete.

The initial release does not claim arbitrary models, all topology combinations,
long-run convergence, or semantic correctness from numerical agreement alone.
Checkpoint-resume trajectory adapters and the Slime bounded-run entry expose
different evidence scopes; consult the specific report's coverage fields.
