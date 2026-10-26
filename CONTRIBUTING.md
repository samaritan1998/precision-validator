# Contributing

Use an isolated environment and build the actual wheel before reporting results.
Run `precisionctl selftest` and `python -m unittest discover -s tests -v`.
Optional training dependencies must be recorded; skipped cases are not passes.

For numerical changes, add a meaningful positive fixture and a failure case
that the old implementation mishandled. Keep tolerance policies frozen before
running the acceptance experiment. Do not change thresholds to turn a failure
into a pass. Preserve input, initialization, masks, and objective semantics.

Never commit model weights, raw tensors, private datasets, credentials, internal
hostnames, or cluster job bundles. Use small generated fixtures and sanitized
result summaries. Describe exactly which hardware and runtime were tested.

The integration installer must leave the supplied Slime source unchanged.
Unrecognized framework layouts fail explicitly rather than silently guessing
parameter ownership or optimizer semantics.
