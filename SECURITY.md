# Security and trust boundaries

Experiment specifications and training source are executable inputs. Run only
trusted plans, adapters, checkpoints, and model code. Hashes detect changes;
they do not establish that the original content was trustworthy.

Raw gradients, initial-parameter identities, and input metadata can be sensitive.
Store them in an access-controlled location. Do not publish raw training data or
credentials with a validation report. Python pickle-based checkpoint formats
must be treated as trusted-code artifacts.

Report security defects privately through the repository host's private
vulnerability reporting facility when enabled. Do not put secrets or private
training artifacts in public issues.
