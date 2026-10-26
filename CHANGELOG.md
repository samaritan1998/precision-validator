# Changelog

## 0.3.0rc2 — unreleased

- Prepare a standalone source distribution, documented Slime integration, and
  repeatable installed-wheel validation for public release.
- Remove private container registry identity from the package's runtime lock;
  runtime image provenance can be supplied through `PRECISION_RUNTIME_IMAGE`.
- Preserve strict local, global, and forward gates and actual optimizer capture.
- Release qualification is pending; see `docs/validation.md`.
