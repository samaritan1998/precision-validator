# Changelog

## 0.3.0rc3 — unreleased

- Check native fused Adam group counters after every observed update, including
  the final step; record actual counter values and reject missing counters.
- Add regression cases for nonfresh, frozen and incorrectly advanced group counters.
- Revalidation is required; rc2 results do not certify the added check.

## 0.3.0rc2 — validation candidate

- Prepare a standalone source distribution, documented Slime integration, and
  repeatable installed-wheel validation for public release.
- Remove private container registry identity from the package's runtime lock;
  runtime image provenance can be supplied through `PRECISION_RUNTIME_IMAGE`.
- Preserve strict local, global, and forward gates and actual optimizer capture.
- Release qualification is pending; see `docs/validation.md`.
