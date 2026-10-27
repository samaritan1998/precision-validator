# Changelog

## 0.3.0

- Publish the validated software distribution with separate software and model numerical verdicts.
- Revalidate the exact final wheel on B200, H800 and A800, including Ray, precision/offload/resume, HF adapters and capture invariance.
- Add an independent full-state trajectory metric auditor and negative evidence controls.
- Preserve all failed numerical comparisons and historical production Qwen findings; no acceptance threshold is relaxed.


## Source tooling audit after 0.3.0rc3

- Reject incomplete CPU check receipts and empty test discovery during release
  verification, including a reproduced false qualification with no checks.
- Add eight regression tests for release evidence and executed test coverage.
- Recheck the existing numerical matrix without changing its failures, and
  independently audit selected complete production Qwen tensors.
- Document the installable wheel filename and separate delivery status from
  model numerical qualification. The tested wheel and its 69 runtime files
  are unchanged.
- Complete installed-wheel verification of the updated release tooling in the
  full native CPU runtime: 44 repository tests, 140 bundled tests, zero skips,
  and the exported three-stage training example. Preserve GPU failures in the
  regenerated matrix record.

## 0.3.0rc3 — unreleased

- Check native fused Adam group counters after every observed update, including
  the final step; record actual counter values and reject missing counters.
- Add regression cases for nonfresh, frozen and incorrectly advanced group counters.
- Revalidation completed with exact-wheel CPU and three-device GPU evidence;
  see `docs/validation.md` for preserved numerical failures and scope. Rc2
  results do not certify the added check.

## 0.3.0rc2 — validation candidate

- Prepare a standalone source distribution, documented Slime integration, and
  repeatable installed-wheel validation for public release.
- Remove private container registry identity from the package's runtime lock;
  runtime image provenance can be supplied through `PRECISION_RUNTIME_IMAGE`.
- Preserve strict local, global, and forward gates and actual optimizer capture.
- Release qualification is pending; see `docs/validation.md`.
