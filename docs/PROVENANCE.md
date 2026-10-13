# Release provenance

This release is derived from the user-supplied `paper_package_submission.zip`.
Its embedded `implicit_deep_inverse_control_github.zip` supplies the repository
layout. The outer archive supplies the submitted manuscript and author images.
The scientific Python files in the outer and embedded archives had identical
SHA-256 hashes before editing.

The three tensor checkpoint files remain byte-identical to the supplied files.
Scientific evaluations, numerical tables, and experimental figures are regenerated
with the repaired code. The new reference is `results/all_results.json`; the
supplied numerical baseline is retained as `validation/baseline_results.json`.
The focused comparison is recorded in `validation/before_after.json`.

`REFERENCE_MANIFEST.json` binds the current scientific files to their hashes and
records the source archive hash. It certifies release-file integrity, rather than
mathematical correctness. The validation report states which calculations were
actually rerun. No remote repository or published release was modified.
