# Release checklist

Run `python reproduce.py check`, `python reproduce.py test`, a complete checkpoint
run, and `python reproduce.py audit` before publishing. Review
`validation/REPORT.md` and the before/after values. Keep the three supplied
checkpoint files unchanged for this release.

The accompanying full paper package contains `github_update.patch`, generated
against the repository snapshot embedded in the supplied archive. Apply it only
after `git apply --check` succeeds in a clean local clone. The patch includes
updated binary figures and removes superseded validation reports. Review the
resulting `git diff --stat` and rerun the release checks before committing.

The numerical repository package can also initialize a new local repository.
Keep its hidden `.github`, `.gitignore`, and `.gitattributes` entries. It contains
no `.git` directory or credentials. Existing license status is retained.

GitHub Actions are included, with read-only workflow permissions. A successful
local validation does not assert that a remote Actions job has run. Publishing
and remote workflow execution remain the author's actions.
