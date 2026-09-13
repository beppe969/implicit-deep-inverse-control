# Release checklist for the repository owner

## Before a public release

Agree on the code/data license with the coauthor and add its text as `LICENSE`.
Update `LICENSE_STATUS.md` and the license field in `CITATION.cff` accordingly.
The supplied source archive did not specify a software license.

Choose the repository name. The proposed folder name is
`implicit-deep-inverse-control`. Citation metadata currently uses version `1.0.0`
and identifies the paper as an unpublished manuscript. Update these fields to
match the actual release and add a repository URL once one exists.

Review `validation/REPORT.md`. It distinguishes checkpoint agreement from the
observed differences after training from scratch. The repository excludes the
private manuscript review and the blue revision files.

## Uploading the prepared folder

Create an empty GitHub repository under the intended account. Keep it empty
initially to avoid conflicts with the README and other prepared files. From the
prepared folder, the local initialization commands are:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Add paper reproducibility package"
```

Then use the remote address displayed for the new repository:

```bash
git remote add origin YOUR_REPOSITORY_REMOTE
git push -u origin main
```

`YOUR_REPOSITORY_REMOTE` is a placeholder for the actual SSH or HTTPS Git remote.
The archive contains no `.git` directory and no credentials. GitHub's generated
repository ZIP should contain the contents of this folder, including the hidden
`.github` directory.

## Checking the uploaded repository

Run the regular CI workflow and inspect its report. Then trigger the full
reproducibility workflow with `checkpoints` mode. The workflow includes all three
studies and the independent numerical audit. Its training mode provides a separate
from-scratch run.

Add the actual repository URL to `CITATION.cff`. Update the manuscript's
reproducibility statement with that URL, or with an archival DOI after deposit.
Keep the manuscript DOI and the software-release DOI in their respective metadata
fields. No DOI or remote repository has been invented in this package.

A tagged release can be created once those checks are complete:

```bash
git tag -a v1.0.0 -m "Reproducibility package for the manuscript"
git push origin v1.0.0
```

For an archival software DOI, configure the repository's Zenodo integration
before publishing the associated GitHub release, then verify the resulting
metadata. Zenodo's current instructions are at:
https://help.zenodo.org/docs/github/

## Suggested repository description

> Reproducible experiments for implicit deep-learning model inversion and robust
> nonlinear inverse control, with measured-history certificates and saved models.

Suggested topics: `implicit-learning`, `nonlinear-control`, `model-inversion`.
