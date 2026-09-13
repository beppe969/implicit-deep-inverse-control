# Source provenance

The authoritative scientific baseline is the supplied `revision_v3.zip`,
including `paper_rev.tex`. The public repository was assembled from
`revision_v3_light_review.zip`, which is the light revision derived solely from
that archive. No older project source was used.

`REFERENCE_MANIFEST.json` records SHA-256 hashes of both input archives and each
of the 21 frozen scientific files. These are the four Python scripts, five saved
result/checkpoint files, and twelve original experimental graphic files.

The three original experiment/review scripts and every original result or graphic
were also compared directly with the corresponding member of `revision_v3.zip`.
Their bytes agree. The fourth script, `audit_light_revision.py`, is the separate
numerical audit added during the current light revision.

Public packaging adds a safe runner and a table exporter. It adds unit tests and
GitHub workflow definitions. The supplementary numerical documentation has been
updated to point to the safe commands. The existing architecture drawing was
extracted into a standalone TikZ source without changing the drawing itself.

Internal scientific assessments and editorial records are excluded. The full
manuscript sources and manuscript PDFs are also excluded; the public deposit
contains the reproducibility materials and citation metadata.

The supplied reference JSON and summary CSV are retained unchanged, including
their historical formatting. Additional portable exports normalize unavailable
values and correctly quote CSV fields. New runs always write to distinct
workspaces. Root-level scientific inputs remain the manuscript reference.
