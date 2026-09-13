# Contributing

Run `python reproduce.py check` and `python reproduce.py test` before proposing a
change. A checkpoint smoke test is available through
`python reproduce.py checkpoints --experiment static`.

Treat `code/experiments.py`, the supplied checkpoints, and the original paper
outputs as frozen scientific inputs. New experiments belong in separate scripts
or branches. Changes to a scientific baseline should include a new reference
manifest and a record explaining which reported results change.

Please attach a comparison report to numerical changes. Record the dependency
versions and preserve the original evaluation seeds. See `LICENSE_STATUS.md`
for the current license status.
