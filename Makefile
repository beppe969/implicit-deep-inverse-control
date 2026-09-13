PYTHON ?= python

.PHONY: check test tables checkpoints audit train example
check:
	$(PYTHON) reproduce.py check
test:
	$(PYTHON) reproduce.py test
tables:
	$(PYTHON) reproduce.py tables
checkpoints:
	$(PYTHON) reproduce.py checkpoints
audit:
	$(PYTHON) reproduce.py audit
train:
	$(PYTHON) reproduce.py train
example:
	$(PYTHON) examples/minimal_inverse.py
