# Thin wrapper: every target runs `python tasks.py <target>` (the logic lives in tasks.py so it also works on
# Windows without make). NOTE: this Makefile itself has not been run (make is not installed on the dev machine);
# `python tasks.py <target>` is the tested path.
PYTHON ?= python
TARGETS := setup data train evaluate analysis export test serve docker-build docker-run examples all

.PHONY: $(TARGETS)
$(TARGETS):
	$(PYTHON) tasks.py $@
