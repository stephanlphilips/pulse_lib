TOPDIR = $(dir $(lastword $(MAKEFILE_LIST)))
FILES = pulse_lib/fast_scan/qblox_fast_scans_refactor.py

all: static

black:
	@echo === black ===
	@(cd $(TOPDIR); black --check --diff --color $(FILES))

lint:
	@echo === pylint ===
	@(cd $(TOPDIR); pylint $(FILES))

flake:
	@echo === flake8 ===
	@(cd $(TOPDIR); pflake8 $(FILES))

mypy:
	@echo === mypy ===
	@(cd $(TOPDIR); mypy $(FILES))

static: black flake lint mypy
