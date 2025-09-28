# Makefile (Ruff via uv)
RUFF = uv run ruff
SRC ?= .

.PHONY: format lint

format:
	$(RUFF) check --select I,F401,F403 --fix $(SRC)
	$(RUFF) format $(SRC)

lint:
	$(RUFF) check $(SRC)
