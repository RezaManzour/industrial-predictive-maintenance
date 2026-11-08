.PHONY: install test lint results figures all
install:
	pip install -e ".[dev,deep]"

test:
	pytest -q --cov=ipm --cov-report=term-missing

lint:
	ruff check .

data:
	python scripts/download_data.py

results: data
	python -m ipm.eda
	python -m ipm.classical
	python -m ipm.multiseed
	python -m ipm.deep
	python -m ipm.cost
	python -m ipm.rl
	python -m ipm.diagnostics

figures:
	python -m ipm.report

all: lint test results figures
