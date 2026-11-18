PYTHON ?= python

.PHONY: install data run train evaluate explain serve test lint format notebook docker clean

install:  ## Install package + dev tools
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

data:  ## Download and checksum the dataset
	$(PYTHON) -m churn download

run:  ## Full pipeline: segment -> train -> evaluate -> explain
	$(PYTHON) -m churn all

train:
	$(PYTHON) -m churn train

evaluate:
	$(PYTHON) -m churn evaluate

explain:
	$(PYTHON) -m churn explain

serve:  ## Start the prediction API on :8000
	uvicorn churn.serve:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

notebook:  ## Re-execute the EDA notebook top to bottom
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

docker:
	docker build -t churn-api .

clean:
	rm -rf artifacts .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
