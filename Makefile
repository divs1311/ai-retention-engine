.PHONY: help install dev data train score campaigns pipeline api dashboard test lint format typecheck docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with every optional extra
	pip install -e ".[all]"

dev:  ## Install with dev tooling and git hooks
	pip install -e ".[all,dev]" pandas-stubs
	pre-commit install

data:  ## Simulate the marketplace
	retention-engine generate

train:  ## Train + evaluate the churn model
	retention-engine train

score:  ## Score active users into the risk queue
	retention-engine score

campaigns:  ## Generate CRM campaigns (uses an LLM if a key is set)
	retention-engine campaigns

pipeline:  ## Run everything end-to-end
	retention-engine run

api:  ## Serve the REST API on :8000
	retention-engine api --reload

dashboard:  ## Launch the Streamlit dashboard on :8501
	streamlit run app/Home.py

test:  ## Run the test suite with coverage
	RE_LOG_LEVEL=WARNING pytest --cov --cov-report=term-missing

lint:  ## Lint and check formatting
	ruff check . && ruff format --check .

format:  ## Auto-format and fix lint
	ruff format . && ruff check . --fix

typecheck:  ## Static type checks
	mypy src

docker:  ## Build and start API + dashboard containers
	docker compose up --build

clean:  ## Remove generated data, artifacts and caches
	rm -rf data artifacts .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov
