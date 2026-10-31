# SYNAPSE developer commands. Run `make setup` once, then `make test`.
# Everything runs in the conda env ENV_NAME, built from conda-forge only.
CONDA     ?= conda
ENV_NAME  ?= synapse
ENV_DIR   := $(shell $(CONDA) info --base 2>/dev/null)/envs/$(ENV_NAME)
PY        := $(ENV_DIR)/bin/python
MOCK_PORT ?= 8001
API_PORT  ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup test mock api ui demo bench

help: ## List targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-z]+:.*## / {printf "  %-6s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Create the conda env and install requirements (CPU-only torch)
	@test -x $(PY) || $(CONDA) create -y -n $(ENV_NAME) -c conda-forge --override-channels python=3.12 pip
	$(PY) -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .
	$(PY) -c "from synapse.backends.retrieval import download_model; download_model()"

test: ## Run pytest with mock backends only, no network
	$(PY) -m pytest

mock: ## Start the mock bank APIs
	$(PY) -m uvicorn mock_services.bank_api:app --host 127.0.0.1 --port $(MOCK_PORT)

api: ## Start the SYNAPSE API server
	$(PY) -m uvicorn synapse.api:app --host 127.0.0.1 --port $(API_PORT)

ui: ## Start the local dashboard
	$(PY) -m uvicorn synapse.api:app --host 127.0.0.1 --port $(API_PORT)

demo: ## Start the local dashboard with in-process mock bank services
	$(PY) -m synapse.cli profile workflows/loan_prescreen.yaml --runs 3 --latency-scale 0.01
	$(PY) -m synapse.cli validate workflows/loan_prescreen.yaml --latency-scale 0.001
	$(PY) -m uvicorn synapse.api:app --host 127.0.0.1 --port $(API_PORT)

bench: ## Run the benchmark harness, writing bench/results/*.json
	$(PY) -m synapse.cli bench
