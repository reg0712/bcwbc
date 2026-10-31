"""M0 scaffold checks: the repo layout exists, every module imports, the CLI lists every
command, an untagged node counts as a write, and tests cannot reach the network."""

from __future__ import annotations

import importlib
import re
import socket
import subprocess
import sys

import pytest

from synapse import ROOT
from synapse.ir import Effect, Node, NodeKind

LAYOUT = [
    "requirements.txt", "Makefile", ".env.example",
    "config/models.yaml", "config/mock.yaml", "config/prices.yaml",
    "mock_services/bank_api.py", "mock_services/data/applicants.json",
    "workflows/loan_prescreen.yaml", "workflows/complaint_triage.yaml",
    "workflows/research_agent_langgraph.py",
    "prompts", "data/hotpot_20.json", "tests", "ui", "bench/results",
]
MODULES = [
    "synapse.ir", "synapse.parser", "synapse.graph", "synapse.executor",
    "synapse.backends.llm", "synapse.backends.tools", "synapse.backends.retrieval",
    "synapse.profiler", "synapse.store", "synapse.cost",
    "synapse.passes.base", "synapse.passes.dead_nodes", "synapse.passes.cse",
    "synapse.passes.parallelise", "synapse.passes.memoise", "synapse.passes.routing",
    "synapse.passes.cascade", "synapse.passes.fusion", "synapse.optimiser",
    "synapse.validator", "synapse.importers.langgraph_importer", "synapse.api",
    "synapse.cli", "synapse.config", "mock_services.bank_api", "mock_services.loan",
]
COMMANDS = ["run", "compile", "profile", "compare", "validate", "bench", "import-langgraph"]


@pytest.mark.parametrize("path", LAYOUT)
def test_layout_path_exists(path: str) -> None:
    assert (ROOT / path).exists(), f"missing {path}"


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_cli_help_lists_every_command() -> None:
    result = subprocess.run([sys.executable, "-m", "synapse.cli", "--help"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    listed = re.search(r"\{([\w,-]+)\}", result.stdout)
    assert listed is not None
    assert listed.group(1).split(",") == COMMANDS


def test_node_without_effect_counts_as_write() -> None:
    assert Node(id="x", kind=NodeKind.TOOL).effect is Effect.WRITE


def test_tests_cannot_reach_the_network() -> None:
    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("example.com", 80), timeout=1)
    with socket.socket() as sock, pytest.raises(RuntimeError, match="network"):
        sock.connect(("8.8.8.8", 53))
