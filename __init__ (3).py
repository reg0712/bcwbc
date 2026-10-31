"""SYNAPSE: compiles AI agent workflows into dependency graphs, profiles their runs,
applies optimisation passes, and proves the outputs are unchanged."""

from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parent.parent
"""Repository root; config/, prompts/, workflows/ and data/ resolve from here."""
