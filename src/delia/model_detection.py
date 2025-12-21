# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Model detection and tier assignment utilities.

Shared logic for detecting available models and assigning them to tiers.
Used by both CLI init wizard and backend_manager auto-detection.
"""

from __future__ import annotations


# Patterns to exclude (vocab files, test files, embedding models, etc.)
EXCLUDED_PATTERNS = [
    "ggml-vocab",
    "vocab-",
    "test",
    "dummy",
    "template",
    "example",
    # Embedding models (not chat models)
    "embed",
    "bert",
    "clip",
    "bge-",           # BGE embedding models
    "bge:",
    "minilm",         # all-minilm and similar
    "nomic-embed",
    "granite-embed",
    "mxbai-embed",
    "e5-",            # E5 embedding models
    "sentence-",      # sentence transformers
]

# Keywords for tier classification
# Priority order matters - first match wins in classify_model()
# Note: "dispatcher" tier removed - dispatching now uses embeddings (faster, more accurate)
TIER_KEYWORDS = {
    "agentic": ["openthinker:7b", "openthinker-agent", "agent"],
    "swe": ["devstral", "deepswe", "swe"],
    "thinking": ["openthinker:32b", "openthinker", "think", "reason", "r1"],
    "coder": ["qwen3-coder:30b", "deepcoder:14b", "coder", "code"],
    "moe": ["nemotron-3-nano", "mixtral", "moe", "30b", "32b"],
    "quick": ["ministral", "7b", "8b", "small", "tiny"],
}


def filter_models(models: list[str]) -> list[str]:
    """Filter out vocab files, test files, and other non-model entries.

    Args:
        models: List of model names from backend

    Returns:
        Filtered list with only actual models
    """
    if not models:
        return []

    filtered = [
        m for m in models
        if m and not any(excl in m.lower() for excl in EXCLUDED_PATTERNS)
    ]

    # Fall back to original if everything was filtered
    return filtered if filtered else models


def classify_model(model_name: str) -> str:
    """Classify a single model into a tier based on its name.

    Args:
        model_name: Name of the model

    Returns:
        Tier name: "thinking", "coder", "moe", "quick", or "dispatcher"
    """
    model_lower = model_name.lower()

    # Check each tier in priority order
    for tier, keywords in TIER_KEYWORDS.items():
        if any(kw in model_lower for kw in keywords):
            return tier

    # Default to quick for unknown models
    return "quick"


def assign_models_to_tiers(models: list[str]) -> dict[str, list[str]]:
    """Assign detected models to tiers based on naming patterns.

    This is the canonical implementation used by both CLI and backend_manager.

    Args:
        models: List of available model names

    Returns:
        Dict mapping tier names to lists of model names. All tiers will have a value
        (using the first available model as fallback).
    """
    filtered = filter_models(models)

    if not filtered:
        return {}

    # If only one model, use it for all tiers
    if len(filtered) == 1:
        model = filtered[0]
        return {
            "quick": [model],
            "coder": [model],
            "moe": [model],
            "thinking": [model],
            "dispatcher": [model],
            "agentic": [model],
            "swe": [model],
        }

    # Multiple models - classify each and assign to tiers
    tiers: dict[str, list[str]] = {
        "quick": [],
        "coder": [],
        "moe": [],
        "thinking": [],
        "dispatcher": [],
        "agentic": [],
        "swe": [],
    }

    # First pass: assign based on keywords
    for model in filtered:
        tier = classify_model(model)
        tiers[tier].append(model)

    # Second pass: fill gaps with first available model
    first_model = filtered[0]
    for tier in tiers:
        if not tiers[tier]:
            tiers[tier] = [first_model]

    return tiers
