# Copyright (C) 2023 the project owner
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
"""Delia's status messages for dashboard display and logging.

The watermelon theme is kept for the dashboard UI, but function names
and internal terminology use clear, descriptive names.
"""

import random
from enum import Enum


class StatusEvent(str, Enum):
    """Status events for dashboard display."""

    REQUEST_RECEIVED = "request_received"
    PROCESSING = "processing"
    MODEL_LOADING = "model_loading"
    COMPLETED = "completed"
    READY = "ready"
    CLEANUP = "cleanup"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"


# Status messages for dashboard (watermelon themed for fun UI)
STATUS_MESSAGES = {
    StatusEvent.REQUEST_RECEIVED: [
        "New request incoming...",
        "Processing request...",
        "Request received!",
        "Starting work...",
    ],
    StatusEvent.PROCESSING: [
        "Working on it...",
        "Processing...",
        "Computing response...",
        "Generating output...",
    ],
    StatusEvent.MODEL_LOADING: [
        "Loading model...",
        "Warming up...",
        "Preparing model...",
        "Model initializing...",
    ],
    StatusEvent.COMPLETED: [
        "Response ready!",
        "Complete!",
        "Done!",
        "Finished!",
    ],
    StatusEvent.READY: [
        "Ready to serve!",
        "System healthy!",
        "All systems go!",
        "Standing by!",
    ],
    StatusEvent.CLEANUP: [
        "Cleaning up...",
        "Freeing resources...",
        "Tidying up...",
    ],
    StatusEvent.RATE_LIMITED: [
        "Rate limit reached!",
        "Too many requests...",
        "Cooling down...",
    ],
    StatusEvent.OVERLOADED: [
        "System busy!",
        "High load...",
        "Queue building up...",
    ],
}

# Map technical log events to display-friendly names
LOG_EVENT_DISPLAY = {
    "model_queued": "request_queued",
    "model_loading_start": "loading_model",
    "model_loaded": "model_ready",
    "model_unloaded": "model_released",
    "model_starting": "processing_start",
    "model_thinking": "deep_reasoning",
    "model_response": "generating",
    "model_completed": "completed",
    "queue_processed": "queue_ready",
    "circuit_open": "circuit_breaker",
    "quota_exceeded": "quota_limit",
}

# Tier-specific status messages (watermelon themed for dashboard)
TIER_MESSAGES = {
    "quick": {
        "start": ["Quick model starting...", "Fast response incoming...", "Quick tier active..."],
        "complete": ["Quick response ready!", "Fast result delivered!", "Quick tier done!"],
    },
    "coder": {
        "start": ["Coder model starting...", "Code analysis beginning...", "Coder tier active..."],
        "complete": ["Code response ready!", "Analysis complete!", "Coder tier done!"],
    },
    "moe": {
        "start": ["MoE model starting...", "Deep reasoning beginning...", "MoE tier active..."],
        "complete": ["Deep response ready!", "Reasoning complete!", "MoE tier done!"],
    },
    "thinking": {
        "start": ["Thinking model starting...", "Extended reasoning...", "Thinking tier active..."],
        "complete": ["Thought response ready!", "Extended reasoning done!", "Thinking tier done!"],
    },
}

# Backend health messages
BACKEND_MESSAGES = {
    "ollama": {"healthy": "Ollama ready!", "unhealthy": "Ollama needs attention..."},
    "llamacpp": {"healthy": "llama.cpp ready!", "unhealthy": "llama.cpp unavailable..."},
    "gemini": {"healthy": "Gemini connected!", "unhealthy": "Gemini unavailable..."},
}

# Startup messages (watermelon themed for dashboard)
STARTUP_MESSAGES = [
    "Delia is ready!",
    "Server started!",
    "Ready to serve!",
    "All systems go!",
]


def get_status_message(event: StatusEvent) -> str:
    """Get a random status message for the given event."""
    msgs = STATUS_MESSAGES.get(event, [""])
    return random.choice(msgs)


def get_display_event(technical_event: str) -> str:
    """Map a technical log event to a display-friendly name."""
    return LOG_EVENT_DISPLAY.get(technical_event, technical_event)


def get_tier_message(tier: str, phase: str = "start") -> str:
    """Get a status message for a model tier."""
    tier_data = TIER_MESSAGES.get(tier, TIER_MESSAGES["quick"])
    msgs = tier_data.get(phase, tier_data["start"])
    return random.choice(msgs)


def format_completion_stats(tokens: int, elapsed_ms: int, tier: str) -> str:
    """Format completion statistics for display."""
    if elapsed_ms < 1000:
        speed = "very fast"
    elif elapsed_ms < 5000:
        speed = "fast"
    elif elapsed_ms < 15000:
        speed = "moderate"
    else:
        speed = "thorough"
    return f"Generated {tokens:,} tokens ({speed}) using {tier} tier"


def get_backend_status_message(provider: str, healthy: bool) -> str:
    """Get a status message for a backend's health."""
    status = "healthy" if healthy else "unhealthy"
    msgs = BACKEND_MESSAGES.get(provider, {"healthy": f"{provider} ready!", "unhealthy": f"{provider} unavailable..."})
    return msgs[status]


def get_startup_message() -> str:
    """Get a random startup message."""
    return random.choice(STARTUP_MESSAGES)


def format_queue_status(position: int, total: int) -> str:
    """Format queue position for display."""
    if position == 1:
        return "Next in queue!"
    elif position <= 3:
        return f"Position #{position} in queue ({total} waiting)"
    return f"Position {position} of {total} in queue"


# Backwards compatibility aliases (deprecated, will be removed)
# These map old garden-themed names to new names
GardenEvent = StatusEvent  # Deprecated alias
MESSAGES = STATUS_MESSAGES  # Deprecated alias
VINE_MESSAGES = TIER_MESSAGES  # Deprecated alias
get_message = get_status_message  # Deprecated alias
get_vine_message = get_tier_message  # Deprecated alias
format_harvest_stats = format_completion_stats  # Deprecated alias
