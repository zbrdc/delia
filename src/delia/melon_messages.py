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
"""Delia's Watermelon-Themed Messages for dashboard display."""

import random
from enum import Enum


class GardenEvent(str, Enum):
    SEED_PLANTED = "seed_planted"
    WATERING = "watering"
    GROWING = "growing"
    HARVEST = "harvest"
    GARDEN_READY = "garden_ready"
    COMPOSTING = "composting"
    DROUGHT = "drought"
    OVERWATERED = "overwatered"


MESSAGES = {
    GardenEvent.SEED_PLANTED: [
        "Planting a fresh seed...",
        "Tucking a new seed into soil...",
        "A new seed joins the garden!",
        "Finding the perfect spot...",
    ],
    GardenEvent.WATERING: [
        "Watering the garden...",
        "Nurturing the melon patch...",
        "Tending to the seedlings...",
        "Giving the vines attention...",
    ],
    GardenEvent.GROWING: [
        "Vines spreading...",
        "Roots growing deeper...",
        "Leaves unfurling...",
        "Soaking up sunshine...",
    ],
    GardenEvent.HARVEST: [
        "Fresh melon picked!",
        "Harvest complete!",
        "Ripe from the vine!",
        "The garden delivers!",
    ],
    GardenEvent.GARDEN_READY: [
        "Garden is thriving!",
        "Patch looking healthy!",
        "Ready to grow!",
        "Soil is perfect!",
    ],
    GardenEvent.COMPOSTING: [
        "Composting old vines...",
        "Clearing the patch...",
        "Making room for new growth...",
    ],
    GardenEvent.DROUGHT: [
        "Garden needs water!",
        "Patch is parched...",
        "Vines are wilting...",
    ],
    GardenEvent.OVERWATERED: [
        "Soil's saturated!",
        "Garden needs a breather...",
        "Let the patch drain...",
    ],
}

LOG_EVENT_DISPLAY = {
    "model_queued": "seed_planted",
    "model_loading_start": "watering_garden",
    "model_loaded": "garden_ready",
    "model_unloaded": "composting",
    "model_starting": "tending_vines",
    "model_thinking": "roots_deepening",
    "model_response": "fruit_ripening",
    "model_completed": "harvest_complete",
    "queue_processed": "seedling_ready",
    "circuit_open": "drought_warning",
    "quota_exceeded": "garden_overflow",
}

VINE_MESSAGES = {
    "quick": {
        "start": ["Quick vine growing...", "Fast sprout emerging...", "Speedy cultivation..."],
        "complete": ["Quick melon harvested!", "Fast fruit picked!", "Speedy harvest done!"],
    },
    "coder": {
        "start": ["Coder vine growing...", "Code tendrils spreading...", "Logic vine climbing..."],
        "complete": ["Code melon harvested!", "Fresh code picked!", "Logic fruit ready!"],
    },
    "moe": {
        "start": ["MoE vine deepening...", "Deep roots spreading...", "Wise vine growing..."],
        "complete": ["Wisdom melon harvested!", "Deep fruit picked!", "MoE harvest ready!"],
    },
    "thinking": {
        "start": ["Thinking vine ripening...", "Patient growth...", "Slow cultivation..."],
        "complete": ["Thought melon harvested!", "Ripened wisdom picked!", "Thoughtful fruit ready!"],
    },
}

BACKEND_MESSAGES = {
    "ollama": {"healthy": "Ollama garden flourishing!", "unhealthy": "Ollama needs attention..."},
    "llamacpp": {"healthy": "Llama happily grazing!", "unhealthy": "Llama seems tired..."},
    "gemini": {"healthy": "Gemini constellation bright!", "unhealthy": "Clouds blocking Gemini..."},
}

STARTUP_MESSAGES = [
    "Delia's garden is open!",
    "Fresh melons coming up!",
    "Watermelon patch ready!",
    "Time to grow wisdom!",
]


def get_message(event: GardenEvent) -> str:
    msgs = MESSAGES.get(event, [""])
    return random.choice(msgs)


def get_display_event(technical_event: str) -> str:
    return LOG_EVENT_DISPLAY.get(technical_event, technical_event)


def get_vine_message(vine: str, phase: str = "start") -> str:
    vine_data = VINE_MESSAGES.get(vine, VINE_MESSAGES["quick"])
    msgs = vine_data.get(phase, vine_data["start"])
    return random.choice(msgs)


def format_harvest_stats(tokens: int, elapsed_ms: int, vine: str) -> str:
    if elapsed_ms < 1000:
        speed = "lightning quick"
    elif elapsed_ms < 5000:
        speed = "nicely ripened"
    elif elapsed_ms < 15000:
        speed = "slow-grown"
    else:
        speed = "patiently cultivated"
    return f"Harvested {tokens:,} tokens ({speed}) from the {vine} vine"


def get_backend_status_message(provider: str, healthy: bool) -> str:
    status = "healthy" if healthy else "unhealthy"
    msgs = BACKEND_MESSAGES.get(provider, {"healthy": f"{provider} ready!", "unhealthy": f"{provider} resting..."})
    return msgs[status]


def get_startup_message() -> str:
    return random.choice(STARTUP_MESSAGES)


def format_queue_status(position: int, total: int) -> str:
    if position == 1:
        return "Next in line for sunshine!"
    elif position <= 3:
        return f"Seedling #{position} in nursery ({total} waiting)"
    return f"Row {position} of {total} in tray"


# =============================================================================
# GARDEN-THEMED TASK MESSAGES
# =============================================================================

TASK_MESSAGES = {
    "plant": {
        "start": [
            "Planting your seed in the garden...",
            "Finding the perfect soil for your task...",
            "Tucking your request into rich earth...",
            "A new seed enters the patch...",
        ],
        "complete": [
            "Your melon is ready!",
            "Fresh harvest from the garden!",
            "Picked ripe from the vine!",
            "The garden delivers your fruit!",
        ],
    },
    "prune": {
        "start": [
            "Examining the code vines...",
            "Inspecting for weeds and tangles...",
            "Checking the health of your branches...",
            "Looking for overgrowth...",
        ],
        "complete": [
            "Pruning complete!",
            "Garden trimmed and tidy!",
            "Weeds identified and marked!",
            "Code vines looking healthier!",
        ],
    },
    "grow": {
        "start": [
            "Cultivating fresh code...",
            "Planting logic seeds...",
            "Growing new branches...",
            "Nurturing your request into code...",
        ],
        "complete": [
            "Fresh code grown!",
            "New branches ready!",
            "Code cultivated successfully!",
            "Your code vine has flourished!",
        ],
    },
    "tend": {
        "start": [
            "Tending to the code garden...",
            "Examining roots and structure...",
            "Checking soil composition...",
            "Analyzing the garden layout...",
        ],
        "complete": [
            "Garden analysis complete!",
            "Tending report ready!",
            "Your patch has been examined!",
            "Garden insights harvested!",
        ],
    },
    "ponder": {
        "start": [
            "Letting thoughts grow slowly...",
            "Deep roots spreading...",
            "Contemplating in the shade...",
            "Wisdom vine unfurling...",
        ],
        "complete": [
            "Thoughts fully ripened!",
            "Wisdom melon ready!",
            "Deep contemplation complete!",
            "Pondering has borne fruit!",
        ],
    },
    "ruminate": {
        "start": [
            "Deep contemplation beginning...",
            "Slow growth for deep wisdom...",
            "Patient cultivation underway...",
            "Roots reaching deep for answers...",
        ],
        "complete": [
            "Deep rumination complete!",
            "Wisdom fully cultivated!",
            "Patient growth has paid off!",
            "Thoughtful harvest ready!",
        ],
    },
    "harvest": {
        "start": [
            "Gathering multiple seeds...",
            "Preparing for batch harvest...",
            "Tending multiple plots...",
            "Garden-wide cultivation starting...",
        ],
        "complete": [
            "Bountiful harvest complete!",
            "Multiple melons picked!",
            "Garden fully harvested!",
            "All fruits gathered!",
        ],
    },
}


def get_task_message(task: str, phase: str = "start") -> str:
    """Get a themed message for a garden task."""
    task_data = TASK_MESSAGES.get(task, TASK_MESSAGES["plant"])
    msgs = task_data.get(phase, task_data["start"])
    return random.choice(msgs)


def format_garden_response(content: str, task: str, vine: str, tokens: int, elapsed_ms: int) -> str:
    """Format a response with garden theming."""
    complete_msg = get_task_message(task, "complete")
    stats = format_harvest_stats(tokens, elapsed_ms, vine)
    return f"{content}\n\n---\n_{complete_msg} {stats}_"


# Map standard task names to garden names
GARDEN_TASK_MAP = {
    "delegate": "plant",
    "review": "prune",
    "generate": "grow",
    "analyze": "tend",
    "think": "ponder",
    "batch": "harvest",
    "summarize": "plant",
    "critique": "prune",
    "plan": "tend",
    "quick": "plant",
}


def get_garden_task_name(task: str) -> str:
    """Map a standard task name to its garden equivalent."""
    return GARDEN_TASK_MAP.get(task, "plant")
