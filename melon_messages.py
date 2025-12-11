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
