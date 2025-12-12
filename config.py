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
"""
Delia Configuration

All user-configurable values are defined here for easy customization.
Values have been validated with Wolfram Alpha for mathematical accuracy.
"""
import os
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field

import paths


@dataclass(frozen=True)
class ModelConfig:
    """Model tier configuration."""
    name: str
    ollama_model: str
    vram_gb: float
    context_tokens: int
    num_ctx: int  # Request context size (conservative for quality)
    max_input_kb: int  # Practical input limit for good output quality

    @property
    def max_input_bytes(self) -> int:
        return self.max_input_kb * 1024


@dataclass
class Config:
    """Main configuration for Delia."""

    # ============================================================
    # BACKEND SELECTION (Legacy - used only as fallback)
    # Options: "ollama", "llamacpp"
    # Default is "ollama" (recommended), can be overridden via DELIA_BACKEND
    # ============================================================
    backend: str = field(
        default_factory=lambda: os.getenv("DELIA_BACKEND", "ollama")
    )

    # ============================================================
    # OLLAMA CONNECTION
    # ============================================================
    ollama_base: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE", "http://localhost:11434")
    )
    ollama_timeout_seconds: float = 300.0
    ollama_connect_timeout: float = 10.0
    # Backend type: "local" or "remote" - determines how it's referred to
    ollama_type: str = field(
        default_factory=lambda: os.getenv("OLLAMA_TYPE", "local")
    )

    # ============================================================
    # LLAMA.CPP CONNECTION (OpenAI-compatible API)
    # ============================================================
    llamacpp_base: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_BASE", "http://localhost:8080")
    )
    llamacpp_timeout_seconds: float = 300.0
    llamacpp_connect_timeout: float = 10.0
    # Default model name for llama.cpp (used when no specific model is loaded)
    llamacpp_model: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_MODEL", "Qwen3-14B-Q4_K_M.gguf")
    )
    # Context size limit for llama.cpp (in tokens) - used for context-aware routing
    # Set via LLAMACPP_CTX_SIZE env var or adjust based on your llama-server -c setting
    llamacpp_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLAMACPP_CTX_SIZE", "8192"))
    )
    # Backend type: "local" or "remote" - determines how it's referred to
    llamacpp_type: str = field(
        default_factory=lambda: os.getenv("LLAMACPP_TYPE", "remote")
    )

    # ============================================================
    # MODEL CONFIGURATION
    # Validated: 40K tokens @ 4 chars/token @ 75% util = 117KB theoretical
    # We use 50KB practical limit for quality output headroom
    # ============================================================
    model_quick: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="quick",
        ollama_model="qwen3:14b",
        vram_gb=9.0,
        context_tokens=40_000,
        num_ctx=8192,  # 8192 * 4 / 1024 = 32KB request context
        max_input_kb=50,  # 50KB ≈ 12,500 tokens, leaves room for output
    ))

    model_coder: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="coder",
        ollama_model="qwen2.5-coder:14b",
        vram_gb=9.0,
        context_tokens=128_000,
        num_ctx=16384,  # 16384 * 4 / 1024 = 64KB request context
        max_input_kb=100,  # 100KB ≈ 25,000 tokens
    ))

    model_moe: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="moe",
        ollama_model="qwen3:30b-a3b",
        vram_gb=17.0,
        context_tokens=128_000,
        num_ctx=16384,
        max_input_kb=100,
    ))

    # Dedicated thinking/reasoning model (uses chain-of-thought)
    model_thinking: ModelConfig = field(default_factory=lambda: ModelConfig(
        name="thinking",
        ollama_model=os.getenv("THINKING_MODEL", "olmo3:7b-think"),  # or "qwen3-coder:30b" for deeper reasoning
        vram_gb=9.0,  # Adjust based on your chosen model
        context_tokens=128_000,
        num_ctx=16384,
        max_input_kb=100,
    ))

    # ============================================================
    # MODEL SELECTION THRESHOLDS
    # Validated: 50KB = 12,500 tokens (well within 40K context of 14B)
    # ============================================================

    # Content size threshold for upgrading to coder/moe (bytes)
    large_content_threshold: int = 50_000  # 50KB → 12,500 tokens

    # Medium content threshold (used when quick is loaded to avoid unnecessary upgrade)
    medium_content_threshold: int = 30_000  # 30KB → 7,500 tokens

    # Tasks that require MoE model (complex multi-step reasoning)
    moe_tasks: frozenset = field(
        default_factory=lambda: frozenset({"plan", "critique"})
    )

    # Tasks that benefit from coder model
    coder_tasks: frozenset = field(
        default_factory=lambda: frozenset({"generate", "review", "analyze"})
    )

    # Tasks that enable thinking mode
    thinking_tasks: frozenset = field(
        default_factory=lambda: frozenset({"plan", "analyze", "critique"})
    )

    # ============================================================
    # GENERATION PARAMETERS
    # ============================================================

    # Temperature for normal generation (lower = more deterministic)
    temperature_normal: float = 0.3

    # Temperature when thinking mode is enabled
    temperature_thinking: float = 0.6

    # ============================================================
    # COST ESTIMATION
    # Validated: GPT-4 averages $0.03 input + $0.06 output = $0.045/1K blended
    # Using conservative $0.03/1K for comparison (input-heavy workloads)
    # ============================================================
    gpt4_cost_per_1k_tokens: float = 0.03

    # ============================================================
    # FILE HANDLING
    # ============================================================
    max_file_size: int = 500_000  # 500KB max file read

    # ============================================================
    # PERSISTENCE
    # ============================================================
    stats_file: Path = field(
        default_factory=lambda: paths.STATS_FILE
    )

    # ============================================================
    # AUTHENTICATION & MULTI-USER
    # ============================================================

    # Enable/disable authentication (set via DELIA_AUTH_ENABLED env var)
    # When disabled, all auth endpoints are hidden and tracking uses session IDs
    auth_enabled: bool = field(
        default_factory=lambda: os.getenv("DELIA_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
    )

    # Enable/disable user tracking (works with or without auth)
    tracking_enabled: bool = field(
        default_factory=lambda: os.getenv("DELIA_TRACKING_ENABLED", "true").lower() in ("true", "1", "yes")
    )

    # ============================================================
    # CONCURRENCY CONTROL
    # ============================================================

    # Maximum concurrent LLM requests per backend (prevents GPU memory exhaustion)
    # Set to 0 for unlimited (let Ollama/llama.cpp manage their own queue)
    max_concurrent_requests_per_backend: int = field(
        default_factory=lambda: int(os.getenv("DELIA_MAX_CONCURRENT", "0"))
    )

    # ============================================================
    # BACKEND TYPE HELPERS
    # ============================================================

    def get_backend_type(self, backend: str) -> str:
        """Get the type ('local' or 'remote') for a backend."""
        if backend == "ollama":
            return self.ollama_type
        elif backend == "llamacpp":
            return self.llamacpp_type
        return "local"  # Default fallback

    def get_local_backend(self) -> str | None:
        """Get the name of the backend configured as 'local', or None if none."""
        if self.ollama_type == "local":
            return "ollama"
        elif self.llamacpp_type == "local":
            return "llamacpp"
        return None

    def get_remote_backend(self) -> str | None:
        """Get the name of the backend configured as 'remote', or None if none."""
        if self.ollama_type == "remote":
            return "ollama"
        elif self.llamacpp_type == "remote":
            return "llamacpp"
        return None

    def get_preferred_backend_for_type(self, backend_type: str) -> str:
        """Get the preferred backend name for a given type ('local' or 'remote').

        Returns the configured backend or falls back to default.
        """
        if backend_type == "local":
            return self.get_local_backend() or "ollama"
        elif backend_type == "remote":
            return self.get_remote_backend() or "llamacpp"
        return "ollama"

    def is_backend_local(self, backend: str) -> bool:
        """Check if a backend is configured as local."""
        return self.get_backend_type(backend) == "local"

    def is_backend_remote(self, backend: str) -> bool:
        """Check if a backend is configured as remote."""
        return self.get_backend_type(backend) == "remote"


# Global config instance
config = Config()


# ============================================================
# BACKEND HEALTH & CIRCUIT BREAKER
# Tracks failures and implements adaptive routing
# ============================================================

@dataclass
class BackendHealth:
    """
    Tracks backend health and implements circuit breaker pattern.

    Circuit Breaker States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Too many failures, requests are blocked
    - HALF-OPEN: Testing if backend recovered (auto after cooldown)

    Context Size Learning:
    - Tracks successful/failed context sizes
    - Recommends safe context sizes based on history
    """
    name: str

    # Failure tracking
    consecutive_failures: int = 0
    last_failure_time: float = 0.0  # Unix timestamp
    last_error_type: str = ""

    # Context size learning
    last_failed_context_size: int = 0
    safe_context_estimate: int = 100_000  # Start optimistic (100KB)
    max_successful_context: int = 0

    # Circuit breaker state
    circuit_open_until: float = 0.0  # Unix timestamp

    # Configuration
    failure_threshold: int = 3  # Failures before opening circuit
    base_cooldown_seconds: float = 30.0  # Base cooldown time
    max_cooldown_seconds: float = 300.0  # Max 5 min cooldown
    context_reduction_factor: float = 0.7  # Reduce context by 30% after timeout

    def record_failure(self, error_type: str, context_size: int = 0) -> None:
        """Record a backend failure and potentially open circuit."""
        import time
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.last_error_type = error_type

        # Learn from timeout failures
        if error_type == "timeout" and context_size > 0:
            self.last_failed_context_size = context_size
            # Reduce safe estimate to 70% of failed size
            new_estimate = int(context_size * self.context_reduction_factor)
            self.safe_context_estimate = min(self.safe_context_estimate, new_estimate)

        # Open circuit if threshold exceeded
        if self.consecutive_failures >= self.failure_threshold:
            # Exponential backoff: 30s, 60s, 120s, 240s, 300s (max)
            cooldown = min(
                self.base_cooldown_seconds * (2 ** (self.consecutive_failures - self.failure_threshold)),
                self.max_cooldown_seconds
            )
            self.circuit_open_until = time.time() + cooldown

    def record_success(self, context_size: int = 0) -> None:
        """Record a successful call and reset circuit."""
        self.consecutive_failures = 0
        self.circuit_open_until = 0.0
        self.last_error_type = ""

        # Learn successful context sizes
        if context_size > 0:
            self.max_successful_context = max(self.max_successful_context, context_size)
            # Slowly increase safe estimate on success (conservative)
            if context_size > self.safe_context_estimate * 0.9:
                self.safe_context_estimate = min(
                    int(context_size * 1.1),  # Allow 10% growth
                    100_000  # Cap at 100KB
                )

    def is_available(self) -> bool:
        """Check if backend is available (circuit not open)."""
        import time
        if self.circuit_open_until > 0 and time.time() < self.circuit_open_until:
            return False
        return True

    def time_until_available(self) -> float:
        """Seconds until circuit closes (0 if available)."""
        import time
        if not self.is_available():
            return max(0, self.circuit_open_until - time.time())
        return 0.0

    def should_reduce_context(self, proposed_size: int) -> tuple[bool, int]:
        """
        Check if context should be reduced based on failure history.

        Returns:
            (should_reduce, recommended_size)
        """
        # If we have recent failures and proposed size exceeds safe estimate
        if self.consecutive_failures > 0 and proposed_size > self.safe_context_estimate:
            recommended = int(self.safe_context_estimate * 0.9)  # 10% safety margin
            return True, recommended
        return False, proposed_size

    def get_status(self) -> dict:
        """Get current health status as dict."""
        import time
        return {
            "available": self.is_available(),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error_type,
            "circuit_open": self.circuit_open_until > time.time(),
            "seconds_until_available": round(self.time_until_available(), 1),
            "safe_context_kb": self.safe_context_estimate // 1024,
        }


# Global health trackers for each backend (dynamic)
BACKEND_HEALTH: dict[str, BackendHealth] = {}


def get_backend_health(backend: str) -> BackendHealth:
    """Get or create health tracker for a backend."""
    if backend not in BACKEND_HEALTH:
        BACKEND_HEALTH[backend] = BackendHealth(name=backend)
    return BACKEND_HEALTH[backend]


# ============================================================
# DERIVED VALUES (for backwards compatibility)
# ============================================================

# Load models from backends.json if available, otherwise use config defaults
# DEPRECATED: Now handled by backend_manager.py and settings.json

STATS_FILE = config.stats_file


# ============================================================
# DYNAMIC MODEL DETECTION
# HuggingFace-informed model parsing for natural language model names
# Supports: Qwen, Llama, Mistral, DeepSeek, Phi, Gemma, Falcon, Yi, GPT variants
# ============================================================

import re
from typing import NamedTuple, Optional

# ============================================================
# REGEX PATTERNS FOR MODEL NAME PARSING
# ============================================================

# Extract parameter count: "14B", "7b", "72B", "0.5B", "1.5b"
_PARAM_REGEX = re.compile(r'(\d+(?:\.\d+)?)\s*[bB](?:illion)?(?![a-zA-Z])', re.IGNORECASE)

# Extract version numbers: "v0.1", "2.5", "3.1", "-v3"
_VERSION_REGEX = re.compile(r'(?:v?(\d+(?:\.\d+)*))', re.IGNORECASE)

# Detect MoE (Mixture of Experts) patterns: "-a3b", "MoE", "mixtral"
_MOE_REGEX = re.compile(r'(-a\d+b|moe|mixtral)', re.IGNORECASE)

# ============================================================
# MODEL FAMILY DETECTION (HuggingFace top models)
# ============================================================

MODEL_FAMILIES = {
    # Family name -> (aliases, is_primarily_coder)
    "qwen": (["qwen", "qwen2", "qwen2.5", "qwen3"], False),
    "llama": (["llama", "llama2", "llama3", "llama-3", "meta-llama", "codellama"], False),
    "mistral": (["mistral", "mixtral", "mistral-nemo"], False),
    "deepseek": (["deepseek", "deepseek-v2", "deepseek-v3", "deepseek-coder"], False),
    "phi": (["phi", "phi-2", "phi-3", "phi-4", "phi2", "phi3", "phi4"], False),
    "gemma": (["gemma", "gemma2", "gemma-2", "codegemma"], False),
    "falcon": (["falcon", "falcon2"], False),
    "yi": (["yi", "yi-coder", "yi-1.5"], False),
    "gpt": (["gpt2", "gpt-j", "gpt-neox", "gpt-neo"], False),
    "starcoder": (["starcoder", "starcoder2", "starcoderbase"], True),
    "codellama": (["codellama", "code-llama"], True),
    "codegemma": (["codegemma", "code-gemma"], True),
    "nemotron": (["nemotron"], False),
    "hermes": (["hermes", "nous-hermes"], False),
    "orca": (["orca", "orca2"], False),
    "wizardcoder": (["wizardcoder", "wizard-coder"], True),
    "magicoder": (["magicoder"], True),
}

# ============================================================
# VARIANT DETECTION
# ============================================================

# Coder-specialized keywords
CODER_KEYWORDS = frozenset([
    "coder", "code", "codellama", "starcoder", "codegemma",
    "wizardcoder", "magicoder", "deepseek-coder", "yi-coder",
    "codestral", "granite-code", "stable-code"
])

# Instruction-tuned keywords
INSTRUCT_KEYWORDS = frozenset([
    "instruct", "chat", "-it", "assistant", "rlhf"
])

# Base model keywords
BASE_KEYWORDS = frozenset([
    "base", "foundation", "pretrain", "raw"
])


class ModelInfo(NamedTuple):
    """Parsed model information."""
    params_b: float           # Parameter count in billions (0 if unknown)
    family: Optional[str]     # Model family (qwen, llama, etc.)
    is_coder: bool            # Is this a code-specialized model?
    is_moe: bool              # Is this a Mixture of Experts model?
    is_instruct: bool         # Is this instruction-tuned?
    raw_name: str             # Original model name


@lru_cache(maxsize=256)
def parse_model_name(name: str) -> ModelInfo:
    """
    Parse a model name to extract characteristics.

    Cached to avoid repeated regex parsing for the same model names.

    Examples:
        "qwen2.5-coder:14b" → (14.0, "qwen", True, False, False, ...)
        "llama-3.1-70b-instruct" → (70.0, "llama", False, False, True, ...)
        "mixtral-8x7b" → (56.0, "mistral", False, True, False, ...)
        "deepseek-v3" → (0.0, "deepseek", False, False, False, ...)
    """
    name_lower = name.lower()

    # Extract parameter count
    param_match = _PARAM_REGEX.search(name)
    params = float(param_match.group(1)) if param_match else 0.0

    # Handle MoE notation like "8x7b" → 56B total
    moe_mult_match = re.search(r'(\d+)x(\d+)b', name_lower)
    if moe_mult_match:
        experts = int(moe_mult_match.group(1))
        per_expert = int(moe_mult_match.group(2))
        params = float(experts * per_expert)

    # Detect model family
    family = None
    family_is_coder = False
    for fam_name, (aliases, is_coder_family) in MODEL_FAMILIES.items():
        if any(alias in name_lower for alias in aliases):
            family = fam_name
            family_is_coder = is_coder_family
            break

    # Detect coder variant
    is_coder = family_is_coder or any(kw in name_lower for kw in CODER_KEYWORDS)

    # Detect MoE variant
    is_moe = bool(_MOE_REGEX.search(name))

    # Detect instruction-tuning
    is_instruct = any(kw in name_lower for kw in INSTRUCT_KEYWORDS)

    return ModelInfo(
        params_b=params,
        family=family,
        is_coder=is_coder,
        is_moe=is_moe,
        is_instruct=is_instruct,
        raw_name=name
    )


@lru_cache(maxsize=256)
def _detect_model_tier_cached(model_name: str) -> str:
    """
    Cached tier detection based purely on model name parsing.

    This is the core logic - separated to enable caching.
    """
    info = parse_model_name(model_name)

    # MoE models get moe tier (complex reasoning capability)
    if info.is_moe:
        return "moe"

    # Large models (≥30B) get moe tier
    if info.params_b >= 30:
        return "moe"

    # Coder-specialized models get coder tier
    if info.is_coder:
        return "coder"

    # Medium models (15-30B) can handle complex tasks
    if 15 <= info.params_b < 30:
        return "coder"

    # Small models or unknown → quick tier
    return "quick"


def detect_model_tier(model_name: str, known_models: Optional[dict[str, str]] = None) -> str:
    """
    Detect which tier a model belongs to based on parsed characteristics.

    Tier Assignment Strategy:
    1. Exact match against configured tier models (if provided)
    2. MoE or large params (≥30B) → moe tier
    3. Coder-specialized → coder tier
    4. Medium params (15-30B) → coder tier (capable enough)
    5. Small params (<15B) or unknown → quick tier

    Returns 'quick', 'coder', or 'moe'.
    """
    # Priority 1: Exact match against configured models (not cached - dict not hashable)
    if known_models:
        model_lower = model_name.lower()
        for tier, tier_model in known_models.items():
            tier_normalized = tier_model.lower().replace(":latest", "")
            model_normalized = model_lower.replace(":latest", "")
            if tier_normalized == model_normalized or tier_normalized in model_normalized:
                return tier

    # Priority 2: Use cached tier detection based on model parsing
    return _detect_model_tier_cached(model_name)

