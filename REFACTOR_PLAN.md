# Delia Full Refactoring Plan

## Goal
Reduce `mcp_server.py` from 4323 lines to under 2000 lines while eliminating global state coupling.

## Current State Analysis

| Section | Lines | Description |
|---------|-------|-------------|
| Imports & Early Config | 1-199 | Imports, auth conditional setup |
| Live Logs & Logging | 200-400 | Dashboard processor, structlog config |
| Usage Tracking | 375-670 | MODEL_USAGE, TASK_STATS, save functions |
| Language Detection | 711-910 | LANGUAGE_CONFIGS, detect_language |
| System Prompts | 911-1100 | get_system_prompt, optimize_prompt |
| Model Selection | 1228-1398 | select_model with complex routing |
| Provider Calls | 1399-1970 | call_ollama (200), call_llamacpp (217), call_gemini (139) |
| Auth Routes | 2461-2920 | OAuth routes (326 lines) |
| Main Tools | 3140-3700 | delegate, think, batch, health, models |

---

## Phase 1: Extract Provider Layer (~700 lines)

Create proper provider classes with dependency injection.

**New Files:**
- `providers/base.py` - Abstract LLMProvider base class
- `providers/ollama.py` - OllamaProvider class
- `providers/llamacpp.py` - LlamaCppProvider class
- `providers/gemini.py` - GeminiProvider class
- `providers/registry.py` - ProviderRegistry to replace call_llm switch

**Pattern:** Provider Registry eliminates switch statement:
```python
class ProviderRegistry:
    def __init__(self, backend_manager, config, stats_service):
        self._providers = {
            "ollama": OllamaProvider(config, stats_service),
            "llamacpp": LlamaCppProvider(config, stats_service),
            "gemini": GeminiProvider(config, stats_service),
        }

    async def call(self, backend: BackendConfig, ...) -> dict:
        return await self._providers[backend.provider].call(backend, ...)
```

---

## Phase 2: Extract Stats Service (~200 lines)

Replace global dicts with proper service class.

**New File:** `stats.py`

```python
class StatsService:
    def __init__(self, data_dir: Path):
        self.model_usage = {...}
        self.task_stats = {...}
        self._lock = threading.Lock()

    def record_call(self, model_tier, task_type, tokens, elapsed_ms):
        """Thread-safe recording (replaces _update_stats_sync)"""

    async def save_all(self):
        """Persist to disk (replaces save_all_stats_async)"""
```

**Eliminates globals:** MODEL_USAGE, TASK_STATS, RECENT_CALLS, RESPONSE_TIMES

---

## Phase 3: Extract Logging Service (~200 lines)

**New File:** `logging_service.py`

```python
class LoggingService:
    def __init__(self, data_dir: Path):
        self.live_logs = []
        self._lock = threading.Lock()

    def configure_structlog(self, use_stderr: bool):
        """Configure structlog with dashboard processor"""

    def dashboard_processor(self, logger, method, event_dict):
        """Capture logs for dashboard"""
```

**Eliminates globals:** LIVE_LOGS, _live_logs_lock

---

## Phase 4: Extract Language Detection (~200 lines)

**New File:** `language.py`

```python
class LanguageDetector:
    LANGUAGE_CONFIGS = {...}  # ~100 lines of config
    PYGMENTS_MAP = {...}

    def detect(self, content: str, file_path: str = "") -> str:
        """Detect programming language"""

    def get_system_prompt(self, language: str, task_type: str) -> str:
        """Get optimized system prompt"""
```

---

## Phase 5: Expand Routing Module (~200 lines)

Enhance existing `routing.py` to include model selection.

```python
class ModelRouter:
    def __init__(self, config, backend_manager):
        ...

    async def select_model(self, task_type, content_size, override, content) -> str:
        """Replaces select_model function"""

    def parse_model_override(self, override: str) -> str:
        """Parse natural language hints"""

    async def select_optimal_backend(self, content, file_path, task_type, backend_type):
        """Replaces _select_optimal_backend_v2"""
```

---

## Phase 6: Extract Auth Routes (~350 lines)

**New File:** `auth_routes.py`

```python
def register_auth_routes(mcp: FastMCP, tracker: UserTracker):
    """Move entire _register_auth_routes function"""
```

---

## Phase 7: Extract Tool Implementations (~400 lines)

**New Files:**
- `tools/delegate.py` - DelegateTool class with all delegate helpers
- `tools/admin.py` - health, models, switch_backend, switch_model
- `tools/resources.py` - MCP resources

---

## Phase 8: Dependency Injection Container

**New File:** `container.py`

```python
class ServiceContainer:
    def __init__(self):
        self.config = Config()
        self.backend_manager = BackendManager()
        self.stats_service = StatsService(paths.DATA_DIR)
        self.logging_service = LoggingService(paths.DATA_DIR)
        self.language_detector = LanguageDetector()
        self.model_router = ModelRouter(self.config, self.backend_manager)
        self.provider_registry = ProviderRegistry(...)
        self.model_queue = ModelQueue()

_container: ServiceContainer | None = None

def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container
```

---

## Extraction Order

Execute in dependency order:

1. **Logging Service** - No dependencies
2. **Stats Service** - Only depends on paths
3. **Language Detection** - No dependencies
4. **Provider Layer** - Depends on stats_service, config
5. **Model Selection** - Depends on config, backend_manager
6. **Auth Routes** - Depends on auth module
7. **MCP Tools** - Depends on all above
8. **Container + Consolidate** - Wire everything together

---

## Estimated Results

| Module | Lines Extracted |
|--------|-----------------|
| providers/ | ~700 |
| stats.py | ~200 |
| logging_service.py | ~200 |
| language.py | ~200 |
| routing.py (expanded) | ~200 |
| auth_routes.py | ~350 |
| tools/ | ~400 |
| **Total** | **~2250** |

**Final mcp_server.py:** ~2073 lines (goal: <2000)

---

## Backwards Compatibility

Keep re-exports in mcp_server.py:
```python
# For backwards compatibility
from .stats import StatsService
from .language import detect_language
from .providers import call_llm, call_ollama
```

---

## Test Strategy

- Run `uv run pytest` after each phase
- Create feature branch per phase
- Add new tests for extracted modules:
  - test_providers.py
  - test_language.py
  - test_logging_service.py
  - test_container.py
