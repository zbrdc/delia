# Testing Profile

Load this profile for: writing tests, debugging test failures, test coverage.

## Test Commands

```bash
uv run pytest                                    # All tests
uv run pytest tests/test_file.py                 # Single file
uv run pytest tests/test_file.py::test_name -v  # Single test
DELIA_DATA_DIR=/tmp/test uv run pytest          # Isolated data
```

## Async Test Structure

```python
import pytest

@pytest.mark.asyncio
async def test_backend_failover():
    """Test graceful failover when primary backend fails."""
    primary = BackendConfig(id="primary", ...)
    secondary = BackendConfig(id="secondary", ...)

    manager = BackendManager()
    manager.backends = [primary, secondary]

    with patch_http_response(primary.url, status=500):
        result = await delegate("quick", "test prompt")

    assert result["metadata"]["backend"] == "secondary"
```

## Mock HTTP Responses

```python
def mock_ollama_response(model: str, response: str):
    def handler(request):
        return httpx.Response(200, json={
            "model": model,
            "response": response,
            "done": True,
        })
    return handler
```

## Tests Follow Implementation

- When features are refactored and working, UPDATE TESTS to match
- Old tests failing due to changed APIs are STALE, not regressions
- Fix imports/mocks/assertions for new architecture
- NEVER revert working code for outdated tests

## Test Suites

| Directory | Purpose |
|-----------|---------|
| `tests/` | Unit tests |
| `tests/comprehensive/` | Integration, stress, coverage |
| `tests/benchmarks/` | Performance benchmarks |
