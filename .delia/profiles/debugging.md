# Debugging Profile

Load this profile for: error investigation, troubleshooting, performance issues.

## Structured Logging

```python
log.info(
    "llm_call_complete",
    backend=backend.id,
    model=model_name,
    latency_ms=int(elapsed * 1000),
    tokens=token_count,
    success=True,
)

log.error(
    "backend_health_check_failed",
    backend=backend.id,
    error=str(e),
    consecutive_failures=backend.consecutive_failures,
)
```

## Health Check

```bash
delia health           # CLI health check
curl localhost:8200/health  # HTTP health
```

## Common Issues

### Circuit Breaker Open
- Check `~/.delia/data/circuit_breaker.json`
- Backend has consecutive failures
- Wait for exponential backoff or restart

### Model Not Found
- Check `delia models` output
- Verify Ollama has model: `ollama list`
- Check tier configuration in `settings.json`

### Import Errors After Refactor
- Stale test imports → Update to new module paths
- Check if function was moved vs deleted
- Grep for new location: `grep -r "def function_name" src/`

## Integration Verification

```bash
# Verify old code removed
grep -r "OLD_PATTERN" src/delia/  # Should return nothing

# Verify new module used
grep -r "from.*new_module import" src/delia/

# Check for placeholder delegations (red flag)
grep -r "from.*original.*import.*as original" src/delia/
```

## Data Files

```
~/.delia/data/
├── usage_stats.json      # Model usage
├── circuit_breaker.json  # Backend failures
├── live_logs.json        # Recent activity
└── delia.db              # SQLite auth
```
