# Debugging Profile

Load this profile for: investigating bugs, fixing errors, analyzing stack traces.

## Debug Order

1. **Stack trace** - Identify exact file and line number
2. **Recent commits** - What changed recently?
3. **Test output** - Which test is failing and why?
4. **Logs** - Check structured log events for context

## Fix Strategy

| Bug Size | Approach |
|----------|----------|
| Small (<10 lines) | Fix directly, commit to main |
| Medium (10-50 lines) | Create fix/ branch, add test |
| Large (>50 lines) | Create branch, document in PR |

## Common Patterns

### Null/Undefined Errors
```
# Check: Is the variable initialized?
# Check: Is the data source returning expected shape?
# Check: Are optional parameters handled?
```

### Async/Timing Issues
```
# Check: Are all awaits present?
# Check: Race conditions in parallel code?
# Check: Timeouts configured appropriately?
```

### Import/Module Errors
```
# Check: Circular imports?
# Check: Correct package installed?
# Check: Virtual environment activated?
```

## After Fixing

1. Add regression test
2. Document root cause in commit message
3. Consider if similar bugs exist elsewhere

## Best Practices

```
ALWAYS:
- Start with the stack trace
- Check recent commits for what changed
- Add regression test for every fix
- Document root cause in commit
- Verify fix in isolation before integrating

AVOID:
- Guessing at fixes without understanding the cause
- Fixing symptoms instead of root cause
- Skipping regression tests
- Making multiple unrelated changes in one fix
- Ignoring similar bugs elsewhere in codebase
```
