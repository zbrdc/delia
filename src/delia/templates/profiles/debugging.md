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

## Deep Debugging Methodology

These techniques find issues that surface-level debugging misses:

### 1. Verification Over Trust
Don't trust documentation or guidance - verify with counts:
```bash
grep -c "import structlog" src/*.py  # Verify claimed patterns exist
grep -c "except:" src/*.py           # Count actual occurrences
```
Zero matches reveals guidance-reality gaps.

### 2. Quantitative Validation
Replace vague terms with exact counts:
- Bad: "Found some bare except clauses"
- Good: "Found 6 bare except clauses at lines 207, 399, 147, 161, 442, 135"

Use `grep --count` or `wc -l` to turn observations into actionable facts.

### 3. Second-Order Pattern Searches
Search for patterns that reveal architectural decisions:

| Search Pattern | What It Reveals |
|----------------|-----------------|
| `global ` | Singleton coupling, hidden dependencies |
| `except:` vs `except Exception` | Error handling philosophy |
| `return_exceptions=True` | Async resilience strategy |
| `import X` vs `import Y` | Framework consistency |

### 4. Cross-Reference Guidance vs Reality
Compare documentation against implementation:
```bash
# If playbook says "use structlog"
grep "import structlog" src/*.py  # Should find matches
grep "import logging" src/*.py    # Should find fewer/none
```

### 5. Document HOW, Not Just WHAT
When finding issues, capture the methodology:
- What search patterns revealed the issue
- What verification steps confirmed it
- Add technique as playbook bullet for future sessions

## Best Practices

```
ALWAYS:
- Start with the stack trace
- Check recent commits for what changed
- Add regression test for every fix
- Document root cause in commit
- Verify fix in isolation before integrating
- VERIFY claims with grep -c before assuming patterns exist
- QUANTIFY issues with exact counts, not vague terms
- DOCUMENT methodology as playbook bullets for other models

AVOID:
- Guessing at fixes without understanding the cause
- Fixing symptoms instead of root cause
- Skipping regression tests
- Making multiple unrelated changes in one fix
- Ignoring similar bugs elsewhere in codebase
- Trusting documentation without verification
- Vague issue counts ("some", "a few", "many")
```
