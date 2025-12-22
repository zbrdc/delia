# Core Profile (Always Loaded)

Universal rules that apply to ALL tasks in this project.

## Library-First Principle (CRITICAL)

**BEFORE implementing any feature, ALWAYS evaluate existing solutions:**

```
1. SEARCH for libraries/tools that solve the problem
2. EVALUATE top 2-3 options (maturity, maintenance, fit)
3. RECOMMEND the best option to user with rationale
4. ONLY implement from scratch if no good solution exists

Questions to ask:
- Is there a well-maintained library for this?
- Does the project already have a dependency that handles this?
- Would a library save significant effort vs custom code?
```

This prevents reinventing wheels and ensures we leverage community solutions.

## Code Standards

- Use type hints on all function signatures
- Prefer async/await for I/O operations
- Use structured logging: `log.info("event_name", key=value)`
- Validate inputs early, fail fast with clear errors

## Before Writing Code

```
CHECKLIST (Cannot skip):
[] Search codebase for similar patterns (DRY)
[] Check existing utilities before creating new ones
[] Verify function signatures match project style
[] Run tests before committing
```

## Anti-Patterns (NEVER DO)

- Placeholder implementations that defer to original code
- Duplicate state across old and new modules
- Hardcoded values that should be configuration
- Skip tests before pushing

## Documentation

- Add docstrings to public functions
- Keep comments focused on "why" not "what"
- Update README when adding major features

## Complete Integration Rule

When extracting/refactoring code:
1. REMOVE old code from source module
2. Verify old patterns are deleted, not bypassed
3. Test that new integration is actually used

## Best Practices

```
ALWAYS:
- Use type hints on all function signatures
- Prefer async/await for I/O operations
- Use structured logging with context
- Validate inputs early, fail fast
- Search codebase for similar patterns before creating new ones
- Run tests before committing

AVOID:
- Placeholder implementations that defer to original code
- Duplicate state across old and new modules
- Hardcoded values that should be configuration
- Skipping tests before pushing
- Deep nesting (max 3 levels)
- Magic numbers without constants
```
