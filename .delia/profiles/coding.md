# Coding Profile

Load this profile for: code generation, reviews, refactoring, implementation tasks.

## Function Signatures

```
# Good: Clear types, optional parameters with defaults
def process_item(
    item: Item,
    options: Options | None = None,
    validate: bool = True,
) -> Result:
    """Process an item with optional validation."""
```

## Error Handling

```
try:
    result = process(data)
except SpecificError as e:
    log.warning("process_failed", error=str(e), data_id=data.id)
    # Handle gracefully or re-raise with context
    raise ProcessingError(f"Failed to process {data.id}") from e
```

## Code Review Checklist

- [ ] No hardcoded secrets or credentials
- [ ] Error cases handled appropriately
- [ ] Tests cover new functionality
- [ ] No breaking changes to public API
- [ ] Performance considered for hot paths

## Best Practices

```
ALWAYS:
- Use type hints on all function signatures
- Handle error cases explicitly
- Write tests for new functionality
- Use guard clauses for early returns
- Follow existing code patterns in the project
- Document public APIs

AVOID:
- Deep nesting (max 3 levels)
- Functions longer than 50 lines
- Magic numbers without constants
- Commented-out code
- Hardcoded secrets or credentials
- Breaking changes without versioning
```
