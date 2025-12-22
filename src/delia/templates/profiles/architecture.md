# Architecture Profile

Load this profile for: design decisions, patterns, refactoring, ADRs.

## Decision Records

For significant architectural decisions, create an ADR:

```
# ADR-XXX: Title

## Status
Proposed | Accepted | Deprecated

## Context
What is the issue we're addressing?

## Decision
What is the change we're proposing?

## Consequences
What becomes easier or harder?
```

## Common Patterns

### Singleton
```
# Use for: managers, registries, shared state
_instance = None

def get_manager():
    global _instance
    if _instance is None:
        _instance = Manager()
    return _instance
```

### Factory
```
# Use for: creating objects with complex initialization
def create_handler(config: Config) -> Handler:
    if config.type == "http":
        return HttpHandler(config)
    elif config.type == "grpc":
        return GrpcHandler(config)
    raise ValueError(f"Unknown type: {config.type}")
```

### Registry
```
# Use for: plugins, providers, extensibility
HANDLERS: dict[str, type[Handler]] = {}

def register(name: str):
    def decorator(cls):
        HANDLERS[name] = cls
        return cls
    return decorator
```

## Refactoring Guidelines

1. Extract when function exceeds 50 lines
2. Split when class has >5 responsibilities
3. Create interface when >2 implementations exist
4. Move to separate module when file exceeds 500 lines

## Best Practices

```
ALWAYS:
- Write ADRs for significant architectural decisions
- Use established patterns (Singleton, Factory, Registry)
- Keep modules loosely coupled
- Document public interfaces
- Consider future extensibility without over-engineering

AVOID:
- Mixing concerns in single modules
- Tight coupling between components
- Circular dependencies
- Premature abstraction
- God classes with too many responsibilities
- Undocumented architectural decisions
```
