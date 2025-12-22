# Python Profile

Load this profile for: Python projects, async patterns, type hints, PEP compliance.

## Core Principles

```
ALWAYS:
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Prefer functional programming over classes
- Use descriptive names with auxiliary verbs (is_active, has_permission)

NEVER:
- Use mutable default arguments
- Catch bare exceptions (except Exception:)
- Use wildcard imports (from x import *)
```

## File Organization

```
project/
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── main.py
│       ├── models/           # Pydantic models
│       ├── services/         # Business logic
│       └── utils/            # Helper functions
├── tests/
├── pyproject.toml
└── README.md
```

## Function Patterns

```python
# Good: Type hints, guard clauses, early returns
async def get_user(
    user_id: str,
    include_profile: bool = False
) -> User | None:
    if not user_id:
        return None

    user = await db.fetch_user(user_id)
    if not user:
        return None

    if include_profile:
        user.profile = await db.fetch_profile(user_id)

    return user
```

## Async Patterns

```python
# Good: Parallel I/O with gather
async def fetch_dashboard_data(user_id: str) -> DashboardData:
    user, orders, prefs = await asyncio.gather(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_preferences(user_id),
    )
    return DashboardData(user=user, orders=orders, prefs=prefs)

# Use def for sync, async def for I/O-bound
def calculate_total(items: list[Item]) -> Decimal:  # CPU-bound
    return sum(item.price for item in items)

async def save_order(order: Order) -> None:  # I/O-bound
    await db.insert(order)
```

## Data Validation (Pydantic)

```python
from pydantic import BaseModel, Field, field_validator

class UserCreate(BaseModel):
    email: str = Field(..., description="User email")
    name: str = Field(..., min_length=1, max_length=100)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v.lower()
```

## Error Handling

```python
# Good: Specific exceptions, context managers
from contextlib import asynccontextmanager

class UserNotFoundError(Exception):
    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f"User not found: {user_id}")

@asynccontextmanager
async def database_transaction():
    async with db.transaction() as tx:
        try:
            yield tx
            await tx.commit()
        except Exception:
            await tx.rollback()
            raise
```

## Logging

```python
import structlog

log = structlog.get_logger()

# Good: Structured logging with context
log.info("user_created", user_id=user.id, email=user.email)
log.error("payment_failed", order_id=order.id, error=str(e))
```

## Package Management (uv)

```bash
# Use uv for all dependency management
uv sync              # Install from lockfile
uv add httpx         # Add dependency
uv add --dev pytest  # Add dev dependency
uv run pytest        # Run in project environment
```

## Best Practices

```
ALWAYS:
- Use type hints on all function signatures
- Use async/await for I/O operations
- Use Pydantic for data validation
- Use structured logging (structlog)
- Follow PEP 8 style guidelines
- Use guard clauses for early returns

AVOID:
- Bare except clauses (use specific exceptions)
- Mutable default arguments (def f(items=[]))
- Global mutable state
- String formatting with % or .format() (use f-strings)
- Blocking calls in async functions
- Import * from modules
```

