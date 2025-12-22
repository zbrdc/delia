# FastAPI Profile

Load this profile for: FastAPI applications, async APIs, Pydantic validation.

## Project Structure

```
src/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app instance
│   ├── config.py            # Settings with pydantic-settings
│   ├── dependencies.py      # Dependency injection
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py
│   │   └── orders.py
│   ├── models/              # Pydantic schemas
│   │   ├── user.py
│   │   └── order.py
│   ├── services/            # Business logic
│   └── db/                  # Database layer
├── tests/
└── pyproject.toml
```

## Route Patterns

```python
from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    db: Database = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    user = await db.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        )
    return UserResponse.model_validate(user)

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreate,
    db: Database = Depends(get_db),
) -> UserResponse:
    user = await db.create_user(data)
    return UserResponse.model_validate(user)
```

## Pydantic Models

```python
from pydantic import BaseModel, Field, ConfigDict

class UserBase(BaseModel):
    email: str = Field(..., description="User email address")
    name: str = Field(..., min_length=1, max_length=100)

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserResponse(UserBase):
    id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
```

## Dependency Injection

```python
from functools import lru_cache
from fastapi import Depends

@lru_cache
def get_settings() -> Settings:
    return Settings()

async def get_db(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[Database, None]:
    db = Database(settings.database_url)
    try:
        yield db
    finally:
        await db.close()

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Database = Depends(get_db),
) -> User:
    user = await verify_token(token, db)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user
```

## Error Handling

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        self.code = code
        self.message = message
        self.status_code = status_code

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )
```

## Async Patterns

```python
# Good: Parallel database queries
async def get_dashboard(user_id: str) -> Dashboard:
    user, orders, stats = await asyncio.gather(
        db.get_user(user_id),
        db.get_orders(user_id),
        db.get_stats(user_id),
    )
    return Dashboard(user=user, orders=orders, stats=stats)

# Good: Background tasks for slow operations
from fastapi import BackgroundTasks

@router.post("/orders")
async def create_order(
    data: OrderCreate,
    background: BackgroundTasks,
) -> OrderResponse:
    order = await db.create_order(data)
    background.add_task(send_confirmation_email, order)
    return order
```

## Testing

```python
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.fixture
async def client():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac

@pytest.mark.asyncio
async def test_create_user(client: AsyncClient):
    response = await client.post("/users", json={"email": "test@example.com", "name": "Test"})
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

## Best Practices

```
ALWAYS:
- Use Pydantic models for request/response validation
- Use dependency injection for services
- Use async functions for I/O operations
- Use HTTPException with proper status codes
- Use background tasks for slow operations
- Configure CORS for production

AVOID:
- Blocking calls in async functions
- Hardcoded configuration values
- Missing response_model on routes
- Catching generic exceptions
- Using global state without proper locking
- Skipping input validation
```

