# Testing Profile

Load this profile for: writing tests, debugging test failures, E2E testing, coverage.

## Test Structure (AAA Pattern)

```python
def test_user_creation():
    """Test that creating a user returns expected result."""
    # Arrange - Set up test data
    email = "test@example.com"
    name = "Test User"

    # Act - Execute the code under test
    user = create_user(email=email, name=name)

    # Assert - Verify expectations
    assert user.email == email
    assert user.name == name
    assert user.id is not None
```

## Async Tests (Python)

```python
import pytest

@pytest.mark.asyncio
async def test_async_fetch():
    """Test async function behavior."""
    result = await fetch_users()
    assert len(result) > 0
    assert all(u.email for u in result)
```

## Mocking External Services

```python
from unittest.mock import patch, AsyncMock, MagicMock

@patch("app.services.external_api")
async def test_with_mock(mock_api):
    mock_api.get_data = AsyncMock(return_value={"status": "ok"})

    result = await process_external_data()

    mock_api.get_data.assert_called_once()
    assert result.status == "ok"
```

## Playwright E2E Testing

```typescript
import { test, expect } from '@playwright/test';

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
  });

  test('should login successfully', async ({ page }) => {
    // Use role-based locators (preferred)
    await page.getByLabel('Email').fill('user@example.com');
    await page.getByLabel('Password').fill('password123');
    await page.getByRole('button', { name: 'Sign in' }).click();

    // Web-first assertions
    await expect(page.getByText('Welcome back')).toBeVisible();
    await expect(page).toHaveURL('/dashboard');
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.getByLabel('Email').fill('wrong@example.com');
    await page.getByLabel('Password').fill('wrongpass');
    await page.getByRole('button', { name: 'Sign in' }).click();

    await expect(page.getByText('Invalid credentials')).toBeVisible();
  });
});
```

## Playwright Best Practices

```typescript
// Prefer role-based locators
page.getByRole('button', { name: 'Submit' })  // Good
page.getByLabel('Email')                       // Good
page.getByTestId('submit-btn')                 // Good (with data-testid)
page.locator('.submit-button')                 // Avoid

// Use web-first assertions (auto-wait)
await expect(element).toBeVisible()           // Good
await expect(element).toHaveText('Hello')     // Good
expect(await element.isVisible()).toBe(true)  // Avoid

// Never hardcode timeouts
await page.waitForSelector('.item')           // Avoid
await expect(page.locator('.item')).toBeVisible()  // Good
```

## Jest/Vitest Testing

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('UserService', () => {
  let service: UserService;
  let mockRepo: MockedObject<UserRepository>;

  beforeEach(() => {
    mockRepo = {
      findById: vi.fn(),
      save: vi.fn(),
    };
    service = new UserService(mockRepo);
  });

  it('should return user by id', async () => {
    const mockUser = { id: '1', name: 'Test' };
    mockRepo.findById.mockResolvedValue(mockUser);

    const result = await service.getUser('1');

    expect(result).toEqual(mockUser);
    expect(mockRepo.findById).toHaveBeenCalledWith('1');
  });

  it('should throw when user not found', async () => {
    mockRepo.findById.mockResolvedValue(null);

    await expect(service.getUser('999'))
      .rejects.toThrow('User not found');
  });
});
```

## Test Categories

| Type | Scope | Speed | When to Use |
|------|-------|-------|-------------|
| Unit | Single function/class | Fast | All business logic |
| Integration | Multiple components | Medium | API endpoints, DB |
| E2E | Full user flows | Slow | Critical paths only |

## Coverage Guidelines

```
REQUIRED:
- New code must have tests
- Bug fixes need regression tests
- Critical paths: 80%+ coverage
- Edge cases for complex logic

NOT REQUIRED:
- Generated code
- Simple getters/setters
- Third-party library wrappers
```

## Test File Organization

```
tests/
├── unit/
│   ├── services/
│   └── utils/
├── integration/
│   └── api/
├── e2e/
│   └── flows/
├── fixtures/
│   └── test_data.py
└── conftest.py
```

## Best Practices

```
ALWAYS:
- Follow AAA pattern (Arrange, Act, Assert)
- Use descriptive test names
- Test edge cases and error paths
- Mock external services
- Use role-based locators in E2E tests
- Add regression tests for bug fixes

AVOID:
- Hardcoded timeouts in tests
- Testing implementation details
- Flaky tests with race conditions
- Skipping tests without reason
- Tests that depend on execution order
- Mocking too much (test integration points)
```

