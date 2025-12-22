# TypeScript Profile

Load this profile for: TypeScript projects, type safety, strict typing patterns.

## Type Declaration Rules

```
ALWAYS:
- Declare types for all variables and function parameters/returns
- Create custom types instead of using `any`
- Use interfaces for object shapes, types for unions/primitives
- Export one item per file (prefer named exports)

NEVER:
- Use `any` without explicit justification
- Use `as` type assertions when proper typing is possible
- Ignore TypeScript compiler errors
```

## Naming Conventions

| Element | Case | Example |
|---------|------|---------|
| Classes, Interfaces, Types | PascalCase | `UserService`, `ApiResponse` |
| Variables, Functions | camelCase | `getUserData`, `isLoading` |
| Files, Directories | kebab-case | `user-service.ts`, `api-handlers/` |
| Constants, Env Vars | UPPERCASE | `MAX_RETRIES`, `API_URL` |
| Booleans | verb prefix | `isActive`, `hasError`, `canDelete` |

## Function Guidelines

```typescript
// Good: Clear types, default parameters, early returns
function getUserById(
  id: string,
  options: GetUserOptions = {}
): User | null {
  if (!id) return null;
  // ... implementation
}

// Good: Object parameters for 3+ args (RORO pattern)
interface CreateUserParams {
  name: string;
  email: string;
  role?: UserRole;
}

function createUser(params: CreateUserParams): CreateUserResult {
  const { name, email, role = 'user' } = params;
  // ... implementation
  return { user, created: true };
}
```

## Code Style

```
- Functions: Under 20 instructions, single purpose
- Classes: Under 200 instructions, max 10 public methods
- Early returns over nested conditionals
- Prefer immutability: use `readonly` and `as const`
- Arrow functions for simple operations (<3 lines)
- Named functions for complex logic
```

## Error Handling

```typescript
// Good: Typed error handling
type Result<T> = { success: true; data: T } | { success: false; error: Error };

async function fetchUser(id: string): Promise<Result<User>> {
  try {
    const user = await api.get(`/users/${id}`);
    return { success: true, data: user };
  } catch (error) {
    return { success: false, error: error as Error };
  }
}
```

## Strict Configuration

```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true
  }
}
```

## Best Practices

```
ALWAYS:
- Enable strict mode in tsconfig.json
- Declare types for all function parameters and returns
- Use interfaces for object shapes, types for unions
- Use const assertions for literal types
- Prefer readonly for immutable properties
- Use discriminated unions for type narrowing

AVOID:
- Using `any` without explicit justification
- Type assertions without validation (as Type)
- Non-null assertions (!) when null is possible
- Implicit any from missing types
- Exporting mutable objects
- Complex type gymnastics when simple types suffice
```

