# React Profile

Load this profile for: React applications, component patterns, hooks, state management.

## Component Architecture

```tsx
// Good: Functional component with clear props interface
interface UserCardProps {
  user: User;
  onSelect?: (user: User) => void;
  isSelected?: boolean;
}

function UserCard({ user, onSelect, isSelected = false }: UserCardProps) {
  const handleClick = useCallback(() => {
    onSelect?.(user);
  }, [user, onSelect]);

  return (
    <div
      className={cn("user-card", isSelected && "selected")}
      onClick={handleClick}
    >
      <Avatar src={user.avatar} />
      <span>{user.name}</span>
    </div>
  );
}
```

## Naming Conventions

| Element | Pattern | Example |
|---------|---------|---------|
| Components | PascalCase | `UserCard`, `AuthProvider` |
| Files | kebab-case | `user-card.tsx`, `auth-context.ts` |
| Hooks | use* prefix | `useAuth`, `useForm`, `useDebounce` |
| Handlers | handle* prefix | `handleClick`, `handleSubmit` |
| Booleans | is/has/can prefix | `isLoading`, `hasError`, `canSubmit` |

## Hooks Best Practices

```tsx
// Good: Custom hook extracting reusable logic
function useUser(userId: string) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchUser() {
      try {
        setIsLoading(true);
        const data = await api.getUser(userId);
        if (!cancelled) setUser(data);
      } catch (e) {
        if (!cancelled) setError(e as Error);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    fetchUser();
    return () => { cancelled = true; };
  }, [userId]);

  return { user, isLoading, error };
}
```

## Performance Optimization

```tsx
// Use memo for expensive list items
const UserListItem = memo(function UserListItem({ user }: Props) {
  return <li>{user.name}</li>;
});

// Use useMemo for expensive computations
const sortedUsers = useMemo(
  () => users.sort((a, b) => a.name.localeCompare(b.name)),
  [users]
);

// Use useCallback for handler props
const handleSelect = useCallback((id: string) => {
  setSelectedId(id);
}, []);
```

## State Management

```tsx
// Prefer local state when possible
const [count, setCount] = useState(0);

// Use context for cross-cutting concerns
const { user, logout } = useAuth();

// Use reducer for complex state logic
const [state, dispatch] = useReducer(formReducer, initialState);

// External stores (Redux/Zustand) for truly global state
```

## Form Handling

```tsx
// Good: Controlled form with validation
function LoginForm() {
  const form = useForm<LoginData>({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: "", password: "" },
  });

  const onSubmit = async (data: LoginData) => {
    await login(data);
  };

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <Input {...form.register("email")} />
      {form.formState.errors.email && (
        <span>{form.formState.errors.email.message}</span>
      )}
      <Button type="submit" disabled={form.formState.isSubmitting}>
        Login
      </Button>
    </form>
  );
}
```

## Accessibility

```
- Use semantic HTML (button, nav, main, article)
- Add ARIA labels for icon-only buttons
- Ensure keyboard navigation works
- Maintain focus management in modals
- Use proper heading hierarchy (h1 → h2 → h3)
```

## Best Practices

```
ALWAYS:
- Use functional components with hooks
- Memoize expensive computations (useMemo)
- Memoize callbacks passed to children (useCallback)
- Use semantic HTML for accessibility
- Handle loading and error states
- Use key prop for list items (stable IDs, not indices)

AVOID:
- Using div for clickable elements (use button)
- Creating inline objects/functions in render
- Mutating state directly
- Skip heading levels in accessibility
- Using index as key for dynamic lists
- Prop drilling more than 2-3 levels deep
```

