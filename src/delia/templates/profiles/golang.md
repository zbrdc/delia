# Go Profile

Load this profile for: Go applications, APIs, microservices, concurrent systems.

## Project Structure

```
project/
├── cmd/
│   └── server/
│       └── main.go          # Entry point
├── internal/
│   ├── handler/             # HTTP handlers
│   ├── service/             # Business logic
│   ├── repository/          # Data access
│   └── model/               # Domain models
├── pkg/                     # Public packages
├── api/                     # OpenAPI specs
├── go.mod
└── go.sum
```

## HTTP Handler Pattern

```go
package handler

import (
    "encoding/json"
    "net/http"
)

type UserHandler struct {
    service UserService
}

func NewUserHandler(s UserService) *UserHandler {
    return &UserHandler{service: s}
}

func (h *UserHandler) GetUser(w http.ResponseWriter, r *http.Request) {
    id := r.PathValue("id")  // Go 1.22+

    user, err := h.service.GetUser(r.Context(), id)
    if err != nil {
        http.Error(w, err.Error(), http.StatusNotFound)
        return
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(user)
}

func (h *UserHandler) CreateUser(w http.ResponseWriter, r *http.Request) {
    var req CreateUserRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, "invalid request", http.StatusBadRequest)
        return
    }

    user, err := h.service.CreateUser(r.Context(), req)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    w.WriteHeader(http.StatusCreated)
    json.NewEncoder(w).Encode(user)
}
```

## Error Handling

```go
// Define custom errors
type NotFoundError struct {
    Resource string
    ID       string
}

func (e *NotFoundError) Error() string {
    return fmt.Sprintf("%s not found: %s", e.Resource, e.ID)
}

// Wrap errors with context
func (s *UserService) GetUser(ctx context.Context, id string) (*User, error) {
    user, err := s.repo.FindByID(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("get user %s: %w", id, err)
    }
    return user, nil
}

// Check error types
if errors.Is(err, sql.ErrNoRows) {
    return nil, &NotFoundError{Resource: "user", ID: id}
}
```

## Interfaces

```go
// Define interfaces where they're used, not implemented
type UserService interface {
    GetUser(ctx context.Context, id string) (*User, error)
    CreateUser(ctx context.Context, req CreateUserRequest) (*User, error)
}

type UserRepository interface {
    FindByID(ctx context.Context, id string) (*User, error)
    Save(ctx context.Context, user *User) error
}

// Implementation
type userService struct {
    repo UserRepository
    log  *slog.Logger
}

func NewUserService(repo UserRepository, log *slog.Logger) UserService {
    return &userService{repo: repo, log: log}
}
```

## Concurrency Patterns

```go
// Worker pool
func processItems(items []Item, workers int) []Result {
    jobs := make(chan Item, len(items))
    results := make(chan Result, len(items))

    // Start workers
    var wg sync.WaitGroup
    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for item := range jobs {
                results <- process(item)
            }
        }()
    }

    // Send jobs
    for _, item := range items {
        jobs <- item
    }
    close(jobs)

    // Wait and collect
    wg.Wait()
    close(results)

    var out []Result
    for r := range results {
        out = append(out, r)
    }
    return out
}

// Context cancellation
func fetchWithTimeout(ctx context.Context, url string) (*Response, error) {
    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
    defer cancel()

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }
    return http.DefaultClient.Do(req)
}
```

## Structured Logging

```go
import "log/slog"

log := slog.New(slog.NewJSONHandler(os.Stdout, nil))

log.Info("user created",
    slog.String("user_id", user.ID),
    slog.String("email", user.Email),
)

log.Error("failed to create user",
    slog.String("error", err.Error()),
)
```

## Best Practices

```
ALWAYS:
- Accept interfaces, return structs
- Use context for cancellation/timeouts
- Handle errors explicitly
- Use defer for cleanup

AVOID:
- Naked returns in long functions
- Global state
- Ignoring errors (use _ only intentionally)
- Premature optimization
```

