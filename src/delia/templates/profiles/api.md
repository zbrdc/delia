# API Profile

Load this profile for: REST endpoints, GraphQL, API versioning, documentation.

## REST Conventions

| Method | Purpose | Idempotent |
|--------|---------|------------|
| GET | Read resource | Yes |
| POST | Create resource | No |
| PUT | Replace resource | Yes |
| PATCH | Partial update | Yes |
| DELETE | Remove resource | Yes |

## URL Structure

```
Good:
GET  /api/v1/users          # List users
GET  /api/v1/users/123      # Get user 123
POST /api/v1/users          # Create user
PUT  /api/v1/users/123      # Update user 123

Bad:
GET  /api/getUsers          # Verb in URL
POST /api/users/create      # Redundant
GET  /api/user/123/delete   # GET for mutation
```

## Response Format

```json
{
  "data": { ... },
  "meta": {
    "page": 1,
    "total": 100
  },
  "errors": []
}
```

## Error Responses

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid email format",
    "field": "email",
    "details": {}
  }
}
```

## Status Codes

| Code | Meaning | Use When |
|------|---------|----------|
| 200 | OK | Successful GET/PUT/PATCH |
| 201 | Created | Successful POST |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Validation failed |
| 401 | Unauthorized | Not authenticated |
| 403 | Forbidden | Not authorized |
| 404 | Not Found | Resource doesn't exist |
| 429 | Too Many Requests | Rate limited |
| 500 | Internal Error | Server bug |

## Versioning Strategy

```
URL versioning (preferred):
/api/v1/users
/api/v2/users

Header versioning (alternative):
Accept: application/vnd.api+json;version=1
```

## API Documentation

- OpenAPI/Swagger for REST
- GraphQL schema for GraphQL
- Include request/response examples
- Document error codes
- Keep in sync with implementation

## Best Practices

```
ALWAYS:
- Use proper HTTP methods (GET for reads, POST for creates)
- Return appropriate status codes
- Version your APIs (/api/v1/)
- Include pagination for list endpoints
- Document all endpoints with examples
- Validate inputs before processing

AVOID:
- Verbs in URLs (use /users, not /getUsers)
- GET for mutations
- Exposing internal errors to clients
- Breaking changes without versioning
- Inconsistent response formats
- Missing rate limiting on public endpoints
```

