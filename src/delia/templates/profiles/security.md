# Security Profile

Load this profile for: authentication, authorization, input validation, secrets management.

## Input Validation

```
ALWAYS validate at system boundaries:
- User input (forms, query params, headers)
- External API responses
- File uploads (type, size, content)
- Environment variables

NEVER trust:
- Client-side validation alone
- User-provided IDs without ownership check
- Deserialized data without schema validation
```

## Authentication Patterns

| Pattern | When to Use |
|---------|-------------|
| JWT | Stateless APIs, microservices |
| Session cookies | Traditional web apps, SSR |
| OAuth2 | Third-party login, API access |
| API keys | Service-to-service, CLI tools |

## Secrets Management

```
NEVER:
- Commit secrets to git
- Log sensitive data
- Store plaintext passwords
- Hardcode API keys

ALWAYS:
- Use environment variables or secret managers
- Rotate credentials regularly
- Use different secrets per environment
- Audit secret access
```

## OWASP Top 10 Checklist

Before deploying, verify protection against:
1. Injection (SQL, NoSQL, OS commands)
2. Broken authentication
3. Sensitive data exposure
4. XML external entities (XXE)
5. Broken access control
6. Security misconfiguration
7. XSS (Cross-site scripting)
8. Insecure deserialization
9. Vulnerable dependencies
10. Insufficient logging

## Authorization Patterns

```python
# Good: Check ownership at data layer
async def get_document(user_id: str, doc_id: str):
    doc = await db.get(doc_id)
    if doc.owner_id != user_id:
        raise PermissionDenied()
    return doc

# Bad: Trust client-provided ownership
async def get_document(doc_id: str):
    return await db.get(doc_id)  # No ownership check!
```

## Best Practices

```
ALWAYS:
- Validate all user input on the server
- Use parameterized queries for databases
- Hash passwords with bcrypt/argon2
- Check authorization at the data layer
- Use HTTPS everywhere
- Keep dependencies updated

AVOID:
- Storing secrets in code or environment files
- Trusting client-provided data
- Using eval() or exec() with user input
- Exposing stack traces to users
- Using deprecated crypto algorithms
- Disabling security headers
```

