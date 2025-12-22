# Performance Profile

Load this profile for: optimization, caching, profiling, latency reduction.

## Performance Hierarchy

```
1. Algorithm complexity (O(n) vs O(n^2))
2. Database queries (N+1, missing indexes)
3. Network calls (batching, caching)
4. Memory allocation (object pooling)
5. Micro-optimizations (last resort)
```

## Caching Strategy

| Cache Type | TTL | Use Case |
|------------|-----|----------|
| In-memory | Seconds | Hot data, computed values |
| Redis/Memcached | Minutes | Session, API responses |
| CDN | Hours | Static assets, public pages |
| Browser | Days | Assets with hash filenames |

## Database Optimization

```sql
-- Check for missing indexes
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'x';

-- N+1 problem: Use joins/includes
-- Bad: N+1 queries
for user in users:
    user.posts = db.query(Post).filter(user_id=user.id)

-- Good: Single query with join
users = db.query(User).options(joinedload(User.posts)).all()
```

## Async Patterns

```python
# Good: Parallel I/O
results = await asyncio.gather(
    fetch_user(user_id),
    fetch_orders(user_id),
    fetch_preferences(user_id),
)

# Bad: Sequential I/O
user = await fetch_user(user_id)
orders = await fetch_orders(user_id)
prefs = await fetch_preferences(user_id)
```

## Profiling Checklist

Before optimizing:
1. Measure current performance (baseline)
2. Identify bottleneck with profiler
3. Fix the biggest bottleneck first
4. Measure again (verify improvement)
5. Repeat until target met

## Common Anti-Patterns

```
Avoid:
- Premature optimization
- Caching without invalidation strategy
- Synchronous calls in request path
- Loading full objects when only ID needed
- String concatenation in loops
- Regex compilation in hot paths
```

## Performance Budgets

| Metric | Target |
|--------|--------|
| Time to First Byte | <200ms |
| API response p95 | <500ms |
| Database query | <50ms |
| Page load | <3s |

## Best Practices

```
ALWAYS:
- Measure before optimizing (baseline)
- Profile to identify bottlenecks
- Use parallel I/O with asyncio.gather()
- Cache expensive computations
- Add database indexes for frequent queries
- Use connection pooling

AVOID:
- Premature optimization
- Caching without invalidation strategy
- Synchronous calls in request path
- Loading full objects when only ID needed
- String concatenation in loops
- N+1 database queries
```

