# Deployment Profile

Load this profile for: CI/CD, containerization, environments, infrastructure.

## Environment Strategy

| Environment | Purpose | Data |
|-------------|---------|------|
| local | Development | Mock/seed data |
| staging | Pre-production testing | Anonymized production copy |
| production | Live users | Real data |

## CI/CD Pipeline

```yaml
# Standard pipeline stages
1. lint        # Code style checks
2. typecheck   # Static type analysis
3. test        # Unit + integration tests
4. build       # Compile/bundle
5. deploy      # Push to environment
```

## Container Best Practices

```dockerfile
# Good: Multi-stage, non-root, minimal image
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:20-alpine
USER node
COPY --from=builder /app/node_modules ./node_modules
COPY . .
CMD ["node", "dist/index.js"]
```

## Environment Variables

```
REQUIRED for deployment:
- DATABASE_URL
- API_KEYS (encrypted)
- LOG_LEVEL
- CORS_ORIGINS

NEVER hardcode:
- Secrets
- Environment-specific URLs
- Feature flags (use config service)
```

## Rollback Strategy

1. Keep previous 3 deployments available
2. Database migrations must be backward-compatible
3. Feature flags for gradual rollout
4. Automated health checks post-deploy
5. Immediate rollback if error rate spikes

## Infrastructure Checklist

Before production deployment:
- [ ] SSL/TLS configured
- [ ] Health check endpoints working
- [ ] Logging and monitoring enabled
- [ ] Backup strategy tested
- [ ] Secrets rotated from dev values
- [ ] Rate limiting configured
- [ ] CDN/caching appropriate

## Best Practices

```
ALWAYS:
- Use multi-stage Docker builds
- Run as non-root user in containers
- Keep previous deployments for rollback
- Use environment variables for config
- Test migrations are backward-compatible
- Implement health check endpoints

AVOID:
- Hardcoding secrets in code or config
- Skipping staging environment
- Deploying without health checks
- Using latest tags in production
- Ignoring error rate spikes post-deploy
- Manual deployments without CI/CD
```

