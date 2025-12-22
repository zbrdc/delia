# Git Workflow Profile

Load this profile for: commits, branches, PRs, version control.

## When to Create Branches

- **Feature branches**: New features, refactors, multi-file changes
  - `feature/add-voting-confidence`, `refactor/extract-tools`
- **Fix branches**: Bug fixes (`fix/circuit-breaker-timeout`)
- **Stay on main**: Single-file docs, typos, trivial changes
- **Ask if uncertain**: Unclear scope → ask before branching

## When to Commit

- **Atomic**: Each commit compiles and passes tests
- **After validation**: Verify change works before committing
- **Separate concerns**: Don't batch unrelated changes

## Commit Messages

```bash
feat: Add confidence-weighted voting per ADR-008
fix: Resolve circuit breaker false positives
refactor: Extract tool handlers to dedicated module
docs: Update orchestration modes for ADR-008
test: Update stale tests for new delegation API
chore: Update dependencies
```

## When to Push

- After stable checkpoint (feature complete, tested)
- Before context switch (stepping away)
- NEVER push broken code
- NEVER force push main/master

## Pull Requests

- PR from feature branch → main
- Draft PR for work-in-progress
- Include: summary, test plan, ADR links
- Small PRs preferred

## What NOT to Commit

- Secrets (`.env`, API keys)
- Generated files (`__pycache__`, builds)
- Local config (`.delia.local.md`)
- Incomplete refactors
