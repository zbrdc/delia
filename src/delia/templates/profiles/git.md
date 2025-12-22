# Git Profile

Load this profile for: version control, branching, commits, PRs.

## Branch Decision

| Change Type | Branch? | Naming |
|-------------|---------|--------|
| Single-file trivial fix | No, commit to main | - |
| Multi-file changes | Yes | feature/, fix/, refactor/ |
| Experimental work | Yes | experiment/ |
| Breaking changes | Yes + PR review | breaking/ |

## Commit Format

```
type(scope): description

Examples:
- feat(auth): Add OAuth2 login flow
- fix(api): Resolve timeout on large requests
- refactor(core): Extract validation to utility
- docs(readme): Update installation steps
- test(unit): Add coverage for edge cases
```

## Commit Message Rules

- Imperative mood: "Add" not "Added"
- Concise subject (<72 chars)
- Body explains "why" not "what"
- Reference issues: "Fixes #123"

## Git Safety

NEVER:
- Force-push to main/master
- Skip hooks without explicit reason
- Commit secrets or credentials
- Leave work uncommitted overnight

## PR Guidelines

- Small PRs preferred (<400 lines)
- Include test plan
- Link related issues
- Request review from relevant owners

## Best Practices

```
ALWAYS:
- Use imperative mood in commit messages
- Write descriptive commit messages explaining "why"
- Keep commits atomic and focused
- Test before pushing
- Create branches for multi-file changes

AVOID:
- Force-pushing to main/master
- Skipping hooks without explicit reason
- Committing secrets or credentials
- Leaving work uncommitted overnight
- Large PRs that are hard to review
- Vague commit messages like "fix bug"
```
