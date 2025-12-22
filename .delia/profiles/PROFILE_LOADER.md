# Delia Profile Loader Guide

**Purpose:** Load appropriate instruction profiles based on context
**Status:** ACE Framework specification
**Updated:** December 21, 2025

---

## Profile Architecture

All interactions start with the **Core Profile** and load additional profiles based on detected intent/context.

### Available Profiles

| Profile | Triggers | Use When |
|---------|----------|----------|
| **core.md** | Always | Every interaction (universal rules) |
| **coding.md** | Code/implement/refactor keywords | Writing or modifying code |
| **testing.md** | Test/pytest/coverage keywords | Writing tests, debugging test failures |
| **git.md** | Branch/commit/merge/PR keywords | Git operations, version control |
| **architecture.md** | Design/ADR/pattern keywords | Architectural decisions, refactoring |
| **debugging.md** | Bug/error/fix/broken keywords | Investigating and fixing issues |

---

## How to Load Profiles (Manual Reference)

### When You Ask About Code:
**Load:** `core.md` + `coding.md`

Example triggers:
- "Implement a new orchestration mode"
- "Add a method to BackendManager"
- "Refactor the routing logic"
- "Write a function to calculate scores"

### When You Ask About Testing:
**Load:** `core.md` + `testing.md`

Example triggers:
- "Write tests for the voting executor"
- "Why is this test failing?"
- "Add coverage for the new feature"
- "Mock the HTTP responses"

### When You Ask About Git:
**Load:** `core.md` + `git.md`

Example triggers:
- "Should I create a branch for this?"
- "Help me write a commit message"
- "Create a PR for this change"
- "How should I handle this merge?"

### When You Report a Bug:
**Load:** `core.md` + `debugging.md`

Example triggers:
- "I'm getting this error: [error]"
- "The circuit breaker isn't working"
- "This backend keeps failing"
- "Why is the response empty?"

### When You Discuss Architecture:
**Load:** `core.md` + `architecture.md`

Example triggers:
- "Should we create an ADR for this?"
- "What pattern should we use here?"
- "How does the orchestration pipeline work?"
- "Design a new subsystem for X"

---

## Context Detection Keywords

### Coding Keywords
```
"implement", "add", "create", "build", "write", "modify",
"function", "class", "method", "refactor", "code"
```

### Testing Keywords
```
"test", "pytest", "coverage", "mock", "assert", "fixture",
"failing test", "test case", "unit test", "integration test"
```

### Git Keywords
```
"branch", "commit", "merge", "rebase", "git", "push", "pull",
"PR", "pull request", "conflict", "remote", "checkout"
```

### Debugging Keywords
```
"bug", "error", "fix", "debug", "issue", "failing", "broken",
"stack trace", "exception", "traceback", "not working"
```

### Architecture Keywords
```
"design", "ADR", "architecture", "pattern", "singleton",
"refactor", "structure", "module", "dependency", "interface"
```

---

## Using Multiple Profiles in One Session

You may load different profiles within a single conversation as context changes:

**Example Session:**
1. Ask about architecture -> `architecture.md` loaded
2. Start implementing feature -> `coding.md` loaded
3. Hit an error during impl -> `debugging.md` loaded
4. Ready to commit -> `git.md` loaded

**The core profile remains loaded throughout.**

---

## For AI Assistants (Implementation Guide)

### At Conversation Start
```
1. Acknowledge interaction
2. Detect context from user message
3. Load core.md
4. Load appropriate context profile(s)
5. Declare loaded profiles in response header
6. Reference profiles when making decisions
```

### In Subsequent Messages
```
1. Track context shift
2. If context changed, load new profile
3. Continue conversation
4. On major context shift: note profile change
```

### Example Response Header
```
## Profile Context
**Loaded:** core.md + coding.md
**Task:** Implement new voting mode with confidence weighting

---

[Your response referencing these profiles]

**See also:** orchestration/executor.py for existing patterns
```

---

## Profile Synchronization

Profiles are maintained in a single location:
- Profiles live in `.delia/profiles/` directory
- All AI assistants reference this location
- Single source of truth
- Changes apply immediately

The main instruction files (CLAUDE.md, .gemini/instructions.md, etc.) contain:
1. The ACE Framework steps (mandatory)
2. Inlined critical rules (for quick reference)
3. Pointers to full profiles

---

## Updating Profiles

When profiles need updates:

1. Edit the specific profile file in `.delia/profiles/`
2. Update any inlined rules in CLAUDE.md if they changed
3. Commit with: `git commit -m "docs(profiles): Update [profile-name] for X"`
4. Sync to other instruction files if needed

---

**This file is the loader specification. Read it to understand which profiles apply to your task.**
