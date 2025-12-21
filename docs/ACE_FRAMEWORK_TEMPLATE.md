# ACE Framework Template

**Version:** 1.0
**Purpose:** Reusable template for implementing the Adaptive Context Engine (ACE) Framework in any project
**Usage:** Feed this template to an AI agent along with project context to generate a customized ACE implementation

---

## What is ACE Framework?

ACE (Adaptive Context Engine) is a profile-based instruction system that ensures AI assistants:
1. Load appropriate context before responding
2. Declare what context they've loaded (accountability)
3. Reference authoritative project documentation
4. Follow project-specific rules consistently

---

## Template Structure

When implementing ACE for a new project, you need:

```
project/
├── CLAUDE.md (or INSTRUCTIONS.md)     # Main instruction file
├── .claude/INSTRUCTIONS.md            # Claude Code specific (symlink or copy)
├── .gemini/instructions.md            # Gemini specific (symlink or copy)
├── .github/copilot-instructions.md    # Copilot specific (symlink or copy)
└── profiles/                          # Context-specific profiles
    ├── PROFILE_LOADER.md              # Meta-documentation
    ├── base.md                        # Always loaded
    ├── feature-implementation.md      # For new features
    ├── bug-fix-debugging.md           # For fixing issues
    ├── branching-commit.md            # For git operations
    └── refactoring-optimization.md    # For code improvements
```

---

## PART 1: Main Instruction File Template

Create `CLAUDE.md` (or `INSTRUCTIONS.md`) with this structure:

```markdown
# [PROJECT_NAME] Development Instructions

**Project:** [One-line description]
**Last Updated:** [Date]

---

## !! MANDATORY: ACE FRAMEWORK - READ BEFORE ANY RESPONSE !!

**THIS SECTION CANNOT BE SKIPPED. VIOLATION = PROJECT STANDARDS BREACH.**

### Step 1: Identify Task Type (REQUIRED)

Before responding to ANY user request, classify the task:

| Task Type | Trigger Keywords | Profile to Load |
|-----------|------------------|-----------------|
| **Research/Questions** | "what", "how", "explain" | `base.md` only |
| **Feature Implementation** | "implement", "add", "create", "build" | `base.md` + `feature-implementation.md` |
| **Bug Fix/Debugging** | "bug", "error", "fix", "broken" | `base.md` + `bug-fix-debugging.md` |
| **Git Operations** | "branch", "commit", "merge", "PR" | `base.md` + `branching-commit.md` |
| **Refactoring** | "refactor", "optimize", "cleanup" | `base.md` + `refactoring-optimization.md` |

### Step 2: Load Profiles (REQUIRED)

Read the appropriate profile files from `profiles/` directory:
- `profiles/base.md` - ALWAYS loaded
- `profiles/[context].md` - Based on task type above

### Step 3: Declare Loaded Profiles (REQUIRED)

Your FIRST response to any task MUST include:

\`\`\`
## Profile Context
**Loaded:** base.md + [relevant-profile].md
**Task:** [One-line description]
\`\`\`

### Step 4: Reference Profiles in Work (REQUIRED)

When making decisions, reference the profile guidance:
- "Per the feature-implementation profile..."
- "Following branching-commit.md guidelines..."

---

## INLINED CRITICAL RULES (From All Profiles)

<!-- CUSTOMIZE: Add your project's most critical rules here -->

### From base.md - Universal Rules
\`\`\`
ALWAYS:
- [Rule 1 - e.g., "Use TypeScript strict mode"]
- [Rule 2 - e.g., "Run tests before committing"]
- [Rule 3 - e.g., "Follow DRY principles"]
\`\`\`

### From feature-implementation.md - Before Building
\`\`\`
PRE-IMPLEMENTATION CHECKLIST:
[] [Check 1 - e.g., "Verify feature is on roadmap"]
[] [Check 2 - e.g., "Search codebase for patterns"]
[] [Check 3 - e.g., "Check database schema"]
\`\`\`

### From branching-commit.md - Git Operations
\`\`\`
BRANCH DECISION:
- Multi-file changes? -> Create branch
- Single-line fix? -> Commit to main
- Experimental? -> Always branch

COMMIT FORMAT:
type(scope): description
\`\`\`

---

## Anti-Patterns (NEVER DO)

\`\`\`
Code:
- [Anti-pattern 1 with fix]
- [Anti-pattern 2 with fix]

Process:
- Skip ACE profile loading -> VIOLATION
- Commit without tests -> NEVER
- Force-push to main -> NEVER
\`\`\`

---

## Validation Checklist (Before Marking Complete)

\`\`\`
[] ACE profile was loaded and declared
[] Code passes linting
[] Tests pass
[] Documentation updated
[] Branch strategy followed
\`\`\`

---

## Project Overview

<!-- CUSTOMIZE: Add your project-specific content below -->

[Describe your project architecture, tech stack, etc.]

---

## Build & Development Commands

\`\`\`bash
# Install
[your install command]

# Run
[your run command]

# Test
[your test command]
\`\`\`

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| `profiles/` | ACE Framework profiles |
| `docs/ARCHITECTURE.md` | System design |
| `docs/ROADMAP.md` | Feature priorities |

---

**ACE Framework profiles located in:** `profiles/`
**For complete profile loading instructions:** `profiles/PROFILE_LOADER.md`
```

---

## PART 2: Base Profile Template

Create `profiles/base.md`:

```markdown
# [PROJECT_NAME] Base Profile

**Profile Type:** Core / Always Loaded
**Version:** 1.0

This profile is loaded for every interaction.

---

## Universal Rules (ALWAYS APPLY)

### Code Standards
<!-- CUSTOMIZE: Your project's core coding rules -->
- [Standard 1]
- [Standard 2]
- [Standard 3]

### Pre-Implementation Checklist
\`\`\`
[] [Check 1]
[] [Check 2]
[] [Check 3]
\`\`\`

---

## Project Context

**Stack:**
- Frontend: [Your frontend]
- Backend: [Your backend]
- Database: [Your database]

---

## Documentation Index

- [Doc 1]: [Purpose]
- [Doc 2]: [Purpose]

---

## Validation Checklist

Before marking work complete:
\`\`\`
[] [Validation 1]
[] [Validation 2]
[] [Validation 3]
\`\`\`

---

**This profile is ALWAYS loaded. Choose additional profiles based on task context.**
```

---

## PART 3: Context-Specific Profile Template

Create each context profile (e.g., `profiles/feature-implementation.md`):

```markdown
# [PROJECT_NAME] [Context] Profile

**Profile Type:** Context-Specific
**Activated By:** [Trigger keywords]
**Version:** 1.0
**Reference:** [Related docs]

Load this profile when [context description].

---

## [Context] Workflow

### 1. Pre-Work (CRITICAL)

Before starting:
\`\`\`
[] [Check 1]
[] [Check 2]
[] [Check 3]
\`\`\`

### 2. Key Decisions

Ask yourself:
- [Question 1]
- [Question 2]
- [Question 3]

---

## Key Patterns

<!-- CUSTOMIZE: Add context-specific code patterns -->

### Pattern 1
\`\`\`[language]
// Example code
\`\`\`

### Pattern 2
\`\`\`[language]
// Example code
\`\`\`

---

## Common Anti-Patterns

\`\`\`
[X] DON'T: [Bad practice] -> [Good practice]
[X] DON'T: [Bad practice] -> [Good practice]
\`\`\`

---

## Completion Checklist

Work is NOT complete until:
\`\`\`
[] [Completion criteria 1]
[] [Completion criteria 2]
[] [Completion criteria 3]
\`\`\`

---

**See [related doc] for additional patterns.**
```

---

## PART 4: Profile Loader Template

Create `profiles/PROFILE_LOADER.md`:

```markdown
# [PROJECT_NAME] Profile Loader Guide

**Purpose:** Load appropriate instruction profiles based on context

---

## Available Profiles

| Profile | Triggers | Use When |
|---------|----------|----------|
| **base.md** | Always | Every interaction |
| **feature-implementation.md** | Feature/implement keywords | Building new functionality |
| **bug-fix-debugging.md** | Bug/error/fix keywords | Investigating issues |
| **branching-commit.md** | Branch/git/commit keywords | Git operations |
| **refactoring-optimization.md** | Refactor/optimize keywords | Improving code |

---

## Context Detection Keywords

### Feature Keywords
\`\`\`
"implement", "add", "feature", "new", "build", "create"
\`\`\`

### Debugging Keywords
\`\`\`
"bug", "error", "fix", "debug", "issue", "failing", "broken"
\`\`\`

### Git Keywords
\`\`\`
"branch", "commit", "merge", "git", "push", "PR"
\`\`\`

### Refactoring Keywords
\`\`\`
"refactor", "optimize", "performance", "cleanup", "improve"
\`\`\`

---

## For AI Assistants

### At Conversation Start
1. Detect context from user message
2. Load base.md
3. Load appropriate context profile(s)
4. Declare loaded profiles in response
5. Reference profiles when making decisions

### Example Response Header
\`\`\`
## Profile Context
**Loaded:** base.md + feature-implementation.md
**Task:** Implement user authentication

---

[Response content]
\`\`\`
```

---

## Implementation Checklist for New Projects

When setting up ACE for a new project:

```
[] Create profiles/ directory
[] Create base.md with project-specific rules
[] Create context profiles for common workflows:
   [] feature-implementation.md
   [] bug-fix-debugging.md
   [] branching-commit.md
   [] refactoring-optimization.md
[] Create PROFILE_LOADER.md
[] Create main CLAUDE.md with:
   [] ACE Framework section (mandatory steps)
   [] Inlined critical rules
   [] Anti-patterns
   [] Validation checklist
   [] Project-specific content
[] Sync to .claude/, .gemini/, .github/ as needed
[] Test with sample prompts
```

---

## Customization Guide

### Essential Customizations

1. **Trigger Keywords**: Match your project's vocabulary
2. **Code Standards**: Your linting, typing, testing rules
3. **Checklists**: Your pre/post work validation steps
4. **Anti-Patterns**: Your project's common mistakes
5. **Documentation Index**: Your actual docs structure

### Optional Enhancements

1. **Tier/Role System**: If your project has user tiers or roles
2. **API Patterns**: If you have specific API conventions
3. **Database Rules**: If you have ORM/schema conventions
4. **Security Rules**: If you have security-critical code

---

## Agent Prompt for Generation

Use this prompt to have an AI generate ACE for your project:

```
I need you to implement the ACE Framework for my project. Here's the context:

**Project Name:** [NAME]
**Tech Stack:** [STACK]
**Key Documentation:** [LIST]
**Core Rules:** [LIST YOUR MUST-FOLLOW RULES]
**Common Workflows:** [LIST YOUR TYPICAL TASKS]
**Anti-Patterns:** [LIST YOUR KNOWN MISTAKES]

Please generate:
1. CLAUDE.md with the full ACE Framework
2. profiles/base.md
3. profiles/feature-implementation.md
4. profiles/bug-fix-debugging.md
5. profiles/branching-commit.md
6. profiles/PROFILE_LOADER.md

Use the ACE Framework Template structure and customize for my project.
```

---

**This template is version-controlled. Updates should follow semantic versioning.**
