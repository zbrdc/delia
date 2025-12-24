# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Automatic Context Detection for Delia Framework.

Detects task type from user messages and automatically loads relevant
playbooks and profiles without requiring explicit tool calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import structlog

log = structlog.get_logger()

TaskType = Literal[
    "coding", "testing", "debugging", "architecture",
    "git", "security", "api", "performance", "deployment", "project"
]

# Keyword patterns for task type detection
# Each pattern has a weight: higher weight = more indicative of task type
# Format: (pattern, weight) where weight is 1-3 (3 = highly specific)
TASK_PATTERNS_WEIGHTED: dict[TaskType, list[tuple[str, int]]] = {
    "coding": [
        (r"\b(implement|add|create|build|write|modify|update)\b", 2),
        (r"\b(function|class|method|module|component)\b", 1),
        (r"\b(code|coding|program|develop)\b", 2),
        (r"\b(feature|functionality)\b", 2),
    ],
    "testing": [
        (r"\b(test|tests|testing|pytest|unittest|jest|mocha)\b", 3),
        (r"\b(coverage|mock|fixture|assert|spec)\b", 2),
        (r"\b(unit test|integration test|e2e|end.to.end)\b", 3),
        (r"\b(tdd|test.driven)\b", 3),
    ],
    "debugging": [
        (r"\b(bug|debug|issue|problem)s?\b", 3),
        (r"\b(error|exception|traceback|stack.?trace)s?\b", 2),
        (r"\b(broken|failing|fails?|crash(ed|es|ing)?|down|outage)\b", 3),
        (r"\b(not working|doesn.?t work|won.?t work|stopped)\b", 3),
        (r"\b(fix|fixing|fixed)\b", 2),  # "fix" often means debugging
        (r"\b(investigate|diagnos(e|ing)|troubleshoot)\b", 3),
    ],
    "architecture": [
        (r"\b(design|architecture|architect)\b", 3),
        (r"\b(pattern|structure|restructure)\b", 2),
        (r"\b(adr|decision|refactor|redesign)\b", 3),
        (r"\b(dependency|interface|abstract|coupling)\b", 2),
        (r"\b(singleton|factory|strategy|observer|repository)\b", 3),
        # Planning/thinking patterns
        (r"\b(plan|planning|planned)\b", 3),
        (r"\b(think|thinking|thought).*(through|about|over)\b", 2),
        (r"\b(approach|strategy|strategize)\b", 2),
        (r"\b(consider|considering|evaluate|evaluating)\b", 2),
        (r"\b(trade.?off|pros?.and.cons?|compare|comparison)\b", 3),
        (r"\b(breakdown|break\s+down|decompose)\b", 2),
        (r"\b(how.should|what.approach|best.way)\b", 3),
    ],
    "git": [
        (r"\b(git|commit|branch|merge|rebase)\b", 3),
        (r"\b(push|pull)\b", 2),  # Separate from PR
        (r"\bPR\b", 3),  # Case-sensitive PR
        (r"\b(pull.?request|merge.?request)\b", 3),
        (r"\b(checkout|stash|diff|cherry.?pick)\b", 2),
        (r"\b(conflict|remote|origin|upstream)\b", 2),
        # Colloquial/conversational git patterns
        (r"\bcheck.?(this.?)?in\b", 3),  # old VCS terminology
        (r"\bland.*(on|to|this|it)\b", 2),  # "land this on main"
        (r"\bsquash\b", 3),  # git squash
        (r"\b(main|master|dev|develop)\s+branch\b", 2),  # branch name mentions
        (r"\bversion.?control\b", 2),  # explicit VCS reference
        (r"\b(amend|revert|reset)\b", 3),  # git commands
    ],
    "security": [
        (r"\b(security|secure|insecure)\b", 3),
        (r"\b(auth|authentication|authorization|oauth|jwt)\b", 3),
        (r"\b(password|token|secret|credential|api.?key)\b", 2),
        (r"\b(vulnerability|exploit|injection|xss|csrf|sql.?injection)\b", 3),
        (r"\b(encrypt|decrypt|hash|sanitize)\b", 2),
    ],
    "api": [
        (r"\b(api|endpoint|route)\b", 3),
        (r"\b(rest|graphql|grpc|websocket)\b", 3),
        (r"\b(request|response|http|status.?code)\b", 2),
        (r"\b(json|payload|body|header)\b", 1),
        (r"\b(GET|POST|PUT|DELETE|PATCH)\b", 2),  # Case-sensitive HTTP methods
    ],
    "performance": [
        (r"\b(performance|optimize|optimization)\b", 3),
        (r"\b(slow|slower|fastest|faster|speed|speed.?up)\b", 3),
        (r"\b(cache|caching|memoize|memoization)\b", 3),
        (r"\b(memory|cpu|gpu|resource)\b", 2),
        (r"\b(profile|profiling|benchmark|benchmarking)\b", 3),
        (r"\b(latency|throughput|bottleneck|timeout)\b", 3),
    ],
    "deployment": [
        (r"\b(deploy|deployment|deploying)\b", 3),
        (r"\b(ci.?cd|pipeline|github.?actions|jenkins)\b", 3),
        (r"\b(docker|container|kubernetes|k8s|helm)\b", 3),
        (r"\b(production|staging|environment)\b", 2),
        (r"\b(release|publish|ship)\b", 2),
    ],
    "project": [
        (r"\b(project|codebase|repository|repo)\b", 2),
        (r"\b(structure|organization|layout|overview)\b", 2),
        (r"\b(setup|configure|install|installation)\b", 2),
        (r"\b(how does|what is|where is|explain|describe)\b", 3),
        (r"\b(documentation|docs|readme)\b", 2),
    ],
}

# Convert to simple pattern dict for backwards compatibility
TASK_PATTERNS: dict[TaskType, list[str]] = {
    task_type: [p[0] for p in patterns]
    for task_type, patterns in TASK_PATTERNS_WEIGHTED.items()
}

# Compile weighted patterns for efficiency
# Format: {task_type: [(compiled_pattern, weight, is_case_sensitive), ...]}
COMPILED_PATTERNS_WEIGHTED: dict[TaskType, list[tuple[re.Pattern, int, bool]]] = {}

for task_type, patterns in TASK_PATTERNS_WEIGHTED.items():
    compiled = []
    for pattern, weight in patterns:
        # Check if pattern should be case-sensitive (uppercase letters in pattern)
        has_uppercase = any(c.isupper() for c in pattern.replace(r"\b", ""))
        if has_uppercase:
            # Case-sensitive patterns (PR, GET, POST, etc.)
            compiled.append((re.compile(pattern), weight, True))
        else:
            # Case-insensitive patterns
            compiled.append((re.compile(pattern, re.IGNORECASE), weight, False))
    COMPILED_PATTERNS_WEIGHTED[task_type] = compiled

# Backwards compatible non-weighted version
COMPILED_PATTERNS: dict[TaskType, list[re.Pattern]] = {
    task_type: [p[0] for p in patterns]
    for task_type, patterns in COMPILED_PATTERNS_WEIGHTED.items()
}


@dataclass
class DetectedContext:
    """Result of context detection."""
    primary_task: TaskType
    secondary_tasks: list[TaskType]
    confidence: float
    matched_keywords: list[str]

    def all_tasks(self) -> list[TaskType]:
        """Return all detected tasks, primary first."""
        return [self.primary_task] + self.secondary_tasks


def detect_task_type(message: str) -> DetectedContext:
    """
    Detect task type(s) from a user message using weighted pattern matching.

    Uses weighted scores where higher weights indicate more specific indicators
    of a task type (weight 3 = highly specific, 1 = general).

    Args:
        message: The user's message or query

    Returns:
        DetectedContext with primary task, secondary tasks, and metadata
    """
    if not message:
        return DetectedContext(
            primary_task="project",
            secondary_tasks=[],
            confidence=0.0,
            matched_keywords=[],
        )

    # Score each task type using weighted patterns
    scores: dict[TaskType, tuple[int, list[str]]] = {}

    for task_type, patterns in COMPILED_PATTERNS_WEIGHTED.items():
        total_weight = 0
        matches: list[str] = []

        for pattern, weight, _is_case_sensitive in patterns:
            found = pattern.findall(message)
            if found:
                # Add weighted score
                total_weight += len(found) * weight
                matches.extend(found)

        if total_weight > 0:
            scores[task_type] = (total_weight, matches)

    if not scores:
        # Default to project for general queries
        return DetectedContext(
            primary_task="project",
            secondary_tasks=[],
            confidence=0.3,
            matched_keywords=[],
        )

    # Sort by weighted score descending
    sorted_tasks = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)

    primary_task = sorted_tasks[0][0]
    primary_score = sorted_tasks[0][1][0]
    primary_keywords = sorted_tasks[0][1][1]

    # Secondary tasks (score > 0 and not primary)
    secondary_tasks = [t for t, (s, _) in sorted_tasks[1:] if s > 0][:2]

    # Confidence based on weighted score (6+ weighted points = 100% confidence)
    confidence = min(1.0, primary_score / 6)

    log.debug(
        "context_detected",
        primary=primary_task,
        secondary=secondary_tasks,
        confidence=confidence,
        weighted_score=primary_score,
        keywords=primary_keywords[:5],
    )

    return DetectedContext(
        primary_task=primary_task,
        secondary_tasks=secondary_tasks,
        confidence=confidence,
        matched_keywords=primary_keywords[:10],
    )


def get_relevant_profiles(context: DetectedContext) -> list[str]:
    """
    Get list of profile filenames relevant to the detected context.

    Args:
        context: The detected context

    Returns:
        List of profile filenames (e.g., ["core.md", "coding.md"])
    """
    profiles = ["core.md"]  # Always include core

    # Map task types to profiles
    task_to_profile = {
        "coding": ["coding.md", "python.md"],
        "testing": ["testing.md"],
        "debugging": ["debugging.md"],
        "architecture": ["architecture.md"],
        "git": ["git.md"],
        "security": ["security.md"],
        "api": ["api.md", "fastapi.md"],
        "performance": ["performance.md"],
        "deployment": ["deployment.md"],
        "project": [],  # Just core is enough
    }

    # Add primary task profiles
    profiles.extend(task_to_profile.get(context.primary_task, []))

    # Add secondary task profiles (first one only to avoid overload)
    for task in context.secondary_tasks[:1]:
        for profile in task_to_profile.get(task, []):
            if profile not in profiles:
                profiles.append(profile)

    return profiles


# =============================================================================
# FILE-BASED CONTEXT DETECTION
# =============================================================================

# File path patterns that indicate task types
# Format: (glob_pattern, task_type, weight)
FILE_PATTERNS: list[tuple[str, TaskType, int]] = [
    # Testing
    ("test_*.py", "testing", 3),
    ("*_test.py", "testing", 3),
    ("*_test.go", "testing", 3),
    ("*.test.ts", "testing", 3),
    ("*.test.tsx", "testing", 3),
    ("*.spec.ts", "testing", 3),
    ("*.spec.tsx", "testing", 3),
    ("tests/**", "testing", 2),
    ("__tests__/**", "testing", 2),
    ("conftest.py", "testing", 3),
    ("pytest.ini", "testing", 2),
    ("jest.config.*", "testing", 2),

    # Deployment/CI
    ("Dockerfile", "deployment", 3),
    ("docker-compose*.yml", "deployment", 3),
    (".github/workflows/*.yml", "deployment", 3),
    (".gitlab-ci.yml", "deployment", 3),
    ("Jenkinsfile", "deployment", 3),
    ("k8s/**", "deployment", 2),
    ("kubernetes/**", "deployment", 2),
    ("helm/**", "deployment", 2),
    ("terraform/**", "deployment", 2),

    # Security
    ("**/auth/**", "security", 2),
    ("**/security/**", "security", 2),
    ("**/authentication/**", "security", 2),
    ("**/authorization/**", "security", 2),

    # API
    ("**/api/**", "api", 2),
    ("**/routes/**", "api", 2),
    ("**/endpoints/**", "api", 2),
    ("openapi.yaml", "api", 3),
    ("swagger.yaml", "api", 3),

    # Git
    (".gitignore", "git", 2),
    (".gitattributes", "git", 2),

    # Performance
    ("**/cache/**", "performance", 2),
    ("**/caching/**", "performance", 2),
]


def detect_from_files(file_paths: list[str]) -> dict[TaskType, int]:
    """
    Detect task type signals from file paths.

    Args:
        file_paths: List of file paths being worked on

    Returns:
        Dict mapping task types to weighted scores
    """
    import fnmatch
    from pathlib import Path

    scores: dict[TaskType, int] = {}

    for file_path in file_paths:
        path = Path(file_path)
        name = path.name
        full_path = str(path)

        for pattern, task_type, weight in FILE_PATTERNS:
            # Try matching against filename and full path
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full_path, pattern):
                scores[task_type] = scores.get(task_type, 0) + weight
                log.debug("file_pattern_match", file=name, pattern=pattern, task=task_type)

    return scores


# File extension to language/framework mapping for profile loading
# Format: extension -> (language, frameworks, profile_hints)
# Profile hints map to actual files in templates/profiles/
EXTENSION_MAP: dict[str, tuple[str, list[str], list[str]]] = {
    # Python
    ".py": ("python", [], ["python.md"]),
    ".pyi": ("python", [], ["python.md"]),
    ".pyx": ("python", ["cython"], ["python.md"]),
    ".ipynb": ("python", ["jupyter"], ["python.md", "ml.md"]),

    # TypeScript/JavaScript
    ".ts": ("typescript", [], ["typescript.md"]),
    ".tsx": ("typescript", ["react"], ["typescript.md", "react.md"]),
    ".js": ("javascript", [], ["typescript.md"]),  # Use typescript.md for JS too
    ".jsx": ("javascript", ["react"], ["typescript.md", "react.md"]),
    ".mjs": ("javascript", [], ["typescript.md"]),
    ".cjs": ("javascript", [], ["typescript.md"]),
    ".vue": ("vue", ["vue"], ["vue.md", "typescript.md"]),
    ".svelte": ("svelte", ["svelte"], ["svelte.md", "typescript.md"]),

    # Go
    ".go": ("go", [], ["golang.md"]),

    # Rust
    ".rs": ("rust", [], ["rust.md"]),

    # C/C++
    ".c": ("c", [], ["c.md"]),
    ".h": ("c", [], ["c.md"]),
    ".cpp": ("cpp", [], ["cpp.md"]),
    ".hpp": ("cpp", [], ["cpp.md"]),
    ".cc": ("cpp", [], ["cpp.md"]),

    # Mobile
    ".swift": ("swift", ["ios"], ["ios.md"]),
    ".m": ("objc", ["ios"], ["ios.md"]),
    ".kt": ("kotlin", ["android"], ["android.md"]),
    ".kts": ("kotlin", ["android"], ["android.md"]),
    ".java": ("java", [], []),  # Could be Android or backend
    ".dart": ("dart", ["flutter"], ["flutter.md"]),

    # PHP/Laravel
    ".php": ("php", [], ["laravel.md"]),
    ".blade.php": ("php", ["laravel"], ["laravel.md"]),

    # Ruby
    ".rb": ("ruby", [], []),
    ".rake": ("ruby", [], []),
    ".erb": ("ruby", ["rails"], []),

    # Shell/DevOps
    ".sh": ("shell", [], []),
    ".bash": ("shell", [], []),
    ".zsh": ("shell", [], []),

    # SQL
    ".sql": ("sql", [], []),

    # Solidity/Web3
    ".sol": ("solidity", [], ["solidity.md"]),

    # Config/Data
    ".yaml": ("yaml", [], []),
    ".yml": ("yaml", [], []),
    ".json": ("json", [], []),
    ".toml": ("toml", [], []),

    # CSS/Styling
    ".css": ("css", [], ["css.md"]),
    ".scss": ("css", ["sass"], ["css.md"]),
    ".less": ("css", ["less"], ["css.md"]),

    # Markup
    ".html": ("html", [], []),
    ".htm": ("html", [], []),
    ".md": ("markdown", [], []),
    ".mdx": ("mdx", ["react"], ["react.md"]),
}


@dataclass
class FileContext:
    """Context derived from file analysis."""
    language: str | None = None
    frameworks: list[str] | None = None
    profile_hints: list[str] | None = None
    task_scores: dict[str, int] | None = None


def detect_from_files_enhanced(file_paths: list[str]) -> tuple[dict[TaskType, int], FileContext]:
    """
    Enhanced file detection that returns both task scores and language context.

    Args:
        file_paths: List of file paths being worked on

    Returns:
        Tuple of (task_scores, FileContext with language/framework info)
    """
    import fnmatch
    from pathlib import Path

    scores: dict[TaskType, int] = {}
    languages: set[str] = set()
    frameworks: set[str] = set()
    profile_hints: set[str] = set()

    for file_path in file_paths:
        path = Path(file_path)
        name = path.name
        full_path = str(path)

        # Check task patterns (testing, deployment, etc.)
        for pattern, task_type, weight in FILE_PATTERNS:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(full_path, pattern):
                scores[task_type] = scores.get(task_type, 0) + weight
                log.debug("file_pattern_match", file=name, pattern=pattern, task=task_type)

        # Check extension for language/framework
        ext = path.suffix.lower()
        if ext in EXTENSION_MAP:
            lang, fws, hints = EXTENSION_MAP[ext]
            languages.add(lang)
            frameworks.update(fws)
            profile_hints.update(hints)

        # Special file detection - frameworks and languages from config files
        name_lower = name.lower()

        # Python ecosystem
        if name in {"pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"}:
            languages.add("python")
            profile_hints.add("python.md")
        elif name == "manage.py" or "django" in full_path:
            frameworks.add("django")
            profile_hints.add("django.md")

        # JavaScript/Node ecosystem
        elif name == "package.json":
            languages.add("javascript")
            # Framework detection will happen via content analysis if needed
        elif name in {"next.config.js", "next.config.mjs", "next.config.ts"}:
            frameworks.add("nextjs")
            profile_hints.add("nextjs.md")
        elif name in {"nuxt.config.js", "nuxt.config.ts"}:
            frameworks.add("nuxt")
            profile_hints.add("vue.md")
        elif name == "angular.json":
            frameworks.add("angular")
            profile_hints.add("angular.md")
        elif name in {"vite.config.ts", "vite.config.js"}:
            frameworks.add("vite")
        elif name in {"svelte.config.js", "svelte.config.ts"}:
            frameworks.add("svelte")
            profile_hints.add("svelte.md")
        elif name == "nest-cli.json" or "nestjs" in full_path:
            frameworks.add("nestjs")
            profile_hints.add("nestjs.md")

        # Rust
        elif name == "Cargo.toml":
            languages.add("rust")
            profile_hints.add("rust.md")

        # Go
        elif name == "go.mod" or name == "go.sum":
            languages.add("go")
            profile_hints.add("golang.md")

        # Ruby/Rails
        elif name == "Gemfile" or name == "Rakefile":
            languages.add("ruby")
        elif name == "config.ru" or "rails" in full_path:
            frameworks.add("rails")

        # PHP/Laravel
        elif name == "composer.json":
            languages.add("php")
        elif name == "artisan" or "laravel" in full_path:
            frameworks.add("laravel")
            profile_hints.add("laravel.md")

        # Mobile
        elif name == "Podfile" or name.endswith(".xcodeproj"):
            frameworks.add("ios")
            profile_hints.add("ios.md")
        elif name == "build.gradle" or name == "build.gradle.kts":
            frameworks.add("android")
            profile_hints.add("android.md")
        elif name == "pubspec.yaml":
            frameworks.add("flutter")
            profile_hints.add("flutter.md")

        # ML/AI
        elif name in {"model.py", "train.py", "dataset.py"} or "models" in full_path:
            profile_hints.add("ml.md")
        elif "torch" in full_path or "tensorflow" in full_path:
            profile_hints.add("deeplearning.md")

        # Web3/Blockchain
        elif name == "hardhat.config.js" or name == "truffle-config.js":
            frameworks.add("solidity")
            profile_hints.add("solidity.md")

        # FastAPI detection
        elif "fastapi" in full_path or name == "main.py":
            # Will verify via content, but hint it
            profile_hints.add("fastapi.md")

    file_ctx = FileContext(
        language=list(languages)[0] if len(languages) == 1 else None,
        frameworks=list(frameworks) if frameworks else None,
        profile_hints=list(profile_hints) if profile_hints else None,
        task_scores=scores if scores else None,
    )

    return scores, file_ctx


# =============================================================================
# CODE-BASED CONTEXT DETECTION
# =============================================================================

# Code patterns that indicate task types
# Format: (regex_pattern, task_type, weight)
CODE_PATTERNS: list[tuple[str, TaskType, int]] = [
    # Testing
    (r"@pytest\.(mark|fixture)", "testing", 3),
    (r"\bassert\s+", "testing", 2),
    (r"\bmock\.|Mock\(|patch\(", "testing", 3),
    (r"describe\(|it\(|test\(|expect\(", "testing", 3),
    (r"beforeEach|afterEach|beforeAll|afterAll", "testing", 2),

    # Security
    (r"password|secret|api_key|token", "security", 2),
    (r"encrypt|decrypt|hash|bcrypt|argon", "security", 3),
    (r"authenticate|authorize|jwt|oauth", "security", 3),
    (r"sanitize|escape|validate_input", "security", 2),

    # Performance
    (r"@cache|@lru_cache|@cached", "performance", 3),
    (r"asyncio\.gather|concurrent\.futures", "performance", 2),
    (r"\.prefetch_related|\.select_related", "performance", 2),
    (r"redis\.|memcache", "performance", 2),

    # API
    (r"@app\.(get|post|put|delete|patch)", "api", 3),
    (r"@router\.(get|post|put|delete|patch)", "api", 3),
    (r"FastAPI|APIRouter|HTTPException", "api", 3),
    (r"fetch\(|axios\.|requests\.", "api", 2),
    (r"response\.json|JsonResponse", "api", 2),

    # Debugging
    (r"print\(|console\.log|logger\.(debug|error)", "debugging", 1),
    (r"breakpoint\(\)|pdb\.set_trace", "debugging", 3),
    (r"traceback|exception|raise\s+\w+Error", "debugging", 2),

    # Deployment
    (r"FROM\s+\w+|COPY\s+|RUN\s+|ENTRYPOINT", "deployment", 3),
    (r"docker|container|kubernetes|k8s", "deployment", 2),

    # Git (rarely in code, but sometimes)
    (r"git\.(commit|push|pull|clone)", "git", 2),
]

# Compile code patterns
COMPILED_CODE_PATTERNS: list[tuple[re.Pattern, TaskType, int]] = [
    (re.compile(pattern, re.IGNORECASE), task_type, weight)
    for pattern, task_type, weight in CODE_PATTERNS
]


def detect_from_code(code_snippet: str) -> dict[TaskType, int]:
    """
    Detect task type signals from code content.

    Args:
        code_snippet: Code being edited or reviewed

    Returns:
        Dict mapping task types to weighted scores
    """
    scores: dict[TaskType, int] = {}

    for pattern, task_type, weight in COMPILED_CODE_PATTERNS:
        matches = pattern.findall(code_snippet)
        if matches:
            scores[task_type] = scores.get(task_type, 0) + (len(matches) * weight)
            log.debug("code_pattern_match", pattern=pattern.pattern[:30], task=task_type, count=len(matches))

    return scores


@dataclass
class EnhancedDetectedContext(DetectedContext):
    """Extended context with file/code analysis results."""
    file_context: FileContext | None = None
    detected_language: str | None = None


def detect_task_type_enhanced(
    message: str,
    working_files: list[str] | None = None,
    code_snippet: str | None = None,
) -> EnhancedDetectedContext:
    """
    Enhanced task type detection using message, files, and code.

    Combines signals from:
    1. Message keywords (existing behavior)
    2. File paths being worked on (new)
    3. Code content being edited (new)

    Args:
        message: The user's message
        working_files: Optional list of file paths being edited
        code_snippet: Optional code content being modified

    Returns:
        EnhancedDetectedContext with combined detection results and language info
    """
    from .language import detect_language

    # Start with base message detection
    base_context = detect_task_type(message)

    # Collect additional scores
    additional_scores: dict[TaskType, int] = {}
    additional_keywords: list[str] = []
    file_ctx: FileContext | None = None
    detected_lang: str | None = None

    # Add file-based signals (enhanced version with language detection)
    if working_files:
        file_scores, file_ctx = detect_from_files_enhanced(working_files)
        for task, score in file_scores.items():
            additional_scores[task] = additional_scores.get(task, 0) + score
        if file_scores:
            additional_keywords.append(f"files:{list(file_scores.keys())}")
        if file_ctx.language:
            additional_keywords.append(f"lang:{file_ctx.language}")
            detected_lang = file_ctx.language

    # Add code-based signals
    if code_snippet:
        code_scores = detect_from_code(code_snippet)
        for task, score in code_scores.items():
            additional_scores[task] = additional_scores.get(task, 0) + score
        if code_scores:
            additional_keywords.append(f"code:{list(code_scores.keys())}")

        # Use existing language detector for code content
        if not detected_lang:
            file_path = working_files[0] if working_files else ""
            detected_lang = detect_language(code_snippet, file_path)

    # If no additional signals, return base detection (as enhanced type)
    if not additional_scores:
        return EnhancedDetectedContext(
            primary_task=base_context.primary_task,
            secondary_tasks=base_context.secondary_tasks,
            confidence=base_context.confidence,
            matched_keywords=base_context.matched_keywords,
            file_context=file_ctx,
            detected_language=detected_lang,
        )

    # Combine scores: base detection + additional signals
    combined_scores: dict[TaskType, int] = {}

    # Add base score
    combined_scores[base_context.primary_task] = int(base_context.confidence * 6)
    for task in base_context.secondary_tasks:
        combined_scores[task] = combined_scores.get(task, 0) + 2

    # Add file/code scores
    for task, score in additional_scores.items():
        combined_scores[task] = combined_scores.get(task, 0) + score

    # Determine new primary and secondary tasks
    sorted_tasks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    primary_task = sorted_tasks[0][0]
    primary_score = sorted_tasks[0][1]
    secondary_tasks = [t for t, s in sorted_tasks[1:] if s > 0][:2]

    # Log if detection changed
    if primary_task != base_context.primary_task:
        log.info(
            "context_enhanced_by_files_or_code",
            original=base_context.primary_task,
            new=primary_task,
            additional_scores=additional_scores,
            language=detected_lang,
        )

    return EnhancedDetectedContext(
        primary_task=primary_task,
        secondary_tasks=secondary_tasks,
        confidence=min(1.0, primary_score / 6),
        matched_keywords=base_context.matched_keywords + additional_keywords,
        file_context=file_ctx,
        detected_language=detected_lang,
    )


class ContextManager:
    """
    Manages automatic context detection and playbook injection.

    This is a singleton that tracks the current context and provides
    methods for auto-loading relevant playbooks.
    """

    _instance: "ContextManager | None" = None

    def __new__(cls) -> "ContextManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_context: DetectedContext | None = None
        self._last_message: str = ""
        self._auto_inject_enabled: bool = True

    @property
    def current_context(self) -> DetectedContext | None:
        return self._current_context

    def detect_and_update(self, message: str) -> DetectedContext:
        """
        Detect context from message and update internal state.

        Args:
            message: User message to analyze

        Returns:
            Detected context
        """
        self._last_message = message
        self._current_context = detect_task_type(message)
        return self._current_context

    def get_auto_context_bullets(self, project_path: str | None = None) -> str:
        """
        Get formatted playbook bullets for current context.

        Returns bullets as a formatted string ready for injection
        into agent responses or tool results.
        """
        if not self._current_context:
            return ""

        from .playbook import get_playbook_manager
        from pathlib import Path

        pm = get_playbook_manager()
        if project_path:
            pm.set_project(Path(project_path))

        parts = []

        # Get bullets for primary task
        bullets = pm.get_top_bullets(self._current_context.primary_task, limit=5)
        if bullets:
            parts.append(f"\n## Auto-loaded Context: {self._current_context.primary_task.title()}\n")
            parts.append("Apply these strategies to your current task:\n")
            for b in bullets:
                parts.append(f"- [{b.id}] {b.content}")

        # Get bullets for secondary tasks (fewer)
        for task in self._current_context.secondary_tasks[:1]:
            bullets = pm.get_top_bullets(task, limit=2)
            if bullets:
                parts.append(f"\n### Also relevant ({task}):\n")
                for b in bullets:
                    parts.append(f"- [{b.id}] {b.content}")

        return "\n".join(parts)

    def enable_auto_inject(self, enabled: bool = True):
        """Enable or disable automatic context injection."""
        self._auto_inject_enabled = enabled
        log.info("auto_inject_toggled", enabled=enabled)

    def is_auto_inject_enabled(self) -> bool:
        return self._auto_inject_enabled


def get_context_manager() -> ContextManager:
    """Get the singleton ContextManager instance."""
    return ContextManager()


# =============================================================================
# DYNAMIC PATTERN LEARNING
# =============================================================================

import json
from pathlib import Path
from datetime import datetime


@dataclass
class LearnedPattern:
    """A pattern learned from user feedback."""
    pattern: str
    task_type: TaskType
    weight: int  # 1-3
    success_count: int = 0
    failure_count: int = 0
    created_at: str = ""
    source: str = "feedback"  # "feedback" or "profile"

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def effectiveness(self) -> float:
        """Calculate pattern effectiveness (0-1)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "task_type": self.task_type,
            "weight": self.weight,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "created_at": self.created_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedPattern":
        return cls(**data)


@dataclass
class NegativePattern:
    """
    A pattern that should NOT trigger a task type.

    Used to learn from false positives like "injection suite" triggering "security"
    when it should be "project" or "architecture".
    """
    trigger_word: str  # The word that caused false positive (e.g., "injection")
    context_pattern: str  # Pattern that distinguishes false positive (e.g., "injection.*suite|suite.*injection")
    wrong_task: TaskType  # The task that was wrongly detected
    correct_task: TaskType  # The task it should have been
    penalty: int = 3  # Score penalty when context matches
    occurrences: int = 1  # How many times this was seen
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "trigger_word": self.trigger_word,
            "context_pattern": self.context_pattern,
            "wrong_task": self.wrong_task,
            "correct_task": self.correct_task,
            "penalty": self.penalty,
            "occurrences": self.occurrences,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NegativePattern":
        return cls(**data)


class PatternLearner:
    """
    Learns and adapts detection patterns based on feedback.

    Stores learned patterns in .delia/learned_patterns.json
    and updates the detection system dynamically.
    """

    def __init__(self, project_path: Path | None = None):
        self.project_path = project_path or Path.cwd()
        self._patterns: list[LearnedPattern] = []
        self._negative_patterns: list[NegativePattern] = []
        self._compiled: dict[TaskType, list[tuple[re.Pattern, int]]] = {}
        self._compiled_negative: dict[TaskType, list[tuple[re.Pattern, int]]] = {}
        self._loaded = False

    def _get_patterns_path(self) -> Path:
        """Get path to learned patterns file."""
        return self.project_path / ".delia" / "learned_patterns.json"

    def load(self) -> None:
        """Load learned patterns from disk."""
        path = self._get_patterns_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._patterns = [LearnedPattern.from_dict(p) for p in data.get("patterns", [])]
                self._negative_patterns = [
                    NegativePattern.from_dict(p) for p in data.get("negative_patterns", [])
                ]
                self._compile_patterns()
                self._compile_negative_patterns()
                log.debug(
                    "learned_patterns_loaded",
                    positive=len(self._patterns),
                    negative=len(self._negative_patterns),
                )
            except Exception as e:
                log.warning("learned_patterns_load_failed", error=str(e))
        self._loaded = True

    def save(self) -> None:
        """Persist learned patterns to disk."""
        path = self._get_patterns_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump({
                    "version": 2,
                    "patterns": [p.to_dict() for p in self._patterns],
                    "negative_patterns": [p.to_dict() for p in self._negative_patterns],
                    "updated_at": datetime.now().isoformat(),
                }, f, indent=2)
            log.debug(
                "learned_patterns_saved",
                positive=len(self._patterns),
                negative=len(self._negative_patterns),
            )
        except Exception as e:
            log.warning("learned_patterns_save_failed", error=str(e))

    def _compile_patterns(self) -> None:
        """Compile learned patterns for efficient matching."""
        self._compiled.clear()
        for pattern in self._patterns:
            if pattern.task_type not in self._compiled:
                self._compiled[pattern.task_type] = []
            try:
                compiled = re.compile(pattern.pattern, re.IGNORECASE)
                self._compiled[pattern.task_type].append((compiled, pattern.weight))
            except re.error:
                log.warning("invalid_learned_pattern", pattern=pattern.pattern)

    def _compile_negative_patterns(self) -> None:
        """Compile negative patterns for efficient matching."""
        self._compiled_negative.clear()
        for neg in self._negative_patterns:
            if neg.wrong_task not in self._compiled_negative:
                self._compiled_negative[neg.wrong_task] = []
            try:
                compiled = re.compile(neg.context_pattern, re.IGNORECASE)
                self._compiled_negative[neg.wrong_task].append((compiled, neg.penalty))
            except re.error:
                log.warning("invalid_negative_pattern", pattern=neg.context_pattern)

    def add_negative_pattern(
        self,
        trigger_word: str,
        context_pattern: str,
        wrong_task: TaskType,
        correct_task: TaskType,
    ) -> NegativePattern:
        """
        Add a negative pattern to prevent false positives.

        When trigger_word causes wrong_task detection but context_pattern matches,
        apply a penalty to wrong_task score.
        """
        if not self._loaded:
            self.load()

        # Check for existing pattern with same context
        for existing in self._negative_patterns:
            if (existing.context_pattern == context_pattern and
                existing.wrong_task == wrong_task):
                existing.occurrences += 1
                self.save()
                return existing

        new_neg = NegativePattern(
            trigger_word=trigger_word,
            context_pattern=context_pattern,
            wrong_task=wrong_task,
            correct_task=correct_task,
        )
        self._negative_patterns.append(new_neg)
        self._compile_negative_patterns()
        self.save()

        log.info(
            "negative_pattern_added",
            trigger=trigger_word,
            context=context_pattern,
            wrong_task=wrong_task,
            correct_task=correct_task,
        )
        return new_neg

    def get_negative_penalty(self, message: str, task_type: TaskType) -> int:
        """Get total penalty for a task type based on negative pattern matches."""
        if not self._loaded:
            self.load()

        total_penalty = 0
        for compiled, penalty in self._compiled_negative.get(task_type, []):
            if compiled.search(message):
                total_penalty += penalty
        return total_penalty

    def add_pattern(
        self,
        pattern: str,
        task_type: TaskType,
        weight: int = 2,
        source: str = "feedback",
    ) -> LearnedPattern:
        """Add a new learned pattern."""
        if not self._loaded:
            self.load()

        # Check for duplicates
        for existing in self._patterns:
            if existing.pattern == pattern and existing.task_type == task_type:
                log.debug("learned_pattern_exists", pattern=pattern)
                return existing

        new_pattern = LearnedPattern(
            pattern=pattern,
            task_type=task_type,
            weight=weight,
            source=source,
        )
        self._patterns.append(new_pattern)
        self._compile_patterns()
        self.save()

        log.info("learned_pattern_added", pattern=pattern, task_type=task_type)
        return new_pattern

    def record_feedback(
        self,
        message: str,
        detected_task: TaskType,
        correct_task: TaskType,
        was_correct: bool,
    ) -> dict:
        """
        Record feedback on a detection to improve future accuracy.

        If detection was wrong, learns new patterns from the message
        that should have indicated the correct task type.

        Args:
            message: The original message
            detected_task: What was detected
            correct_task: What should have been detected
            was_correct: Whether detection was correct

        Returns:
            Dict with learning results
        """
        if not self._loaded:
            self.load()

        result = {
            "was_correct": was_correct,
            "patterns_updated": 0,
            "patterns_added": 0,
        }

        if was_correct:
            # Boost patterns that matched
            for pattern in self._patterns:
                if pattern.task_type == detected_task:
                    try:
                        if re.search(pattern.pattern, message, re.IGNORECASE):
                            pattern.success_count += 1
                            result["patterns_updated"] += 1
                    except re.error:
                        pass
        else:
            # Demote patterns that matched incorrectly
            for pattern in self._patterns:
                if pattern.task_type == detected_task:
                    try:
                        if re.search(pattern.pattern, message, re.IGNORECASE):
                            pattern.failure_count += 1
                            result["patterns_updated"] += 1
                    except re.error:
                        pass

            # =================================================================
            # NEGATIVE PATTERN LEARNING
            # =================================================================
            # Identify which base patterns caused the false positive and learn
            # context patterns that distinguish them.
            # Example: "injection" triggers "security", but "injection suite"
            # should not. Learn that "injection.*suite" → NOT security.
            # =================================================================
            result["negative_patterns_added"] = 0
            trigger_words = []

            # Find which base patterns matched and caused the wrong detection
            for pattern_str, weight in TASK_PATTERNS_WEIGHTED.get(detected_task, []):
                try:
                    match = re.search(pattern_str, message, re.IGNORECASE)
                    if match:
                        # Extract the actual matched word
                        matched_text = match.group(0).lower()
                        trigger_words.append(matched_text)
                except re.error:
                    pass

            # For each trigger word, find neighboring context words
            message_lower = message.lower()
            words = re.findall(r'\b[a-zA-Z]{3,}\b', message_lower)

            for trigger in trigger_words[:2]:  # Limit to top 2 triggers
                # Find words that appear near the trigger
                try:
                    trigger_idx = None
                    for i, w in enumerate(words):
                        if trigger in w or w in trigger:
                            trigger_idx = i
                            break

                    if trigger_idx is not None:
                        # Get neighboring words (within 3 positions)
                        neighbors = []
                        for offset in range(-3, 4):
                            idx = trigger_idx + offset
                            if 0 <= idx < len(words) and idx != trigger_idx:
                                neighbors.append(words[idx])

                        # Create context pattern: trigger + neighbor
                        for neighbor in neighbors[:2]:
                            if len(neighbor) > 2:
                                # Pattern matches trigger near neighbor
                                context_pattern = rf"\b{re.escape(trigger)}\b.*\b{re.escape(neighbor)}\b|\b{re.escape(neighbor)}\b.*\b{re.escape(trigger)}\b"
                                self.add_negative_pattern(
                                    trigger_word=trigger,
                                    context_pattern=context_pattern,
                                    wrong_task=detected_task,
                                    correct_task=correct_task,
                                )
                                result["negative_patterns_added"] += 1
                except Exception:
                    pass

            # Extract words from message that might indicate correct task
            # Simple approach: learn any unique words not in common words
            common_words = {"the", "a", "an", "is", "are", "was", "were", "be",
                          "been", "being", "have", "has", "had", "do", "does",
                          "did", "will", "would", "could", "should", "may",
                          "might", "must", "shall", "can", "to", "of", "in",
                          "for", "on", "with", "at", "by", "from", "as", "into",
                          "through", "during", "before", "after", "above", "below",
                          "between", "under", "again", "further", "then", "once",
                          "here", "there", "when", "where", "why", "how", "all",
                          "each", "few", "more", "most", "other", "some", "such",
                          "no", "nor", "not", "only", "own", "same", "so", "than",
                          "too", "very", "just", "and", "but", "if", "or", "because",
                          "until", "while", "this", "that", "these", "those", "i",
                          "me", "my", "we", "our", "you", "your", "he", "him", "she",
                          "her", "it", "its", "they", "them", "their", "what", "which",
                          "who", "whom", "please", "help", "need", "want", "like"}

            words = re.findall(r'\b[a-zA-Z]{3,}\b', message.lower())
            unique_words = [w for w in words if w not in common_words]

            # Add top unique words as new patterns if we have any
            for word in unique_words[:3]:
                # Check if this word isn't already in base patterns
                already_exists = False
                for patterns in TASK_PATTERNS_WEIGHTED.get(correct_task, []):
                    if word in patterns[0].lower():
                        already_exists = True
                        break

                if not already_exists:
                    self.add_pattern(
                        pattern=rf"\b{word}\b",
                        task_type=correct_task,
                        weight=2,
                        source="feedback",
                    )
                    result["patterns_added"] += 1

        if result["patterns_updated"] > 0 or result["patterns_added"] > 0:
            self.save()

        log.info(
            "detection_feedback_recorded",
            was_correct=was_correct,
            detected=detected_task,
            correct=correct_task,
            updates=result["patterns_updated"],
            added=result["patterns_added"],
        )

        return result

    def get_boosted_patterns(self, task_type: TaskType) -> list[tuple[re.Pattern, int]]:
        """Get learned patterns for a task type, sorted by effectiveness."""
        if not self._loaded:
            self.load()

        patterns = []
        for pattern in self._patterns:
            if pattern.task_type == task_type and pattern.effectiveness >= 0.4:
                try:
                    compiled = re.compile(pattern.pattern, re.IGNORECASE)
                    # Boost weight based on effectiveness
                    boosted_weight = int(pattern.weight * (0.5 + pattern.effectiveness))
                    patterns.append((compiled, boosted_weight))
                except re.error:
                    pass

        return patterns

    def prune_ineffective(self, min_effectiveness: float = 0.3, min_uses: int = 5) -> int:
        """Remove patterns that are consistently wrong."""
        if not self._loaded:
            self.load()

        original_count = len(self._patterns)
        self._patterns = [
            p for p in self._patterns
            if (p.success_count + p.failure_count < min_uses or
                p.effectiveness >= min_effectiveness)
        ]

        removed = original_count - len(self._patterns)
        if removed > 0:
            self._compile_patterns()
            self.save()
            log.info("ineffective_patterns_pruned", count=removed)

        return removed

    def get_stats(self) -> dict:
        """Get statistics about learned patterns."""
        if not self._loaded:
            self.load()

        by_task = {}
        for pattern in self._patterns:
            if pattern.task_type not in by_task:
                by_task[pattern.task_type] = {"count": 0, "avg_effectiveness": 0.0}
            by_task[pattern.task_type]["count"] += 1

        for task_type, stats in by_task.items():
            patterns = [p for p in self._patterns if p.task_type == task_type]
            if patterns:
                stats["avg_effectiveness"] = sum(p.effectiveness for p in patterns) / len(patterns)

        return {
            "total_patterns": len(self._patterns),
            "total_negative_patterns": len(self._negative_patterns),
            "by_task_type": by_task,
            "top_patterns": [
                {"pattern": p.pattern, "task_type": p.task_type, "effectiveness": p.effectiveness}
                for p in sorted(self._patterns, key=lambda x: x.effectiveness, reverse=True)[:10]
            ],
            "negative_patterns": [
                {
                    "trigger": n.trigger_word,
                    "wrong_task": n.wrong_task,
                    "correct_task": n.correct_task,
                    "occurrences": n.occurrences,
                }
                for n in self._negative_patterns[:10]
            ],
        }


# Global learner instance
_pattern_learner: PatternLearner | None = None


def get_pattern_learner(project_path: Path | None = None) -> PatternLearner:
    """Get or create the pattern learner for a project."""
    global _pattern_learner
    if _pattern_learner is None or (project_path and _pattern_learner.project_path != project_path):
        _pattern_learner = PatternLearner(project_path)
    return _pattern_learner


def detect_with_learning(message: str, project_path: Path | None = None) -> DetectedContext:
    """
    Detect task type using both base patterns and learned patterns.

    This is the recommended entry point that combines static patterns
    with dynamically learned ones.
    """
    # Get base detection
    context = detect_task_type(message)

    # Enhance with learned patterns
    learner = get_pattern_learner(project_path)
    if not learner._loaded:
        learner.load()

    if learner._patterns:
        # Re-score with learned patterns included
        scores: dict[TaskType, int] = {}

        # Start with base detection score
        scores[context.primary_task] = int(context.confidence * 6)

        # Add learned pattern scores
        for pattern in learner._patterns:
            if pattern.effectiveness >= 0.4:  # Only use effective patterns
                try:
                    if re.search(pattern.pattern, message, re.IGNORECASE):
                        task = pattern.task_type
                        boost = int(pattern.weight * (0.5 + pattern.effectiveness))
                        scores[task] = scores.get(task, 0) + boost
                except re.error:
                    pass

        # Apply negative pattern penalties
        # These reduce scores for tasks that matched due to ambiguous words
        for task in list(scores.keys()):
            penalty = learner.get_negative_penalty(message, task)
            if penalty > 0:
                scores[task] = max(0, scores[task] - penalty)
                log.debug(
                    "negative_pattern_penalty_applied",
                    task=task,
                    penalty=penalty,
                    new_score=scores[task],
                )

        # Re-determine primary if learned patterns boosted another type
        if scores:
            best_task = max(scores.items(), key=lambda x: x[1])
            if best_task[0] != context.primary_task and best_task[1] > scores.get(context.primary_task, 0):
                log.debug(
                    "learned_pattern_override",
                    original=context.primary_task,
                    new=best_task[0],
                )
                # Update context with learned pattern boost
                return DetectedContext(
                    primary_task=best_task[0],
                    secondary_tasks=[context.primary_task] + [t for t in context.secondary_tasks if t != best_task[0]][:1],
                    confidence=min(1.0, best_task[1] / 6),
                    matched_keywords=context.matched_keywords,
                )

    return context
