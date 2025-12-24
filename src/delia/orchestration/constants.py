# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Common constants for Orchestration.
"""

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".cpp", ".h",
    ".java", ".kt", ".swift", ".rb", ".php", ".sh", ".json", ".toml", ".yaml", ".yml",
    ".md", ".txt",  # Documentation files
}

# Directories to ignore
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv",
    ".next", "target", "vendor", ".idea", ".vscode",
    "data", "outputs", ".delia", "logs", "cache",  # Runtime data dirs
    "tests", "test", "__tests__", "spec", "fixtures",  # Test directories
}

# File patterns to ignore (matched against filename)
IGNORE_FILE_PATTERNS = {
    "test_",      # Python test files: test_foo.py
    "_test.",     # Python test files: foo_test.py
    ".test.",     # JS/TS test files: foo.test.ts
    ".spec.",     # JS/TS spec files: foo.spec.ts
    "conftest",   # Pytest fixtures
}


def should_ignore_file(path) -> bool:
    """Check if a file path should be ignored during scanning."""
    from pathlib import Path
    p = Path(path) if not hasattr(path, 'parts') else path

    # Check directory
    if any(part in IGNORE_DIRS for part in p.parts):
        return True

    # Check file patterns
    if any(pattern in p.name for pattern in IGNORE_FILE_PATTERNS):
        return True

    return False
