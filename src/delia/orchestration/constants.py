# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Common constants for Orchestration.
"""

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".go", ".c", ".cpp", ".h", 
    ".java", ".kt", ".swift", ".rb", ".php", ".sh", ".json", ".toml", ".yaml", ".yml"
}

# Directories to ignore
IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", "dist", "build", ".venv", "venv", 
    ".next", "target", "vendor", ".idea", ".vscode"
}
