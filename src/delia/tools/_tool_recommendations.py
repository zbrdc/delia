# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Tool recommendations based on detected task context.

Provides intelligent tool suggestions to help LLMs discover
the right tools for each task type.
"""

from __future__ import annotations

from typing import Any


# Memory tools - useful for ALL tasks
MEMORY_TOOLS = [
    {"tool": "memory(action='list')", "use": "Check existing project knowledge"},
    {"tool": "memory(action='read')", "use": "Load relevant documented insights"},
    {"tool": "memory(action='write')", "use": "Persist learnings for future sessions"},
]

# Framework workflow tools - ensure proper Delia methodology
FRAMEWORK_TOOLS = [
    {"tool": "think_about_task_adherence", "use": "Verify alignment before modifying code"},
    {"tool": "think_about_completion", "use": "Checklist before finishing task"},
    {"tool": "complete_task", "use": "Record outcome and close learning loop"},
]

# Core file tools - useful for most tasks
FILE_TOOLS = [
    {"tool": "read_file", "use": "Read file contents with line numbers"},
    {"tool": "search_for_pattern", "use": "Find code patterns with regex"},
    {"tool": "find_file", "use": "Locate files by glob pattern"},
]

# Semantic search tools - always available
SEMANTIC_TOOLS = [
    {"tool": "semantic_search", "use": "Find code by meaning, not just text"},
    {"tool": "get_related_files", "use": "Find files related via dependencies"},
    {"tool": "codebase_graph", "use": "Query project dependency structure"},
]

TASK_TOOLS: dict[str, list[dict[str, str]]] = {
    "coding": [
        # Navigation (understand before editing)
        {"tool": "lsp_get_symbols", "use": "Map file structure before editing"},
        {"tool": "lsp_find_references", "use": "Find all usages before changing"},
        {"tool": "lsp_goto_definition", "use": "Jump to function/class source"},
        {"tool": "lsp_hover", "use": "Get type info and documentation"},
        {"tool": "lsp_find_symbol_semantic", "use": "Search code by meaning"},
        # Editing (safe modifications)
        {"tool": "lsp_rename_symbol", "use": "Rename across entire codebase"},
        {"tool": "lsp_replace_symbol_body", "use": "Replace function/class body"},
        {"tool": "edit_file", "use": "Make targeted text replacements"},
        # Refactoring
        {"tool": "lsp_extract_method", "use": "Extract code into new function"},
        {"tool": "lsp_move_symbol", "use": "Move symbol to different file"},
        {"tool": "lsp_organize_imports", "use": "Clean up imports after changes"},
    ],
    "debugging": [
        {"tool": "lsp_find_references", "use": "Trace where code is called"},
        {"tool": "lsp_goto_definition", "use": "Jump to source of errors"},
        {"tool": "lsp_find_referencing_symbols", "use": "Find calling functions"},
        {"tool": "lsp_hover", "use": "Check types at error location"},
        {"tool": "lsp_get_hot_files", "use": "See recently modified files"},
        {"tool": "search_for_pattern", "use": "Find error patterns in code"},
        {"tool": "git_log", "use": "See recent changes that may have caused bug"},
    ],
    "testing": [
        {"tool": "lsp_get_symbols", "use": "See what functions need tests"},
        {"tool": "lsp_find_references", "use": "Find existing test patterns"},
        {"tool": "lsp_find_symbol", "use": "Locate test files/fixtures"},
        {"tool": "search_for_pattern", "use": "Find test patterns to follow"},
        {"tool": "lsp_get_dependencies", "use": "Understand what to mock"},
    ],
    "architecture": [
        {"tool": "lsp_get_symbols", "use": "Survey module structure"},
        {"tool": "lsp_get_dependencies", "use": "Visualize file relationships"},
        {"tool": "lsp_find_symbol_semantic", "use": "Search by architectural concept"},
        {"tool": "lsp_get_hot_files", "use": "Identify active development areas"},
        {"tool": "list_dir", "use": "Explore directory structure"},
    ],
    "git": [
        {"tool": "git_log", "use": "View commit history"},
        {"tool": "git_show", "use": "See commit details and diff"},
        {"tool": "git_blame", "use": "Find who changed what"},
        {"tool": "lsp_get_hot_files", "use": "See recently modified files"},
    ],
    "project": [
        {"tool": "lsp_find_symbol", "use": "Find classes/functions by name"},
        {"tool": "lsp_find_symbol_semantic", "use": "Search by concept/meaning"},
        {"tool": "list_dir", "use": "Explore project structure"},
        {"tool": "find_file", "use": "Locate specific files"},
        {"tool": "lsp_get_hot_files", "use": "See active development areas"},
    ],
    "security": [
        {"tool": "lsp_find_references", "use": "Trace sensitive data flow"},
        {"tool": "lsp_find_referencing_symbols", "use": "Find callers of sensitive code"},
        {"tool": "search_for_pattern", "use": "Find security patterns (auth, crypto)"},
        {"tool": "lsp_get_dependencies", "use": "Check what accesses sensitive modules"},
    ],
    "api": [
        {"tool": "lsp_get_symbols", "use": "See endpoint structure"},
        {"tool": "lsp_find_references", "use": "Find endpoint usages"},
        {"tool": "lsp_find_symbol", "use": "Locate route handlers"},
        {"tool": "lsp_hover", "use": "Check request/response types"},
        {"tool": "search_for_pattern", "use": "Find API patterns"},
    ],
    "deployment": [
        {"tool": "find_file", "use": "Locate config files"},
        {"tool": "search_for_pattern", "use": "Find environment variables"},
        {"tool": "list_dir", "use": "Check deployment structure"},
    ],
    "performance": [
        {"tool": "lsp_find_references", "use": "Find hot path callers"},
        {"tool": "lsp_get_dependencies", "use": "Analyze dependency chains"},
        {"tool": "lsp_find_referencing_symbols", "use": "Find performance-critical code"},
        {"tool": "lsp_get_hot_files", "use": "Focus on recently changed files"},
    ],
}


def get_recommended_tools(context: Any) -> list[dict[str, str]]:
    """Get recommended tools for a detected task context.

    Args:
        context: Detected context with primary_task and secondary_tasks

    Returns:
        Ordered list of tool recommendations with use cases
    """
    # Start with task-specific tools (most relevant first)
    recommended = list(TASK_TOOLS.get(context.primary_task, []))

    # Add tools from secondary tasks (avoid duplicates)
    for secondary in context.secondary_tasks[:2]:
        for tool in TASK_TOOLS.get(secondary, []):
            if tool not in recommended:
                recommended.append(tool)

    # Add semantic search tools (always useful)
    recommended.extend(SEMANTIC_TOOLS)

    # Add core file tools
    recommended.extend(FILE_TOOLS)

    # Add framework workflow tools
    recommended.extend(FRAMEWORK_TOOLS)

    # Add memory tools
    recommended.extend(MEMORY_TOOLS)

    return recommended
