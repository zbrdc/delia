# Copyright (C) 2023 the project owner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Typed enums for structured tool interfaces.

These enums provide explicit type safety for LLM-to-LLM communication,
replacing natural language inference with typed fields.
"""
from enum import Enum


class TaskType(str, Enum):
    """Task types with explicit semantics."""
    REVIEW = "review"
    ANALYZE = "analyze"
    GENERATE = "generate"
    SUMMARIZE = "summarize"
    CRITIQUE = "critique"
    QUICK = "quick"
    PLAN = "plan"
    THINK = "think"


class ModelTier(str, Enum):
    """Explicit model tier selection."""
    QUICK = "quick"       # 7B models - fast responses
    CODER = "coder"       # 14B code-specialized
    MOE = "moe"           # 30B+ mixture of experts
    THINKING = "thinking" # Extended reasoning models


class ContentType(str, Enum):
    """Explicit content classification."""
    CODE = "code"
    TEXT = "text"
    MIXED = "mixed"
    DOCUMENTATION = "documentation"
    LOG = "log"
    ERROR = "error"


class Language(str, Enum):
    """Programming language specification."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    RUBY = "ruby"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    SHELL = "shell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    OTHER = "other"


class BackendPreference(str, Enum):
    """Backend routing preference."""
    AUTO = "auto"         # Let Delia choose
    LOCAL = "local"       # Prefer local GPU
    REMOTE = "remote"     # Prefer remote/cloud


class Severity(str, Enum):
    """Issue severity levels for code review findings."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AnalysisType(str, Enum):
    """Types of code analysis."""
    COMPLEXITY = "complexity"
    DEPENDENCIES = "dependencies"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECTURE = "architecture"
    GENERAL = "general"


class ReasoningDepth(str, Enum):
    """Depth of reasoning for think tasks."""
    QUICK = "quick"
    NORMAL = "normal"
    DEEP = "deep"
