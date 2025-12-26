# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Coding-specific tools for Delia agents.

Provides tools optimized for software development workflows:
- Test execution with structured results
- Diff/patch application
- Git operations
- Linting/formatting

These tools build on the base builtins but are specialized for coding tasks.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import structlog

from ..types import Workspace
from .executor import validate_path
from .registry import ToolDefinition, ToolRegistry
from .builtins import get_default_tools

log = structlog.get_logger()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TestResult:
    """Structured test execution result."""
    passed: int
    failed: int
    skipped: int
    errors: int
    total: int
    duration_ms: int
    failures: list[dict]  # [{test: str, message: str, traceback: str}]
    output: str  # Raw output (truncated)
    success: bool


@dataclass
class DiffResult:
    """Result of applying a diff."""
    success: bool
    files_modified: list[str]
    hunks_applied: int
    hunks_failed: int
    message: str


@dataclass
class GitStatus:
    """Structured git status."""
    branch: str
    staged: list[str]
    modified: list[str]
    untracked: list[str]
    ahead: int
    behind: int


# =============================================================================
# Test Execution
# =============================================================================

async def run_tests(
    path: str = ".",
    pattern: str | None = None,
    verbose: bool = True,
    timeout: int = 300,
    framework: Literal["auto", "pytest", "jest", "cargo", "go"] = "auto",
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Run tests and return structured results.
    
    Auto-detects the test framework based on project files.
    
    Args:
        path: Test file or directory (relative to workspace)
        pattern: Test name pattern to match (e.g., "test_auth*")
        verbose: Show individual test results
        timeout: Maximum execution time in seconds
        framework: Test framework (auto-detect if not specified)
        workspace: Optional workspace confinement
        
    Returns:
        Structured test results with pass/fail counts and failure details
    """
    # Validate path
    valid, error = validate_path(path, workspace)
    if not valid:
        return f"Error: {error}"
    
    # Resolve path
    if workspace and not Path(path).is_absolute():
        test_path = workspace.root / path if path != "." else workspace.root
    else:
        test_path = Path(path).expanduser().resolve()
    
    # Auto-detect framework
    if framework == "auto":
        framework = _detect_test_framework(test_path)
    
    # Build command based on framework
    cmd, parser = _get_test_command(framework, str(test_path), pattern, verbose)
    
    if not cmd:
        return f"Error: Could not determine test command for framework: {framework}"
    
    # Execute tests
    cwd = str(workspace.root) if workspace else str(test_path.parent if test_path.is_file() else test_path)
    
    try:
        log.info("run_tests_starting", framework=framework, path=str(test_path), cwd=cwd)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            shell=isinstance(cmd, str),
        )
        
        # Parse results based on framework
        test_result = parser(result.stdout, result.stderr, result.returncode)
        
        # Format output
        return _format_test_result(test_result, framework)
        
    except subprocess.TimeoutExpired:
        return f"Error: Tests timed out after {timeout}s"
    except FileNotFoundError as e:
        return f"Error: Test command not found. Is {framework} installed? ({e})"
    except Exception as e:
        return f"Error running tests: {e}"


def _detect_test_framework(path: Path) -> str:
    """Detect test framework from project files."""
    search_dir = path if path.is_dir() else path.parent
    
    # Check for framework indicators
    if (search_dir / "pytest.ini").exists() or (search_dir / "pyproject.toml").exists():
        if (search_dir / "pyproject.toml").exists():
            content = (search_dir / "pyproject.toml").read_text()
            if "[tool.pytest" in content or "pytest" in content:
                return "pytest"
    
    if (search_dir / "package.json").exists():
        try:
            pkg = json.loads((search_dir / "package.json").read_text())
            if "jest" in pkg.get("devDependencies", {}) or "jest" in pkg.get("dependencies", {}):
                return "jest"
        except (json.JSONDecodeError, OSError):
            pass
    
    if (search_dir / "Cargo.toml").exists():
        return "cargo"
    
    if (search_dir / "go.mod").exists():
        return "go"
    
    # Default to pytest for Python files
    if list(search_dir.glob("**/*.py")):
        return "pytest"
    
    return "pytest"  # Default


def _get_test_command(
    framework: str,
    path: str,
    pattern: str | None,
    verbose: bool,
) -> tuple[list[str] | str | None, callable]:
    """Get test command and result parser for framework."""
    
    if framework == "pytest":
        cmd = ["python", "-m", "pytest", path, "--tb=short"]
        if verbose:
            cmd.append("-v")
        if pattern:
            cmd.extend(["-k", pattern])
        cmd.append("--color=no")
        return cmd, _parse_pytest_output
    
    elif framework == "jest":
        cmd = ["npx", "jest", path, "--no-color"]
        if verbose:
            cmd.append("--verbose")
        if pattern:
            cmd.extend(["--testNamePattern", pattern])
        cmd.append("--json")
        return cmd, _parse_jest_output
    
    elif framework == "cargo":
        cmd = ["cargo", "test"]
        if path != ".":
            cmd.append(path)
        if pattern:
            cmd.append(pattern)
        cmd.append("--no-fail-fast")
        return cmd, _parse_cargo_output
    
    elif framework == "go":
        cmd = ["go", "test", "-v"]
        if path != ".":
            cmd.append(path)
        if pattern:
            cmd.extend(["-run", pattern])
        return cmd, _parse_go_output
    
    return None, lambda *_: None


def _parse_pytest_output(stdout: str, stderr: str, returncode: int) -> TestResult:
    """Parse pytest output into structured result."""
    output = stdout + stderr
    
    # Extract summary line: "X passed, Y failed, Z skipped in Ns"
    summary_match = re.search(
        r'(\d+)\s+passed.*?(?:(\d+)\s+failed)?.*?(?:(\d+)\s+skipped)?.*?(?:(\d+)\s+error)?.*?in\s+([\d.]+)s',
        output, re.IGNORECASE
    )
    
    passed = int(summary_match.group(1)) if summary_match and summary_match.group(1) else 0
    failed = int(summary_match.group(2)) if summary_match and summary_match.group(2) else 0
    skipped = int(summary_match.group(3)) if summary_match and summary_match.group(3) else 0
    errors = int(summary_match.group(4)) if summary_match and summary_match.group(4) else 0
    duration = float(summary_match.group(5)) if summary_match and summary_match.group(5) else 0
    
    # Extract failure details
    failures = []
    failure_blocks = re.findall(
        r'FAILED\s+([\w/.:]+).*?(?=FAILED|={3,}|$)',
        output, re.DOTALL
    )
    for block in failure_blocks[:10]:  # Limit to 10 failures
        failures.append({
            "test": block.strip().split()[0] if block.strip() else "unknown",
            "message": block[:500],
        })
    
    return TestResult(
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        total=passed + failed + skipped + errors,
        duration_ms=int(duration * 1000),
        failures=failures,
        output=output[:5000],  # Truncate
        success=returncode == 0,
    )


def _parse_jest_output(stdout: str, stderr: str, returncode: int) -> TestResult:
    """Parse Jest JSON output into structured result."""
    try:
        # Jest outputs JSON with --json flag
        data = json.loads(stdout)
        
        return TestResult(
            passed=data.get("numPassedTests", 0),
            failed=data.get("numFailedTests", 0),
            skipped=data.get("numPendingTests", 0),
            errors=0,
            total=data.get("numTotalTests", 0),
            duration_ms=int(data.get("testResults", [{}])[0].get("endTime", 0) - 
                          data.get("testResults", [{}])[0].get("startTime", 0)),
            failures=[
                {"test": f["fullName"], "message": f.get("failureMessages", [""])[0][:500]}
                for tr in data.get("testResults", [])
                for f in tr.get("assertionResults", [])
                if f.get("status") == "failed"
            ][:10],
            output=stdout[:5000],
            success=data.get("success", False),
        )
    except json.JSONDecodeError:
        # Fallback to basic parsing
        return TestResult(
            passed=0, failed=0, skipped=0, errors=1,
            total=0, duration_ms=0,
            failures=[{"test": "parse_error", "message": "Could not parse Jest output"}],
            output=stdout[:5000] + stderr[:1000],
            success=False,
        )


def _parse_cargo_output(stdout: str, stderr: str, returncode: int) -> TestResult:
    """Parse cargo test output."""
    output = stdout + stderr
    
    # "test result: ok. X passed; Y failed; Z ignored"
    match = re.search(r'(\d+)\s+passed;\s+(\d+)\s+failed;\s+(\d+)\s+ignored', output)
    
    passed = int(match.group(1)) if match else 0
    failed = int(match.group(2)) if match else 0
    skipped = int(match.group(3)) if match else 0
    
    return TestResult(
        passed=passed, failed=failed, skipped=skipped, errors=0,
        total=passed + failed + skipped,
        duration_ms=0,
        failures=[],
        output=output[:5000],
        success=returncode == 0,
    )


def _parse_go_output(stdout: str, stderr: str, returncode: int) -> TestResult:
    """Parse go test output."""
    output = stdout + stderr
    
    passed = len(re.findall(r'--- PASS:', output))
    failed = len(re.findall(r'--- FAIL:', output))
    skipped = len(re.findall(r'--- SKIP:', output))
    
    return TestResult(
        passed=passed, failed=failed, skipped=skipped, errors=0,
        total=passed + failed + skipped,
        duration_ms=0,
        failures=[],
        output=output[:5000],
        success=returncode == 0,
    )


def _format_test_result(result: TestResult, framework: str) -> str:
    """Format test result for display."""
    status = "PASSED" if result.success else "FAILED"
    
    lines = [
        f"# Test Results ({framework})",
        f"**Status:** {status}",
        f"",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Passed | {result.passed} |",
        f"| Failed | {result.failed} |",
        f"| Skipped | {result.skipped} |",
        f"| Errors | {result.errors} |",
        f"| **Total** | **{result.total}** |",
        f"",
        f"**Duration:** {result.duration_ms}ms",
    ]
    
    if result.failures:
        lines.append("")
        lines.append("## Failures")
        for f in result.failures[:5]:
            lines.append(f"### {f.get('test', 'unknown')}")
            lines.append("```")
            lines.append(f.get('message', '')[:500])
            lines.append("```")
    
    return "\n".join(lines)


# =============================================================================
# Diff/Patch Application
# =============================================================================

async def apply_diff(
    diff: str,
    file_path: str | None = None,
    dry_run: bool = False,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Apply a unified diff to file(s).
    
    Args:
        diff: Unified diff content
        file_path: Target file (optional, extracted from diff if not provided)
        dry_run: If True, show what would change without applying
        workspace: Optional workspace confinement
        
    Returns:
        Result of applying the diff
    """
    # Write diff to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
        f.write(diff)
        patch_file = f.name
    
    try:
        # Build patch command
        cmd = ["patch", "-p1"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend(["--input", patch_file])
        
        # Determine working directory
        if workspace:
            cwd = str(workspace.root)
        elif file_path:
            valid, error = validate_path(file_path, workspace)
            if not valid:
                return f"Error: {error}"
            cwd = str(Path(file_path).parent.resolve())
        else:
            cwd = "."
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=cwd,
        )
        
        if result.returncode == 0:
            action = "Would apply" if dry_run else "Applied"
            return f"{action} patch successfully:\n{result.stdout}"
        else:
            return f"Patch failed:\n{result.stderr}\n{result.stdout}"
            
    except FileNotFoundError:
        # Fallback: manual application for simple diffs
        return await _apply_diff_manual(diff, file_path, dry_run, workspace)
    except Exception as e:
        return f"Error applying diff: {e}"
    finally:
        Path(patch_file).unlink(missing_ok=True)


async def _apply_diff_manual(
    diff: str,
    file_path: str | None,
    dry_run: bool,
    workspace: Workspace | None,
) -> str:
    """Manually apply a simple diff (when patch command unavailable)."""
    # Extract file path from diff header if not provided
    if not file_path:
        match = re.search(r'^[-+]{3}\s+[ab]/(.+)$', diff, re.MULTILINE)
        if match:
            file_path = match.group(1)
        else:
            return "Error: Could not determine target file from diff"
    
    # Validate and resolve path
    valid, error = validate_path(file_path, workspace)
    if not valid:
        return f"Error: {error}"
    
    if workspace and not Path(file_path).is_absolute():
        full_path = workspace.root / file_path
    else:
        full_path = Path(file_path).expanduser().resolve()
    
    if not full_path.exists():
        return f"Error: File not found: {file_path}"
    
    # Read current content
    original = full_path.read_text()
    lines = original.splitlines(keepends=True)
    
    # Parse hunks and apply
    # This is a simplified implementation for basic diffs
    hunks_applied = 0
    
    for hunk_match in re.finditer(
        r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*?\n((?:[-+ ].*\n)*)',
        diff
    ):
        old_start = int(hunk_match.group(1)) - 1
        new_content = []
        
        for line in hunk_match.group(5).splitlines(keepends=True):
            if line.startswith('+') and not line.startswith('+++'):
                new_content.append(line[1:])
            elif line.startswith(' '):
                new_content.append(line[1:])
            # Skip lines starting with '-' (deletions)
        
        # This is simplified - real patch is more complex
        hunks_applied += 1
    
    if dry_run:
        return f"Would apply {hunks_applied} hunk(s) to {file_path}"
    
    # For safety, just show what we would do
    return f"Manual diff application not fully implemented. Use 'patch' command or apply changes manually.\nHunks found: {hunks_applied}"


# =============================================================================
# Git Operations
# =============================================================================

async def git_status(
    path: str = ".",
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Get current git status with structured output.
    
    Args:
        path: Repository path
        workspace: Optional workspace confinement
        
    Returns:
        Formatted git status
    """
    if workspace and not Path(path).is_absolute():
        repo_path = workspace.root if path == "." else workspace.root / path
    else:
        repo_path = Path(path).expanduser().resolve()
    
    try:
        # Get branch
        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, cwd=str(repo_path)
        )
        branch = branch_result.stdout.strip() or "HEAD"
        
        # Get status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(repo_path)
        )
        
        staged = []
        modified = []
        untracked = []
        
        for line in status_result.stdout.splitlines():
            if len(line) < 3:
                continue
            index_status = line[0]
            work_status = line[1]
            filename = line[3:]
            
            if index_status in 'MADRC':
                staged.append(filename)
            if work_status == 'M':
                modified.append(filename)
            if index_status == '?' and work_status == '?':
                untracked.append(filename)
        
        # Get ahead/behind
        ahead_behind = subprocess.run(
            ["git", "rev-list", "--left-right", "--count", f"{branch}...@{{u}}"],
            capture_output=True, text=True, cwd=str(repo_path)
        )
        ahead, behind = 0, 0
        if ahead_behind.returncode == 0:
            parts = ahead_behind.stdout.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
        
        # Format output
        lines = [
            f"# Git Status",
            f"**Branch:** {branch}",
        ]
        
        if ahead or behind:
            lines.append(f"**Remote:** {ahead} ahead, {behind} behind")
        
        if staged:
            lines.append(f"\n## Staged ({len(staged)})")
            for f in staged[:20]:
                lines.append(f"  + {f}")
        
        if modified:
            lines.append(f"\n## Modified ({len(modified)})")
            for f in modified[:20]:
                lines.append(f"  ~ {f}")
        
        if untracked:
            lines.append(f"\n## Untracked ({len(untracked)})")
            for f in untracked[:20]:
                lines.append(f"  ? {f}")
        
        if not staged and not modified and not untracked:
            lines.append("\nWorking tree clean")
        
        return "\n".join(lines)
        
    except FileNotFoundError:
        return "Error: git command not found"
    except Exception as e:
        return f"Error getting git status: {e}"


async def git_diff(
    path: str = ".",
    staged: bool = False,
    file: str | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Show git diff for changes.
    
    Args:
        path: Repository path
        staged: Show staged changes (default: unstaged)
        file: Specific file to diff
        workspace: Optional workspace confinement
        
    Returns:
        Git diff output
    """
    if workspace and not Path(path).is_absolute():
        repo_path = workspace.root if path == "." else workspace.root / path
    else:
        repo_path = Path(path).expanduser().resolve()
    
    cmd = ["git", "diff", "--color=never"]
    if staged:
        cmd.append("--staged")
    if file:
        cmd.append("--")
        cmd.append(file)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_path),
        )
        
        if not result.stdout.strip():
            what = "staged" if staged else "unstaged"
            target = f" for {file}" if file else ""
            return f"No {what} changes{target}"
        
        output = result.stdout
        if len(output) > 10000:
            output = output[:10000] + "\n\n... [Diff truncated]"
        
        header = f"# Git Diff ({'staged' if staged else 'unstaged'})"
        if file:
            header += f" - {file}"
        
        return f"{header}\n\n```diff\n{output}\n```"
        
    except Exception as e:
        return f"Error getting git diff: {e}"


async def git_log(
    path: str = ".",
    file: str | None = None,
    n: int = 10,
    since: str | None = None,
    author: str | None = None,
    oneline: bool = False,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Show git commit history.

    Args:
        path: Repository path
        file: Filter to specific file
        n: Number of commits to show (default 10)
        since: Date filter (e.g., "2024-01-01", "1 week ago")
        author: Author filter (partial match)
        oneline: Compact one-line format
        workspace: Optional workspace confinement

    Returns:
        Formatted commit history
    """
    if workspace and not Path(path).is_absolute():
        repo_path = workspace.root if path == "." else workspace.root / path
    else:
        repo_path = Path(path).expanduser().resolve()

    if oneline:
        fmt = "--oneline"
    else:
        fmt = "--format=%H%n%an <%ae>%n%ai%n%s%n%b%n---"

    cmd = ["git", "log", fmt, f"-n{n}"]

    if since:
        cmd.append(f"--since={since}")
    if author:
        cmd.append(f"--author={author}")
    if file:
        cmd.append("--")
        cmd.append(file)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_path),
        )

        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"

        output = result.stdout.strip()
        if not output:
            return "No commits found matching criteria"

        if len(output) > 15000:
            output = output[:15000] + "\n\n... [Log truncated]"

        header = f"# Git Log ({n} commits)"
        if file:
            header += f" - {file}"

        return f"{header}\n\n{output}"

    except Exception as e:
        return f"Error getting git log: {e}"


async def git_blame(
    file: str,
    path: str = ".",
    start_line: int | None = None,
    end_line: int | None = None,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Show line-by-line authorship for a file.

    Args:
        file: File to blame
        path: Repository path
        start_line: Start line (1-indexed)
        end_line: End line (1-indexed)
        workspace: Optional workspace confinement

    Returns:
        Blame output with commit, author, date, and line content
    """
    if workspace and not Path(path).is_absolute():
        repo_path = workspace.root if path == "." else workspace.root / path
    else:
        repo_path = Path(path).expanduser().resolve()

    cmd = ["git", "blame", "--date=short"]

    if start_line and end_line:
        cmd.extend(["-L", f"{start_line},{end_line}"])
    elif start_line:
        cmd.extend(["-L", f"{start_line},"])

    cmd.append(file)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_path),
        )

        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"

        output = result.stdout.strip()
        if not output:
            return f"No blame data for {file}"

        if len(output) > 20000:
            output = output[:20000] + "\n\n... [Blame truncated]"

        header = f"# Git Blame - {file}"
        if start_line:
            header += f" (lines {start_line}-{end_line or 'end'})"

        return f"{header}\n\n```\n{output}\n```"

    except Exception as e:
        return f"Error getting git blame: {e}"


async def git_show(
    commit: str,
    file: str | None = None,
    path: str = ".",
    stat: bool = False,
    *,
    workspace: Workspace | None = None,
) -> str:
    """
    Show commit details and diff.

    Args:
        commit: Commit hash or reference (e.g., "HEAD", "abc123", "HEAD~3")
        file: Specific file to show changes for
        path: Repository path
        stat: Show diffstat instead of full diff
        workspace: Optional workspace confinement

    Returns:
        Commit details with diff
    """
    if workspace and not Path(path).is_absolute():
        repo_path = workspace.root if path == "." else workspace.root / path
    else:
        repo_path = Path(path).expanduser().resolve()

    cmd = ["git", "show", "--color=never", commit]

    if stat:
        cmd.append("--stat")
    if file:
        cmd.append("--")
        cmd.append(file)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(repo_path),
        )

        if result.returncode != 0:
            return f"Error: {result.stderr.strip()}"

        output = result.stdout.strip()
        if not output:
            return f"Commit not found: {commit}"

        if len(output) > 20000:
            output = output[:20000] + "\n\n... [Output truncated]"

        return f"# Git Show - {commit}\n\n```\n{output}\n```"

    except Exception as e:
        return f"Error getting git show: {e}"


# =============================================================================
# Registry Builder
# =============================================================================

def get_coding_tools(
    workspace: Workspace | None = None,
    allow_write: bool = False,
    allow_exec: bool = False,
) -> ToolRegistry:
    """
    Get registry with coding-specific tools.
    
    Includes all default tools plus coding tools:
    - run_tests: Execute tests with structured results
    - apply_diff: Apply unified diffs
    - git_status: Get repository status
    - git_diff: Show changes
    
    Args:
        workspace: Optional workspace confinement
        allow_write: Auto-approve write operations
        allow_exec: Auto-approve shell execution
        
    Returns:
        ToolRegistry with all coding tools
    """
    # Start with default tools
    registry = get_default_tools(workspace, allow_write, allow_exec)
    
    # Add coding-specific tools
    from functools import partial
    
    # Workspace-bound handlers
    if workspace:
        run_tests_handler = partial(run_tests, workspace=workspace)
        apply_diff_handler = partial(apply_diff, workspace=workspace)
        git_status_handler = partial(git_status, workspace=workspace)
        git_diff_handler = partial(git_diff, workspace=workspace)
        path_desc = f"Path (relative to workspace: {workspace.root})"
    else:
        run_tests_handler = run_tests
        apply_diff_handler = apply_diff
        git_status_handler = git_status
        git_diff_handler = git_diff
        path_desc = "Path (absolute or relative to current directory)"
    
    registry.register(ToolDefinition(
        name="run_tests",
        description="Run tests and get structured results. Auto-detects pytest, jest, cargo test, or go test.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Test file or directory. {path_desc}",
                    "default": "."
                },
                "pattern": {
                    "type": "string",
                    "description": "Test name pattern to match (e.g., 'test_auth*')"
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show individual test results (default: true)",
                    "default": True
                },
                "timeout": {
                    "type": "integer",
                    "description": "Max execution time in seconds (default: 300)",
                    "default": 300
                },
                "framework": {
                    "type": "string",
                    "enum": ["auto", "pytest", "jest", "cargo", "go"],
                    "description": "Test framework (default: auto-detect)",
                    "default": "auto"
                }
            }
        },
        handler=run_tests_handler,
        permission_level="exec",
        dangerous=not allow_exec,  # Running tests is exec-level
    ))
    
    registry.register(ToolDefinition(
        name="apply_diff",
        description="Apply a unified diff (patch) to files. Use for code modifications.",
        parameters={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "Unified diff content to apply"
                },
                "file_path": {
                    "type": "string",
                    "description": "Target file (optional, extracted from diff if not provided)"
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview changes without applying (default: false)",
                    "default": False
                }
            },
            "required": ["diff"]
        },
        handler=apply_diff_handler,
        permission_level="write",
        dangerous=not allow_write,
    ))
    
    registry.register(ToolDefinition(
        name="git_status",
        description="Get git repository status: branch, staged/modified/untracked files, ahead/behind remote.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Repository path. {path_desc}",
                    "default": "."
                }
            }
        },
        handler=git_status_handler,
    ))
    
    registry.register(ToolDefinition(
        name="git_diff",
        description="Show git diff for staged or unstaged changes.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Repository path. {path_desc}",
                    "default": "."
                },
                "staged": {
                    "type": "boolean",
                    "description": "Show staged changes (default: false, shows unstaged)",
                    "default": False
                },
                "file": {
                    "type": "string",
                    "description": "Specific file to diff (optional)"
                }
            }
        },
        handler=git_diff_handler,
    ))
    

    # Session Management and Project Memory
    from .orchestration import _session_compact_handler, _session_stats_handler, _project_memories_handler
    
    registry.register(ToolDefinition(
        name="session_compact",
        description="Compact a session's conversation history using LLM summarization. Reduces token count while preserving key info.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to compact"},
                "force": {"type": "boolean", "description": "Force compaction even if below threshold", "default": False},
            },
            "required": ["session_id"]
        },
        handler=_session_compact_handler,
    ))

    registry.register(ToolDefinition(
        name="session_stats",
        description="Get compaction statistics for a session, including current token usage and recommendations.",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to check"},
            },
            "required": ["session_id"]
        },
        handler=_session_stats_handler,
    ))

    registry.register(ToolDefinition(
        name="project_memories",
        description="List project memories (DELIA.md files) loaded into context. Shows instruction hierarchy and size.",
        parameters={
            "type": "object",
            "properties": {
                "reload": {"type": "boolean", "description": "Force reload of all project memories", "default": False},
            }
        },
        handler=_project_memories_handler,
    ))

    # LSP tools for code intelligence
    from .lsp import lsp_goto_definition, lsp_find_references, lsp_hover

    registry.register(ToolDefinition(
        name="lsp_goto_definition",
        description="Find the definition of a symbol at the given file position. Returns file path and line number where the symbol is defined.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "character": {"type": "integer", "description": "Character position (0-indexed)"}
            },
            "required": ["path", "line", "character"]
        },
        handler=lsp_goto_definition,
        dangerous=False,
        permission_level="read",
        requires_workspace=True,
    ))

    registry.register(ToolDefinition(
        name="lsp_find_references",
        description="Find all references to a symbol at the given file position. Returns list of locations where the symbol is used.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "character": {"type": "integer", "description": "Character position (0-indexed)"}
            },
            "required": ["path", "line", "character"]
        },
        handler=lsp_find_references,
        dangerous=False,
        permission_level="read",
        requires_workspace=True,
    ))

    registry.register(ToolDefinition(
        name="lsp_hover",
        description="Get documentation and type information for the symbol at the given position. Returns docstrings, type signatures, etc.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "line": {"type": "integer", "description": "Line number (1-indexed)"},
                "character": {"type": "integer", "description": "Character position (0-indexed)"}
            },
            "required": ["path", "line", "character"]
        },
        handler=lsp_hover,
        dangerous=False,
        permission_level="read",
        requires_workspace=True,
    ))

    # Sandboxed execution tools (optional - requires llm-sandbox)
    from ..sandbox import is_sandbox_available
    if is_sandbox_available():
        from .sandbox_tools import (
            shell_exec_sandboxed,
            code_execute,
            _format_sandbox_result,
        )

        registry.register(ToolDefinition(
            name="shell_exec_sandboxed",
            description="Execute shell commands in an isolated Docker container. Safe for untrusted commands.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["command"]
            },
            handler=shell_exec_sandboxed,
            permission_level="exec",
            dangerous=False,  # Safe because it's sandboxed
        ))

        registry.register(ToolDefinition(
            name="code_execute",
            description="Execute code in an isolated Docker container. Supports Python, JavaScript, Java, C++, Go, R.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code to execute"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "java", "cpp", "go", "r"],
                        "description": "Programming language (default: python)",
                        "default": "python"
                    },
                    "libraries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Libraries to install before execution (e.g., ['numpy', 'pandas'])"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Execution timeout in seconds (default: 30)",
                        "default": 30
                    }
                },
                "required": ["code"]
            },
            handler=code_execute,
            permission_level="exec",
            dangerous=False,  # Safe because it's sandboxed
        ))

    return registry

