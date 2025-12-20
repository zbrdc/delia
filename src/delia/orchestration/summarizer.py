# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Hierarchical Codebase Summarizer.

Provides high-level 'Cliff Notes' for every file in the project.
This allows the Planner to 'see' the entire repo structure in a few
thousand tokens, enabling Gemini-class architectural awareness.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from .. import paths
from ..file_helpers import read_file_safe
from ..config import config

log = structlog.get_logger()

# File to store the project summary index
SUMMARY_INDEX_FILE = paths.DATA_DIR / "project_summary.json"

@dataclass
class FileSummary:
    """A high-level summary of a file's purpose and key exports."""
    path: str
    summary: str
    exports: list[str]
    dependencies: list[str]
    mtime: float

class CodeSummarizer:
    """
    Generates and maintains a hierarchical index of the codebase.
    """

    def __init__(self, call_llm_fn: Any):
        self.call_llm = call_llm_fn
        self.summaries: dict[str, FileSummary] = {}
        self._lock = asyncio.Lock()
        self.root = Path.cwd()

    async def initialize(self) -> None:
        """Load existing summaries from disk."""
        if SUMMARY_INDEX_FILE.exists():
            try:
                data = json.loads(SUMMARY_INDEX_FILE.read_text())
                for path, s_data in data.items():
                    self.summaries[path] = FileSummary(**s_data)
                log.info("summarizer_loaded", files=len(self.summaries))
            except Exception as e:
                log.warning("summarizer_load_failed", error=str(e))

    async def sync_project(self, force: bool = False) -> int:
        """Scan project and update summaries for changed files."""
        async with self._lock:
            updated_count = 0
            
            # Use common constants
            from .constants import CODE_EXTENSIONS, IGNORE_DIRS
            
            for path in self.root.rglob("*"):
                if any(part in IGNORE_DIRS for part in path.parts):
                    continue
                if path.suffix not in CODE_EXTENSIONS or not path.is_file():
                    continue
                    
                rel_path = str(path.relative_to(self.root))
                mtime = path.stat().st_mtime
                
                if force or rel_path not in self.summaries or self.summaries[rel_path].mtime < mtime:
                    # Summarize the file
                    success = await self._summarize_file(rel_path, path, mtime)
                    if success:
                        updated_count += 1
                        # Throttle to avoid overwhelming local backend during first sync
                        await asyncio.sleep(0.1)

            if updated_count > 0:
                self._save_index()
                log.info("project_sync_complete", updated=updated_count)
            
            return updated_count

    async def _summarize_file(self, rel_path: str, full_path: Path, mtime: float) -> bool:
        """Use the 'quick' model to generate a one-sentence summary of a file."""
        content, error = read_file_safe(str(full_path))
        if not content or len(content) < 50:
            return False

        # Build a very compact prompt for the quick model
        prompt = f"""Summarize the purpose of this file in ONE short sentence. 
Identify main classes/functions exported.

FILE: {rel_path}
CONTENT:
{content[:4000]} # Limit context for speed
"""
        
        system_prompt = "You are a technical documenter. Be extremely concise. Output JSON format: {\"summary\": \"...\", \"exports\": [\"...\"], \"deps\": [\"...\"]}"

        try:
            result = await self.call_llm(
                model=config.model_quick.default_model,
                prompt=prompt,
                system=system_prompt,
                task_type="summarize",
                temperature=0.0,
            )

            if result.get("success"):
                from .outputs import parse_structured_output
                from pydantic import BaseModel
                
                class FileSchema(BaseModel):
                    summary: str
                    exports: list[str]
                    deps: list[str]

                # Quick models might output text + JSON, so we use our robust parser
                data = parse_structured_output(result.get("response", "{{}}"), FileSchema)
                
                self.summaries[rel_path] = FileSummary(
                    path=rel_path,
                    summary=data.summary,
                    exports=data.exports,
                    dependencies=data.deps,
                    mtime=mtime
                )
                return True
        except Exception as e:
            log.debug("file_summarization_failed", file=rel_path, error=str(e))
        
        return False

    def get_project_overview(self) -> str:
        """Get a bullet-point overview of the entire project."""
        if not self.summaries:
            return "Project overview not yet generated."
            
        lines = ["## Project Structure & File Summaries"]
        # Group by directory for better hierarchy
        dirs = {}
        for path, s in sorted(self.summaries.items()):
            parent = str(Path(path).parent)
            if parent not in dirs:
                dirs[parent] = []
            dirs[parent].append(f"- `{Path(path).name}`: {s.summary}")
            
        for parent, items in sorted(dirs.items()):
            if parent == ".":
                lines.append("### Root")
            else:
                lines.append(f"### {parent}/")
            lines.extend(items)
            
        return "\n".join(lines)

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            data = {path: s.__dict__ for path, s in self.summaries.items()}
            temp_file = SUMMARY_INDEX_FILE.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(SUMMARY_INDEX_FILE)
        except Exception as e:
            log.warning("summarizer_save_error", error=str(e))

# Module-level instance holder
_summarizer: CodeSummarizer | None = None

def get_summarizer(call_llm_fn: Any = None) -> CodeSummarizer:
    """Get the global CodeSummarizer instance."""
    global _summarizer
    if _summarizer is None:
        if call_llm_fn is None:
            from ..llm import call_llm
            call_llm_fn = call_llm
        _summarizer = CodeSummarizer(call_llm_fn)
    return _summarizer
