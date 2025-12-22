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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from ..file_helpers import read_file_safe
from ..routing import select_model

log = structlog.get_logger()

# Project-specific summary index (.delia/ in CWD)
SUMMARY_INDEX_FILE = Path.cwd() / ".delia" / "project_summary.json"

@dataclass
class FileSummary:
    """A high-level summary of a file purpose and key exports."""
    path: str
    mtime: float
    summary: str | None = None
    exports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

class CodeSummarizer:
    """
    Generates and maintains a hierarchical index of the codebase.
    """

    def __init__(self, call_llm_fn: Any):
        self.call_llm = call_llm_fn
        self.summaries: dict[str, FileSummary] = {}
        self._lock = asyncio.Lock()
        self.root = Path.cwd()
        # Automatic Root Discovery: Look for .git or PROJECT_ROOT
        for parent in [self.root] + list(self.root.parents):
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                self.root = parent
                break

    def _extract_key_sections(self, rel_path: str, content: str) -> str:
        """
        Extract key sections from a large file for better embeddings.

        For Python: Uses AST to extract docstring + class/function signatures
        For other languages: Falls back to regex-based extraction
        """
        if rel_path.endswith('.py'):
            return self._extract_python_signatures(content)
        elif rel_path.endswith(('.ts', '.tsx', '.js', '.jsx')):
            return self._extract_js_signatures(content)
        else:
            # Fallback: first 1500 chars
            return content[:1500]

    def _extract_python_signatures(self, content: str) -> str:
        """Extract module docstring and all class/function signatures using AST."""
        import ast
        parts = []

        try:
            tree = ast.parse(content)

            # Module docstring
            docstring = ast.get_docstring(tree)
            if docstring:
                parts.append(f'"""{docstring[:300]}"""')

            # Extract imports (first 10)
            imports = []
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            if imports:
                parts.append(f"# Imports: {', '.join(imports[:10])}")

            # Extract class and function signatures
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    bases = ', '.join(
                        ast.unparse(b) if hasattr(ast, 'unparse') else str(b)
                        for b in node.bases[:3]
                    )
                    class_doc = ast.get_docstring(node)
                    doc_hint = f'  """{class_doc[:100]}..."""' if class_doc else ""
                    parts.append(f"class {node.name}({bases}):{doc_hint}")

                    # Add method signatures
                    for item in node.body[:10]:  # First 10 methods
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            async_prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                            try:
                                args = ast.unparse(item.args) if hasattr(ast, 'unparse') else "..."
                            except Exception:
                                args = "..."
                            parts.append(f"    {async_prefix}def {item.name}({args}): ...")

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                    try:
                        args = ast.unparse(node.args) if hasattr(ast, 'unparse') else "..."
                    except Exception:
                        args = "..."
                    func_doc = ast.get_docstring(node)
                    doc_hint = f'  """{func_doc[:80]}..."""' if func_doc else ""
                    parts.append(f"{async_prefix}def {node.name}({args}):{doc_hint}")

        except SyntaxError:
            # Fall back to regex for invalid Python
            return self._extract_signatures_regex(content)

        return "\n".join(parts) if parts else content[:1500]

    def _extract_js_signatures(self, content: str) -> str:
        """Extract function/class signatures from JS/TS using regex."""
        import re
        parts = []

        # Extract imports
        imports = re.findall(r"^import\s+.+?from\s+['\"](.+?)['\"]", content, re.MULTILINE)
        if imports:
            parts.append(f"// Imports: {', '.join(imports[:10])}")

        # Extract class definitions
        classes = re.findall(r"^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", content, re.MULTILINE)
        for name, base in classes:
            parts.append(f"class {name}{' extends ' + base if base else ''} {{ }}")

        # Extract function definitions
        funcs = re.findall(
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)",
            content, re.MULTILINE
        )
        for name, args in funcs[:20]:
            parts.append(f"function {name}({args}) {{ }}")

        # Extract arrow functions assigned to exports
        arrows = re.findall(
            r"^(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            content, re.MULTILINE
        )
        for name in arrows[:10]:
            parts.append(f"const {name} = () => {{ }}")

        return "\n".join(parts) if parts else content[:1500]

    def _extract_signatures_regex(self, content: str) -> str:
        """Fallback regex-based extraction for any language."""
        import re
        parts = []

        # Generic class pattern
        classes = re.findall(r"^(?:class|struct|interface)\s+(\w+)", content, re.MULTILINE)
        for name in classes[:10]:
            parts.append(f"class {name}")

        # Generic function pattern
        funcs = re.findall(r"^(?:def|func|fn|function|pub fn|async fn)\s+(\w+)", content, re.MULTILINE)
        for name in funcs[:20]:
            parts.append(f"function {name}()")

        return "\n".join(parts) if parts else content[:1500]

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
                
        # Trigger background sync (non-blocking for fast startup)
        asyncio.create_task(self.sync_project())
        self._initialized = True



    async def sync_project(self, force: bool = False, summarize: bool = False, parallel: int = 4) -> int:
        """Scan project AND memories and update symbols/embeddings for all files."""
        async with self._lock:
            from .constants import CODE_EXTENSIONS, IGNORE_DIRS

            # PER-PROJECT ISOLATION: Use project-specific memories directory
            memories_dir = Path.cwd() / ".delia" / "memories"

            # Use olmo-3:7b-instruct for code summarization
            # Reliably outputs JSON (with markdown fences which we handle)
            # Note: qwen3:14b is a thinking model that doesn't respond well
            summary_model = "olmo-3:7b-instruct" if summarize else None

            # Collect files to process
            files_to_process = []
            
            # 1. Code Files
            for path in self.root.rglob("*"):
                if any(part in IGNORE_DIRS for part in path.parts):
                    continue
                if path.suffix not in CODE_EXTENSIONS or not path.is_file():
                    continue

                rel_path = str(path.relative_to(self.root))
                mtime = path.stat().st_mtime

                if force or rel_path not in self.summaries or self.summaries[rel_path].mtime < mtime:
                    files_to_process.append((rel_path, path, mtime))

            # 2. Memory Files (New for P3.1)
            if memories_dir.exists():
                for path in memories_dir.glob("*.md"):
                    rel_path = f"memory://{path.stem}"
                    mtime = path.stat().st_mtime
                    if force or rel_path not in self.summaries or self.summaries[rel_path].mtime < mtime:
                        files_to_process.append((rel_path, path, mtime))

            if not files_to_process:
                log.info("project_sync_no_changes")
                return 0

            log.info("project_sync_starting", files=len(files_to_process), parallel=parallel)

            # Process files in parallel batches
            updated_count = 0
            semaphore = asyncio.Semaphore(parallel)

            async def process_file(rel_path: str, full_path: Path, mtime: float) -> bool:
                async with semaphore:
                    log.info("indexing_file", path=rel_path)

                    # Step 1: Fast Path - Vector Embedding
                    success = await self._index_file(rel_path, full_path, mtime)

                    # Step 2: LLM Summarization (parallel with semaphore limit)
                    if success and summary_model:
                        await self._summarize_file(rel_path, full_path, mtime, summary_model)

                    return success

            # Run all files concurrently (semaphore limits actual parallelism)
            results = await asyncio.gather(
                *[process_file(rel, path, mt) for rel, path, mt in files_to_process],
                return_exceptions=True
            )

            updated_count = sum(1 for r in results if r is True)

            self._save_index()
            if updated_count > 0:
                log.info("project_sync_complete", updated=updated_count)
            return updated_count



    async def _index_file(self, rel_path: str, full_path: Path, mtime: float) -> bool:
        """Generate vector embedding for a file using direct Ollama API (CPU-bound)."""
        content, error = read_file_safe(str(full_path))
        if not content:
            return False

        summary = self.summaries.get(rel_path, FileSummary(
            path=rel_path, mtime=mtime
        ))

        # Sanitize content for embedding API
        # Remove null bytes and normalize unicode
        content = content.replace('\x00', '')
        content = content.encode('utf-8', errors='replace').decode('utf-8')

        # For large files (>5KB), extract key sections using AST
        # This gives better semantic representation than raw truncation
        if len(content) > 5000:
            extracted = self._extract_key_sections(rel_path, content)
            embed_content = f"File: {rel_path}\n{extracted[:1500]}"
        else:
            # Truncate to ~512 tokens (~1500 chars) for embedding model limit
            embed_content = f"File: {rel_path}\n{content[:1500]}"

        # Direct HTTP call to Ollama to bypass all provider/queue logic
        import httpx
        from ..backend_manager import backend_manager

        # Find an Ollama backend for embeddings (prefer local)
        url = "http://localhost:11434"  # Default fallback
        for backend in backend_manager.backends.values():
            if backend.enabled and backend.provider == "ollama" and backend.type == "local":
                url = backend.url.rstrip("/")
                break
        else:
            # Try any ollama backend
            for backend in backend_manager.backends.values():
                if backend.enabled and backend.provider == "ollama":
                    url = backend.url.rstrip("/")
                    break

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{url}/api/embed",
                        json={
                            "model": "mxbai-embed-large",
                            "input": embed_content
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if "embeddings" in data and len(data["embeddings"]) > 0:
                            summary.embedding = data["embeddings"][0]
                            summary.mtime = mtime
                            self.summaries[rel_path] = summary
                            return True
                    elif response.status_code == 400:
                        # Bad request - likely model not found or input issue
                        try:
                            err_data = response.json()
                            err_msg = err_data.get("error", "unknown")
                        except Exception:
                            err_msg = response.text[:200]
                        log.debug(
                            "ollama_embedding_bad_request",
                            file=rel_path,
                            error=err_msg,
                            input_len=len(embed_content),
                        )
                        # Don't retry 400s - they won't succeed
                        return False
                    elif response.status_code == 500 and attempt < 2:
                        log.debug("ollama_retry", attempt=attempt+1, file=rel_path)
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    else:
                        log.debug("ollama_embedding_failed", status=response.status_code, file=rel_path)
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                log.debug("direct_embedding_exception", error=str(e), file=rel_path)

        return False

    async def _summarize_file(self, rel_path: str, full_path: Path, mtime: float, model: str) -> bool:
        """Use direct Ollama API to generate a one-sentence summary (parallel-safe)."""
        content, error = read_file_safe(str(full_path))
        if not content or len(content) < 50:
            return False

        prompt = f"""Summarize the purpose of this file in ONE short sentence.
Identify main classes/functions exported.

FILE: {rel_path}
CONTENT:
{content[:4000]}

Output ONLY valid JSON: {{"summary": "...", "exports": [...], "deps": [...]}}"""

        # Direct HTTP call to Ollama - bypasses queue for parallel efficiency
        import httpx
        from ..backend_manager import backend_manager

        # Find an Ollama backend (prefer local)
        url = "http://localhost:11434"  # Default fallback
        for backend in backend_manager.backends.values():
            if backend.enabled and backend.provider == "ollama" and backend.type == "local":
                url = backend.url.rstrip("/")
                break
        else:
            for backend in backend_manager.backends.values():
                if backend.enabled and backend.provider == "ollama":
                    url = backend.url.rstrip("/")
                    break

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1, "num_predict": 512}
                    }
                )
                if response.status_code != 200:
                    log.debug("summarize_http_failed", file=rel_path, status=response.status_code)
                    return False

                result_text = response.json().get("response", "")

                # Parse JSON from response - handle thinking models
                import json as json_lib
                import re

                # Step 1: Strip thinking blocks (<think>...</think>)
                result_text = re.sub(r'<think>.*?</think>', '', result_text, flags=re.DOTALL)

                # Step 2: Extract JSON from markdown code fences
                fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, flags=re.DOTALL)
                if fence_match:
                    result_text = fence_match.group(1)

                # Step 3: Try to parse as JSON
                try:
                    data = json_lib.loads(result_text.strip())
                except json_lib.JSONDecodeError:
                    # Step 4: Find any JSON object in the text
                    # Use a more robust pattern that handles nested structures
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text)
                    if json_match:
                        try:
                            data = json_lib.loads(json_match.group())
                        except json_lib.JSONDecodeError:
                            log.debug("summarize_json_parse_failed", file=rel_path, raw=result_text[:200])
                            return False
                    else:
                        log.debug("summarize_json_not_found", file=rel_path, raw=result_text[:200])
                        return False

                if rel_path in self.summaries:
                    self.summaries[rel_path].summary = data.get("summary", "")
                    self.summaries[rel_path].exports = data.get("exports", [])
                    self.summaries[rel_path].dependencies = data.get("deps", [])

                return True
        except Exception as e:
            log.debug("summarize_exception", file=rel_path, error=str(e))

        return False

    def get_project_overview(self) -> str:
        """Get a bullet-point overview of the entire project."""
        if not self.summaries:
            return ""  # Return empty string so nothing is injected

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

    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over indexed files using cosine similarity.

        Math: O(n × d) where n=files, d=1024 dimensions
        For 500 files: ~0.04ms raw compute, ~1ms with Python overhead

        Args:
            query: Natural language query (e.g., "authentication logic")
            top_k: Number of results to return

        Returns:
            List of {path, score, summary, exports} sorted by relevance
        """
        import numpy as np

        # Get files with embeddings
        files_with_embeddings = [
            (path, summary)
            for path, summary in self.summaries.items()
            if summary.embedding is not None
        ]

        if not files_with_embeddings:
            log.warning("semantic_search_no_embeddings")
            return []

        # Embed the query
        import httpx
        from ..backend_manager import backend_manager

        # Find an Ollama backend for embeddings (prefer local)
        url = "http://localhost:11434"  # Default fallback
        for backend in backend_manager.backends.values():
            if backend.enabled and backend.provider == "ollama" and backend.type == "local":
                url = backend.url.rstrip("/")
                break
        else:
            for backend in backend_manager.backends.values():
                if backend.enabled and backend.provider == "ollama":
                    url = backend.url.rstrip("/")
                    break

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{url}/api/embed",
                    json={"model": "mxbai-embed-large", "input": query}
                )
                if response.status_code != 200:
                    log.warning("semantic_search_embed_failed", status=response.status_code)
                    return []
                data = response.json()
                if "embeddings" not in data or not data["embeddings"]:
                    return []
                query_vec = np.array(data["embeddings"][0], dtype=np.float32)
        except Exception as e:
            log.warning("semantic_search_query_embed_error", error=str(e))
            return []

        # Compute cosine similarity against all file embeddings
        # This is the O(n × d) operation - ~0.04ms for 500 files
        scores = []
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        for path, summary in files_with_embeddings:
            file_vec = np.array(summary.embedding, dtype=np.float32)
            file_norm = np.linalg.norm(file_vec)
            if file_norm == 0:
                continue
            similarity = float(np.dot(query_vec, file_vec) / (query_norm * file_norm))
            scores.append({
                "path": path,
                "score": round(similarity, 4),
                "summary": summary.summary,
                "exports": summary.exports,
            })

        # Sort by score descending, return top_k
        scores.sort(key=lambda x: -x["score"])
        return scores[:top_k]

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
