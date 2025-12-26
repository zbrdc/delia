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

from ..context import get_project_path
from ..file_helpers import read_file_safe
from ..routing import select_model
from .vector_store import get_vector_store

log = structlog.get_logger()


def _get_summary_index_file(project_path: Path | None = None) -> Path:
    """Get the summary index file path for a project."""
    root = get_project_path(project_path)
    return root / ".delia" / "project_summary.json"

@dataclass
class FileSummary:
    """A high-level summary of a file purpose and key exports."""
    path: str
    mtime: float
    summary: str | None = None
    exports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

class CodeSummarizer:
    """
    Generates and maintains a hierarchical index of the codebase.
    """

    def __init__(self, call_llm_fn: Any, project_path: Path | None = None):
        self.call_llm = call_llm_fn
        self.summaries: dict[str, FileSummary] = {}
        self._lock = asyncio.Lock()
        self.root = get_project_path(project_path)
        # Automatic Root Discovery: Look for .git or PROJECT_ROOT
        for parent in [self.root] + list(self.root.parents):
            if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
                self.root = parent
                break

    def _summary_file(self) -> Path:
        """Get the summary index file path for this summarizer."""
        return _get_summary_index_file(self.root)

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
        summary_file = self._summary_file()
        if summary_file.exists():
            try:
                data = json.loads(summary_file.read_text())
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
            from .constants import CODE_EXTENSIONS, should_ignore_file

            # PER-PROJECT ISOLATION: Use project-specific memories directory
            memories_dir = self.root / ".delia" / "memories"

            # Use olmo-3:7b-instruct for code summarization
            # Reliably outputs JSON (with markdown fences which we handle)
            # Note: qwen3:14b is a thinking model that doesn't respond well
            summary_model = "olmo-3:7b-instruct" if summarize else None

            # Collect files to process
            files_to_process = []

            # 1. Code Files (excluding tests)
            for path in self.root.rglob("*"):
                if should_ignore_file(path):
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

            # Sync embeddings to ChromaDB for efficient vector search
            await self._sync_to_chromadb()

            if updated_count > 0:
                log.info("project_sync_complete", updated=updated_count)
            return updated_count

    async def _sync_to_chromadb(self) -> None:
        """Sync all file embeddings to ChromaDB for efficient vector search."""
        store = get_vector_store(self.root)
        project_name = self.root.name

        synced = 0
        for rel_path, summary in self.summaries.items():
            if summary.embedding is None:
                continue

            try:
                # Ensure exports are strings (some files may have dict exports)
                exports = [
                    str(e) if not isinstance(e, str) else e
                    for e in (summary.exports or [])
                ]
                store.add_code_file(
                    file_path=rel_path,
                    content=summary.summary or f"File: {rel_path}",
                    embedding=summary.embedding,
                    summary=summary.summary,
                    exports=exports,
                    project=project_name,
                    skip_quality_check=True,  # Summaries are text, not code
                )
                synced += 1
            except Exception as e:
                log.debug("chromadb_sync_error", path=rel_path, error=str(e))

        if synced > 0:
            log.info("chromadb_code_synced", files=synced, project=project_name)

    def _build_contextual_prefix(self, rel_path: str, summary: "FileSummary") -> str:
        """Build contextual prefix for embedding (Anthropic's contextual retrieval).

        Prepends source/document/section context to help embeddings understand
        what this chunk is about in relation to the broader project.
        """
        parts = []

        # Project context
        parts.append(f"From {self.root.name} project.")

        # File type context
        suffix = Path(rel_path).suffix.lower()
        file_type_map = {
            ".py": "Python source file",
            ".ts": "TypeScript source file",
            ".tsx": "TypeScript React component",
            ".js": "JavaScript source file",
            ".jsx": "JavaScript React component",
            ".rs": "Rust source file",
            ".go": "Go source file",
            ".md": "Markdown documentation",
            ".json": "JSON configuration",
            ".yaml": "YAML configuration",
            ".yml": "YAML configuration",
        }
        file_type = file_type_map.get(suffix, "source file")
        parts.append(f"File type: {file_type}.")

        # Summary context if available
        if summary.summary:
            parts.append(f"Purpose: {summary.summary}")

        # Keywords for semantic matching
        if summary.keywords:
            keywords_str = ", ".join(summary.keywords[:8])
            parts.append(f"Concepts: {keywords_str}.")

        # Exports context if available
        if summary.exports:
            exports_str = ", ".join(summary.exports[:5])
            parts.append(f"Exports: {exports_str}.")

        return " ".join(parts)

    def _is_low_value_content(self, content: str, rel_path: str) -> bool:
        """Check if content is low-value boilerplate that should be skipped.

        Filters out:
        - License headers and boilerplate
        - Empty or nearly empty files
        - Auto-generated files
        - Lock files and binary-looking content
        """
        import re

        # Skip very small files
        if len(content.strip()) < 50:
            return True

        # Skip lock files
        if rel_path.endswith(('.lock', '-lock.json', '.lockb')):
            return True

        # Skip auto-generated markers
        auto_gen_patterns = [
            r'^# AUTO-?GENERATED',
            r'^// AUTO-?GENERATED',
            r'^/\* AUTO-?GENERATED',
            r'DO NOT EDIT',
            r'Generated by',
        ]
        for pattern in auto_gen_patterns:
            if re.search(pattern, content[:500], re.IGNORECASE | re.MULTILINE):
                return True

        # Skip if mostly license text
        license_patterns = [
            r'^MIT License',
            r'^Apache License',
            r'^BSD License',
            r'^GNU General Public License',
            r'Permission is hereby granted, free of charge',
        ]
        for pattern in license_patterns:
            if re.match(pattern, content.strip(), re.IGNORECASE):
                return True

        return False

    async def _index_file(self, rel_path: str, full_path: Path, mtime: float) -> bool:
        """Generate vector embedding for a file using configured embeddings provider."""
        from ..embeddings import get_embeddings_client

        content, error = read_file_safe(str(full_path))
        if not content:
            return False

        # Skip low-value content
        if self._is_low_value_content(content, rel_path):
            log.debug("skipping_low_value_content", file=rel_path)
            return False

        summary = self.summaries.get(rel_path, FileSummary(
            path=rel_path, mtime=mtime
        ))

        # Sanitize content for embedding API
        content = content.replace('\x00', '')
        content = content.encode('utf-8', errors='replace').decode('utf-8')

        # Build contextual prefix (Anthropic's contextual retrieval technique)
        context_prefix = self._build_contextual_prefix(rel_path, summary)

        # For large files (>5KB), extract key sections using AST
        if len(content) > 5000:
            extracted = self._extract_key_sections(rel_path, content)
            embed_content = f"{context_prefix}\n\nFile: {rel_path}\n{extracted[:1500]}"
        else:
            embed_content = f"{context_prefix}\n\nFile: {rel_path}\n{content[:1500]}"

        try:
            client = await get_embeddings_client()
            embedding = await client.embed(embed_content)
            if embedding is not None and len(embedding) > 0:
                # Convert to list if numpy array
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                summary.embedding = embedding
                summary.mtime = mtime
                self.summaries[rel_path] = summary
                return True
        except Exception as e:
            log.debug("embedding_failed", file=rel_path, error=str(e))

        return False

    async def _summarize_file(self, rel_path: str, full_path: Path, mtime: float, model: str) -> bool:
        """Use direct Ollama API to generate a one-sentence summary (parallel-safe)."""
        content, error = read_file_safe(str(full_path))
        if not content or len(content) < 50:
            return False

        prompt = f"""Analyze this source file and provide:
1. A one-sentence summary explaining WHAT problem this solves and WHY it exists (not just what it does mechanically)
2. Main exported classes/functions
3. Key conceptual keywords for search (e.g., "learning", "caching", "validation")

FILE: {rel_path}
CONTENT:
{content[:4000]}

Output ONLY valid JSON: {{"summary": "...", "exports": [...], "keywords": [...], "deps": [...]}}"""

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
                    self.summaries[rel_path].keywords = data.get("keywords", [])
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
        use_hybrid: bool = True,
        use_rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Semantic search over indexed files with hybrid + reranking.

        Uses nebnet-optimized search pipeline:
        1. Embed query with Voyage (query input_type, cached)
        2. Hybrid search: semantic (70%) + keyword (30%)
        3. Rerank top candidates with Voyage rerank-2

        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Use hybrid semantic+keyword search (default True)
            use_rerank: Apply Voyage reranking (default True)
        """
        from ..embeddings import get_embeddings_client

        # Embed the query using configured provider (Voyage AI, Ollama, etc.)
        try:
            client = await get_embeddings_client()
            query_embedding = await client.embed_query(query)
            if query_embedding is None or len(query_embedding) == 0:
                log.warning("semantic_search_embed_failed")
                return []
        except Exception as e:
            log.warning("semantic_search_query_embed_error", error=str(e))
            return []

        # Convert numpy array to list if needed
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        # Query ChromaDB with hybrid search
        store = get_vector_store(self.root)
        project_name = self.root.name

        try:
            if use_hybrid:
                # Hybrid search: semantic + keyword
                results = store.hybrid_search(
                    collection_name=store.COLLECTION_CODE,
                    query_embedding=query_embedding,
                    query_text=query,
                    n_results=top_k * 2 if use_rerank else top_k,  # Get more for reranking
                    where={"project": project_name} if project_name else None,
                    semantic_weight=0.7,
                )
            else:
                results = store.search_code(
                    query_embedding=query_embedding,
                    project=project_name,
                    n_results=top_k * 2 if use_rerank else top_k,
                )
        except Exception as e:
            log.warning("chromadb_search_error", error=str(e))
            # Fallback to in-memory search if ChromaDB fails
            return self._fallback_search(query_embedding, top_k)

        if not results:
            return []

        # Apply Voyage reranking for better result ordering
        if use_rerank and len(results) > 1:
            try:
                documents = [r.get("content", "") or r.get("metadata", {}).get("summary", "") for r in results]
                rerank_results = await client.rerank(query, documents, top_k=top_k)

                # Reorder based on rerank scores
                reranked = []
                for idx, score in rerank_results:
                    if idx < len(results):
                        r = results[idx]
                        r["rerank_score"] = score
                        r["score"] = round(score * 100, 1)  # Use rerank score as primary
                        reranked.append(r)
                results = reranked
                log.debug("rerank_applied", count=len(results))
            except Exception as e:
                log.debug("rerank_skipped", error=str(e))
                results = results[:top_k]
        else:
            results = results[:top_k]

        # Convert to expected format
        formatted = []
        for r in results:
            meta = r.get("metadata", {})
            exports_str = meta.get("exports", "")
            formatted.append({
                "path": meta.get("path", r.get("id", "")),
                "score": round(r.get("score", 0), 4),
                "summary": meta.get("summary", ""),
                "exports": exports_str.split(",") if exports_str else [],
            })

        return formatted

    def _fallback_search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """Fallback to in-memory numpy search if ChromaDB fails."""
        import numpy as np

        files_with_embeddings = [
            (path, summary)
            for path, summary in self.summaries.items()
            if summary.embedding is not None
        ]

        if not files_with_embeddings:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        scores = []
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

        scores.sort(key=lambda x: -x["score"])
        return scores[:top_k]

    def _save_index(self) -> None:
        """Save index to disk."""
        try:
            summary_file = self._summary_file()
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            data = {path: s.__dict__ for path, s in self.summaries.items()}
            temp_file = summary_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(summary_file)
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
