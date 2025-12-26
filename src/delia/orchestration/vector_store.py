# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
ChromaDB-backed vector store for Delia's semantic search.

Provides unified storage for:
- Code file embeddings (semantic code search)
- Playbook bullets (context-aware retrieval)
- Memories (persistent project knowledge)
- Profiles (best-practice retrieval)

Based on nebnet-mcp's DocStore pattern, adapted for Delia's multi-collection needs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog

from ..context import get_project_path

log = structlog.get_logger()

# Lazy imports for ChromaDB to avoid startup penalty
_chromadb = None
_ChromaSettings = None


def _get_chromadb():
    """Lazy import of chromadb."""
    global _chromadb, _ChromaSettings
    if _chromadb is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        _chromadb = chromadb
        _ChromaSettings = ChromaSettings
    return _chromadb, _ChromaSettings


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for hybrid search."""
    # Programming terms to preserve case for
    preserve_case = {
        "async", "await", "def", "class", "import", "from", "return",
        "TypeError", "ValueError", "RuntimeError", "Exception",
        "FastAPI", "Pydantic", "SQLAlchemy", "pytest", "asyncio",
        "MCP", "LSP", "LLM", "API", "CLI", "SDK",
    }

    # Find preserved terms first
    preserved = []
    for term in preserve_case:
        if term.lower() in text.lower():
            preserved.append(term)

    # Extract other words (3+ chars, alphanumeric)
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', text)

    # Filter stopwords
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "have", "are",
        "was", "were", "been", "being", "will", "would", "could", "should",
        "can", "may", "might", "must", "shall", "need", "use", "using",
        "how", "what", "when", "where", "which", "who", "why", "does", "not",
    }

    keywords = preserved + [w for w in words if w.lower() not in stopwords and w not in preserved]

    # Return unique, limited to top 10
    seen = set()
    unique = []
    for kw in keywords:
        lower = kw.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(kw)

    return unique[:10]


class VectorStore:
    """ChromaDB-backed vector store with multiple collections.

    Per-project storage: Each project gets its own ChromaDB at <project>/.delia/chroma/

    Collections:
    - code: File summaries and code snippets for semantic search
    - playbook: Playbook bullets for context retrieval
    - memories: Persistent project knowledge
    - profiles: Best-practice templates
    """

    # Collection names
    COLLECTION_CODE = "delia_code"
    COLLECTION_PLAYBOOK = "delia_playbook"
    COLLECTION_MEMORIES = "delia_memories"
    COLLECTION_PROFILES = "delia_profiles"

    def __init__(self, project_path: Path | str | None = None):
        """Initialize the vector store for a specific project.

        Args:
            project_path: Project root directory. ChromaDB stored at <project>/.delia/chroma/
                         Defaults to project context or current working directory.
        """
        self.project_path = get_project_path(project_path)
        self.persist_dir = self.project_path / ".delia" / "chroma"
        self._client = None
        self._collections: dict[str, Any] = {}

    @property
    def client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            chromadb, ChromaSettings = _get_chromadb()
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            log.info("Initialized ChromaDB", path=str(self.persist_dir))
        return self._client

    def get_collection(self, name: str):
        """Get or create a collection by name."""
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            log.debug("Using collection", name=name, count=self._collections[name].count())
        return self._collections[name]

    # =========================================================================
    # CONTEXTUAL HEADERS (LLM-friendly result formatting)
    # =========================================================================

    def _build_context(
        self,
        collection_name: str,
        doc_id: str,
        metadata: dict[str, Any],
    ) -> str:
        """Build LLM-friendly contextual breadcrumb for search results.

        Adapted from nebnet-mcp optimization: provides hierarchical context
        so LLMs understand where content comes from without redundant metadata.

        Examples:
            "Playbook > coding > Use pathlib over os.path"
            "Code > src/auth.py > login()"
            "Memory > architecture > Database design decisions"
        """
        parts = []

        if collection_name == self.COLLECTION_PLAYBOOK:
            parts.append("Playbook")
            if task_type := metadata.get("task_type"):
                parts.append(task_type)
            # Add truncated content hint
            if doc_id and len(doc_id) > 20:
                parts.append(doc_id[:20] + "...")
            else:
                parts.append(doc_id or "bullet")

        elif collection_name == self.COLLECTION_CODE:
            parts.append("Code")
            if path := metadata.get("path"):
                # Use relative path
                parts.append(path.split("/")[-1] if "/" in path else path)
            if exports := metadata.get("exports"):
                # Show first export as hint
                first_export = exports.split(",")[0] if "," in exports else exports
                if first_export:
                    parts.append(first_export)

        elif collection_name == self.COLLECTION_MEMORIES:
            parts.append("Memory")
            if name := metadata.get("name"):
                parts.append(name)

        elif collection_name == self.COLLECTION_PROFILES:
            parts.append("Profile")
            if name := metadata.get("name"):
                parts.append(name)

        else:
            parts.append(collection_name.replace("delia_", "").title())
            parts.append(doc_id[:30] if doc_id else "item")

        return " > ".join(parts)

    # =========================================================================
    # GENERIC OPERATIONS
    # =========================================================================

    def add_items(
        self,
        collection_name: str,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> int:
        """Add items to a collection.

        Args:
            collection_name: Target collection
            ids: Unique IDs for each item
            embeddings: Vector embeddings
            documents: Text content
            metadatas: Metadata dicts

        Returns:
            Number of items added
        """
        if not ids:
            return 0

        collection = self.get_collection(collection_name)

        # ChromaDB has a max batch size of 5461
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        log.info("Added items to store", collection=collection_name, count=len(ids))
        return len(ids)

    def search(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search in a collection.

        Args:
            collection_name: Target collection
            query_embedding: Query vector
            n_results: Max results
            where: Optional filter dict

        Returns:
            List of results with id, content, metadata, score
        """
        collection = self.get_collection(collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results with contextual headers (nebnet-mcp optimization)
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = max(0.0, min(1.0, 1 - distance))

                formatted.append({
                    "id": doc_id,
                    "context": self._build_context(collection_name, doc_id, meta),
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": meta,
                    "score": round(similarity * 100, 1),  # Normalized 0-100 scale
                    "raw_score": similarity,  # Keep original for internal use
                })

        return formatted

    def hybrid_search(
        self,
        collection_name: str,
        query_embedding: list[float],
        query_text: str,
        n_results: int = 10,
        where: dict | None = None,
        semantic_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Combine semantic and keyword search.

        Args:
            collection_name: Target collection
            query_embedding: Query vector for semantic search
            query_text: Original query for keyword extraction
            n_results: Max results
            where: Optional filter dict
            semantic_weight: Weight for semantic vs keyword (0-1)

        Returns:
            Merged and ranked results
        """
        collection = self.get_collection(collection_name)

        # Get more candidates from semantic search
        semantic_limit = n_results * 3
        semantic_results = self.search(
            collection_name,
            query_embedding,
            n_results=semantic_limit,
            where=where,
        )

        # Keyword search
        keywords = extract_keywords(query_text)
        keyword_results: dict[str, dict] = {}

        for keyword in keywords[:5]:
            try:
                results = collection.get(
                    where=where,
                    where_document={"$contains": keyword.lower()},
                    include=["documents", "metadatas"],
                    limit=n_results,
                )

                if results["ids"]:
                    for i, doc_id in enumerate(results["ids"]):
                        if doc_id not in keyword_results:
                            meta = results["metadatas"][i] if results["metadatas"] else {}
                            keyword_results[doc_id] = {
                                "id": doc_id,
                                "content": results["documents"][i] if results["documents"] else "",
                                "metadata": meta,
                                "keyword_matches": 1,
                                "matched_keywords": [keyword],
                            }
                        else:
                            keyword_results[doc_id]["keyword_matches"] += 1
                            keyword_results[doc_id]["matched_keywords"].append(keyword)
            except Exception as e:
                log.debug("Keyword search failed", keyword=keyword, error=str(e))
                continue

        # Merge results with hybrid scoring
        merged: dict[str, dict] = {}

        # Add semantic results (already have context from search())
        for i, r in enumerate(semantic_results):
            doc_id = r["id"]
            raw_score = r.get("raw_score", r.get("score", 0) / 100)
            semantic_score = min(1.0, max(0.0, raw_score))
            merged[doc_id] = {
                **r,
                "semantic_score": semantic_score,
                "keyword_score": 0.0,
                "hybrid_score": semantic_score * semantic_weight,
            }

        # Merge keyword results
        keyword_weight = 1.0 - semantic_weight
        max_matches = max((r["keyword_matches"] for r in keyword_results.values()), default=1)

        for doc_id, r in keyword_results.items():
            keyword_score = r["keyword_matches"] / max_matches

            if doc_id in merged:
                merged[doc_id]["keyword_score"] = keyword_score
                merged[doc_id]["matched_keywords"] = r.get("matched_keywords", [])
                merged[doc_id]["hybrid_score"] += keyword_score * keyword_weight
            else:
                # Add context for keyword-only results
                meta = r.get("metadata", {})
                merged[doc_id] = {
                    **r,
                    "context": self._build_context(collection_name, doc_id, meta),
                    "semantic_score": 0.0,
                    "keyword_score": keyword_score,
                    "hybrid_score": keyword_score * keyword_weight,
                    "score": round(keyword_score * keyword_weight * 100, 1),
                    "raw_score": keyword_score * keyword_weight,
                }

        # Sort by hybrid score
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )

        return sorted_results[:n_results]

    def delete_by_filter(self, collection_name: str, where: dict) -> int:
        """Delete items matching a filter."""
        collection = self.get_collection(collection_name)

        results = collection.get(where=where, include=[])
        if results["ids"]:
            collection.delete(ids=results["ids"])
            log.info("Deleted items", collection=collection_name, count=len(results["ids"]))
            return len(results["ids"])
        return 0

    def get_stats(self, collection_name: str | None = None) -> dict[str, Any]:
        """Get statistics about the store."""
        if collection_name:
            collection = self.get_collection(collection_name)
            return {
                "collection": collection_name,
                "count": collection.count(),
            }

        # Get stats for all collections
        stats = {
            "persist_dir": str(self.persist_dir),
            "collections": {},
        }
        for name in [self.COLLECTION_CODE, self.COLLECTION_PLAYBOOK,
                     self.COLLECTION_MEMORIES, self.COLLECTION_PROFILES]:
            try:
                collection = self.get_collection(name)
                stats["collections"][name] = collection.count()
            except Exception:
                stats["collections"][name] = 0

        return stats

    def clear_collection(self, collection_name: str) -> None:
        """Clear all data from a collection."""
        try:
            self.client.delete_collection(collection_name)
            self._collections.pop(collection_name, None)
            log.info("Cleared collection", name=collection_name)
        except Exception as e:
            log.debug("Collection clear failed", name=collection_name, error=str(e))

    # =========================================================================
    # PLAYBOOK-SPECIFIC OPERATIONS
    # =========================================================================

    def add_playbook_bullet(
        self,
        bullet_id: str,
        content: str,
        embedding: list[float],
        task_type: str,
        project: str | None = None,
        utility_score: float = 0.5,
    ) -> None:
        """Add a playbook bullet to the store."""
        self.add_items(
            self.COLLECTION_PLAYBOOK,
            ids=[bullet_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "task_type": task_type,
                "project": project or "global",
                "utility_score": utility_score,
            }],
        )

    def search_playbook(
        self,
        query_embedding: list[float],
        task_type: str | None = None,
        project: str | None = None,
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search playbook bullets semantically.

        Args:
            query_embedding: Query vector
            task_type: Filter by task type (coding, testing, etc.)
            project: Filter by project (or include global)
            n_results: Max results
        """
        where = None
        conditions = []

        if task_type:
            conditions.append({"task_type": task_type})

        if project:
            # Include both project-specific and global bullets
            conditions.append({
                "$or": [
                    {"project": project},
                    {"project": "global"},
                ]
            })

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        return self.search(self.COLLECTION_PLAYBOOK, query_embedding, n_results, where)

    # =========================================================================
    # MEMORY-SPECIFIC OPERATIONS
    # =========================================================================

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        name: str,
        project: str | None = None,
    ) -> None:
        """Add a memory to the store."""
        self.add_items(
            self.COLLECTION_MEMORIES,
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "name": name,
                "project": project or "global",
            }],
        )

    def search_memories(
        self,
        query_embedding: list[float],
        project: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search memories semantically."""
        where = None
        if project:
            where = {
                "$or": [
                    {"project": project},
                    {"project": "global"},
                ]
            }
        return self.search(self.COLLECTION_MEMORIES, query_embedding, n_results, where)

    # =========================================================================
    # PROFILE-SPECIFIC OPERATIONS
    # =========================================================================

    def add_profile(
        self,
        profile_id: str,
        content: str,
        embedding: list[float],
        name: str,
        project: str | None = None,
    ) -> None:
        """Add a reference profile to the store."""
        self.add_items(
            self.COLLECTION_PROFILES,
            ids=[profile_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "name": name,
                "project": project or "global",
            }],
        )

    def search_profiles(
        self,
        query_embedding: list[float],
        project: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search profiles semantically."""
        where = None
        if project:
            where = {
                "$or": [
                    {"project": project},
                    {"project": "global"},
                ]
            }
        return self.search(self.COLLECTION_PROFILES, query_embedding, n_results, where)

    # =========================================================================
    # CODE-SPECIFIC OPERATIONS
    # =========================================================================

    def _is_low_value_code(self, file_path: str, content: str) -> bool:
        """Check if code file is low-value and should be skipped.

        Adapted from nebnet-mcp quality filtering. Skips:
        - Empty or tiny files (<50 chars)
        - Auto-generated files
        - Files with only imports/docstrings
        - Configuration boilerplate
        """
        # Skip empty/tiny files (30 chars ~ 1 meaningful line)
        if len(content.strip()) < 30:
            return True

        # Skip auto-generated patterns
        skip_patterns = [
            "__pycache__",
            ".pyc",
            "node_modules",
            ".min.js",
            ".min.css",
            "package-lock.json",
            "yarn.lock",
            ".egg-info",
            "dist/",
            "build/",
        ]
        for pattern in skip_patterns:
            if pattern in file_path:
                return True

        # Skip files with only imports, docstrings, and boilerplate
        # Check for actual code patterns (functions, classes, assignments)
        code_patterns = [
            "def ", "class ", "async def ",  # Python
            "function ", "const ", "let ", "var ",  # JavaScript
            "fn ", "impl ", "struct ", "enum ",  # Rust
            "func ", "type ", "interface ",  # Go
        ]

        has_code = False
        for pattern in code_patterns:
            if pattern in content:
                has_code = True
                break

        # Also check for meaningful assignments/operations
        if not has_code:
            lines = content.strip().split("\n")
            meaningful_lines = 0
            for line in lines:
                stripped = line.strip()
                # Skip empty, comments, imports only
                if not stripped:
                    continue
                if stripped.startswith(("#", "//", "/*", "*", '"""', "'''")):
                    continue
                if stripped.startswith(("import ", "from ", "require(", "export {")):
                    continue
                # Count as meaningful if not a skip pattern
                meaningful_lines += 1
                if meaningful_lines >= 5:
                    has_code = True
                    break

        return not has_code

    def add_code_file(
        self,
        file_path: str,
        content: str,
        embedding: list[float],
        summary: str | None = None,
        exports: list[str] | None = None,
        project: str | None = None,
        skip_quality_check: bool = False,
    ) -> bool:
        """Add a code file summary to the store.

        Returns:
            True if file was indexed, False if skipped due to quality filter.
        """
        # Quality filtering (nebnet-mcp optimization)
        if not skip_quality_check and self._is_low_value_code(file_path, content):
            log.debug("Skipped low-value code file", path=file_path)
            return False

        self.add_items(
            self.COLLECTION_CODE,
            ids=[file_path],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{
                "path": file_path,
                "summary": summary or "",
                "exports": ",".join(exports or []),
                "project": project or "unknown",
            }],
        )
        return True

    def search_code(
        self,
        query_embedding: list[float],
        project: str | None = None,
        n_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Search code files semantically."""
        where = None
        if project:
            where = {"project": project}
        return self.search(self.COLLECTION_CODE, query_embedding, n_results, where)


# Per-project store instances (keyed by resolved project path)
_stores: dict[str, VectorStore] = {}


def get_vector_store(project_path: Path | str | None = None) -> VectorStore:
    """Get the VectorStore instance for a specific project.

    Args:
        project_path: Project root directory. Defaults to project context.

    Returns:
        VectorStore instance for that project (cached).
    """
    resolved = get_project_path(project_path)
    key = str(resolved.resolve())

    if key not in _stores:
        _stores[key] = VectorStore(resolved)
        log.debug("created_vector_store", project=key)

    return _stores[key]


def reset_vector_store(project_path: Path | str | None = None) -> None:
    """Reset the VectorStore instance for a project.

    Args:
        project_path: Project to reset. If None, resets all.
    """
    global _stores
    if project_path is None:
        _stores.clear()
    else:
        key = str(Path(project_path).resolve())
        _stores.pop(key, None)
