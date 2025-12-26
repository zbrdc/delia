# Copyright (C) 2024 Delia Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""Embeddings client for semantic content classification.

Uses a local embeddings model (e.g., nomic-embed) to classify content
semantically rather than via regex patterns.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np
import structlog

# Load ~/.delia/.env for API keys (Voyage, etc.)
_delia_env = Path.home() / ".delia" / ".env"
if _delia_env.exists():
    for line in _delia_env.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# Optional dependencies for local fallback
SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

from .paths import DATA_DIR

log = structlog.get_logger()

# Default embeddings settings
EMBEDDINGS_ENDPOINT = "/v1/embeddings"

# SINGLE embedding model for the entire system (fast, small, good quality)
# all-MiniLM-L6-v2: 80MB, ~50ms inference, 384 dimensions
SHARED_EMBEDDING_MODEL = os.getenv("DELIA_EMBEDDING_MODEL", "mxbai-embed-large")

# Ordered list of preferred models for auto-detection (fallback only)
PREFERRED_MODELS = ["mxbai-embed-large", "nomic-embed-text",
    "mxbai-embed-large",
    "BAAI/bge-m3",  # Default: fast, small, good quality
    "BAAI/bge-m3",
    "mxbai-embed-large",
    "BAAI/bge-large-en-v1.5",
    "nomic-embed-text",
]

# Shared model singleton - loaded once, used everywhere
_shared_model: "SentenceTransformer | None" = None
_shared_model_lock = asyncio.Lock() if asyncio else None


def get_shared_model() -> "SentenceTransformer":
    """
    Get the shared embedding model (singleton).

    Loads model on first call, reuses for all subsequent calls.
    This ensures we only load ONE model across the entire system.
    """
    global _shared_model
    if _shared_model is not None:
        return _shared_model

    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not installed")

    log.info("loading_shared_embedding_model", model=SHARED_EMBEDDING_MODEL)
    _shared_model = SentenceTransformer(SHARED_EMBEDDING_MODEL)

    # Keep on CPU to preserve VRAM for inference models
    use_gpu = os.getenv("DELIA_EMBEDDINGS_GPU", "").lower() in ("1", "true", "yes")
    if not use_gpu:
        _shared_model = _shared_model.to("cpu")
        log.info("shared_embedding_model_on_cpu", model=SHARED_EMBEDDING_MODEL)

    return _shared_model


async def get_shared_model_async() -> "SentenceTransformer":
    """Async version that loads model in thread pool."""
    global _shared_model
    if _shared_model is not None:
        return _shared_model

    async with _shared_model_lock:
        if _shared_model is not None:
            return _shared_model
        _shared_model = await asyncio.to_thread(get_shared_model)
    return _shared_model

# Reference embeddings file
REFERENCE_EMBEDDINGS_FILE = "reference_embeddings.json"


@dataclass
class ContentClassification:
    """Result of semantic content classification."""

    category: str  # "code", "planning", "simple", "debug", "refactor"
    confidence: float  # 0.0 - 1.0
    subcategory: str | None = None  # e.g., "python", "typescript"
    reasoning: str = ""
    all_scores: dict[str, float] = field(default_factory=dict)


class EmbeddingsProvider(Protocol):
    """Protocol for embeddings providers."""
    async def embed(self, text: str) -> np.ndarray: ...
    async def health_check(self) -> bool: ...
    async def close(self) -> None: ...


class ExternalEmbeddingsProvider:
    """Client for external embeddings API (e.g., llama.cpp, Ollama)."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self.detected_model: str | None = os.getenv("DELIA_EMBEDDING_MODEL")

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def detect_model(self) -> str | None:
        """Probe the backend to see which preferred models are available."""
        if self.detected_model:
            return self.detected_model
            
        if not self.base_url:
            return None

        client = await self._get_client()
        try:
            # Try Ollama tags endpoint
            response = await client.get(f"{self.base_url}/api/tags", timeout=2.0)
            if response.status_code == 200:
                models = [m["name"].split(":")[0] for m in response.json().get("models", [])]
                for preferred in PREFERRED_MODELS:
                    if any(preferred in m for m in models):
                        self.detected_model = preferred
                        log.info("embeddings_model_autodetected", model=preferred)
                        return preferred
        except Exception:
            pass
            
        return PREFERRED_MODELS[-1] # Fallback to MiniLM

    async def embed(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        if not self.base_url:
            raise RuntimeError("External embeddings URL not configured")

        client = await self._get_client()
        model = await self.detect_model() or "mxbai-embed-large"

        try:
            # Try OpenAI compatible endpoint
            response = await client.post(
                f"{self.base_url}{EMBEDDINGS_ENDPOINT}",
                json={"input": text, "model": model},
            )
            
            if response.status_code != 200:
                # Fallback for Ollama native /api/embeddings
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"prompt": text, "model": model},
                )
                response.raise_for_status()
                embedding = response.json()["embedding"]
            else:
                # Handle OpenAI-compatible response format
                data = response.json()
                embedding = data["data"][0]["embedding"]
                
            return np.array(embedding, dtype=np.float32)

        except Exception as e:
            log.debug("external_embeddings_failed", error=str(e), model=model)
            raise

    async def health_check(self) -> bool:
        """Check if embeddings service is available."""
        if not self.base_url:
            return False
            
        client = await self._get_client()
        try:
            response = await client.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            try:
                # Fallback check for Ollama or other providers
                response = await client.get(self.base_url, timeout=2.0)
                return response.status_code == 200
            except Exception:
                return False


class LocalEmbeddingsProvider:
    """Provider using the SHARED sentence-transformers model (loaded once, used everywhere)."""

    def __init__(self, model_name: str | None = None, force_cpu: bool = True):
        # Note: model_name is ignored - we always use the shared model
        self.force_cpu = force_cpu

    async def _get_model(self) -> Any:
        """Get the shared model (loaded once for entire system)."""
        return await get_shared_model_async()

    async def embed(self, text: str) -> np.ndarray:
        model = await self._get_model()
        # Run inference in thread
        embedding = await asyncio.to_thread(model.encode, text)
        return np.array(embedding, dtype=np.float32)

    async def health_check(self) -> bool:
        return SENTENCE_TRANSFORMERS_AVAILABLE

    async def close(self) -> None:
        # Don't clear the shared model - it's used by other components
        pass


class VoyageEmbeddingsProvider:
    """Provider using Voyage AI API for embeddings.

    Voyage AI offers high-quality embeddings optimized for code and documentation.
    Uses voyage-code-3 model by default (1024 dimensions).

    Features:
    - Batch embedding for efficiency (batch size 96)
    - Query caching for repeated searches
    - Retries with exponential backoff

    Set DELIA_VOYAGE_API_KEY environment variable to enable.
    """

    VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
    DEFAULT_MODEL = "voyage-code-3"
    BATCH_SIZE = 96  # Voyage limit is ~120 texts or 320k tokens per batch
    MAX_RETRIES = 3

    # Query cache (class-level for sharing across instances)
    _query_cache: dict[str, list[float]] = {}
    _cache_max_size: int = 1000

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        # Check multiple env var names for Voyage API key
        self.api_key = api_key or os.getenv("DELIA_VOYAGE_API_KEY") or os.getenv("NEBNET_VOYAGE_API_KEY") or os.getenv("VOYAGE_API_KEY")
        self.model = model or os.getenv("DELIA_VOYAGE_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _sanitize_text(self, text: str) -> str:
        """Sanitize text for embedding - limit length and clean control chars."""
        import re
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text[:8000].strip()

    async def embed(self, text: str, input_type: str = "document") -> np.ndarray:
        """Generate embedding using Voyage AI.

        Args:
            text: Text to embed
            input_type: "query" for search queries, "document" for content
        """
        embeddings = await self.embed_batch([text], input_type=input_type)
        return embeddings[0]

    async def embed_batch(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[np.ndarray]:
        """Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            input_type: "query" for search queries, "document" for content

        Returns:
            List of embedding arrays in same order as input
        """
        if not self.api_key:
            raise RuntimeError("DELIA_VOYAGE_API_KEY not set")

        if not texts:
            return []

        client = await self._get_client()
        sanitized_texts = [self._sanitize_text(t) for t in texts]
        all_embeddings: list[np.ndarray] = []

        # Process in batches
        for i in range(0, len(sanitized_texts), self.BATCH_SIZE):
            batch = sanitized_texts[i:i + self.BATCH_SIZE]

            for attempt in range(self.MAX_RETRIES):
                try:
                    response = await client.post(
                        self.VOYAGE_API_URL,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "input": batch,
                            "input_type": input_type,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    batch_embeddings = [
                        np.array(item["embedding"], dtype=np.float32)
                        for item in data["data"]
                    ]
                    all_embeddings.extend(batch_embeddings)
                    log.debug(
                        "voyage_batch_complete",
                        batch=i // self.BATCH_SIZE + 1,
                        total=(len(sanitized_texts) - 1) // self.BATCH_SIZE + 1,
                        count=len(batch),
                    )
                    break
                except Exception as e:
                    log.warning(
                        "voyage_batch_failed",
                        attempt=attempt + 1,
                        error=str(e),
                        batch_size=len(batch),
                    )
                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    else:
                        # All retries failed - return zeros for this batch
                        log.error("voyage_batch_exhausted", batch_start=i)
                        all_embeddings.extend([
                            np.zeros(1024, dtype=np.float32) for _ in batch
                        ])

        return all_embeddings

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with caching (optimized for search matching)."""
        cache_key = f"{self.model}:query:{text[:100]}"

        # Check cache
        if cache_key in self._query_cache:
            return np.array(self._query_cache[cache_key], dtype=np.float32)

        embedding = await self.embed(text, input_type="query")

        # Cache (with size limit)
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        self._query_cache[cache_key] = embedding.tolist()

        return embedding

    async def embed_document(self, text: str) -> np.ndarray:
        """Embed a document (optimized for content storage)."""
        return await self.embed(text, input_type="document")

    async def embed_documents_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple documents efficiently."""
        return await self.embed_batch(texts, input_type="document")

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents using Voyage rerank-2 API.

        Returns list of (original_index, relevance_score) tuples, sorted by score descending.
        This provides a second-pass refinement after initial semantic search.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        if not self.api_key:
            raise RuntimeError("DELIA_VOYAGE_API_KEY not set")

        if not documents:
            return []

        client = await self._get_client()
        try:
            response = await client.post(
                "https://api.voyageai.com/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "rerank-2",
                    "query": query,
                    "documents": documents,
                    "top_k": min(top_k, len(documents)),
                },
            )
            response.raise_for_status()
            data = response.json()
            # Returns list of {index, relevance_score}
            return [(item["index"], item["relevance_score"]) for item in data["data"]]
        except Exception as e:
            log.warning("voyage_rerank_failed", error=str(e))
            # Fallback: return original order with dummy scores
            return [(i, 1.0 - i * 0.01) for i in range(min(top_k, len(documents)))]

    async def health_check(self) -> bool:
        """Check if Voyage AI is available."""
        if not self.api_key:
            return False

        # Try a minimal embedding to verify API key works
        try:
            await self.embed("test")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class HybridEmbeddingsClient:
    """
    Hybrid embeddings client with tiered fallback:
    1. Voyage AI (if DELIA_VOYAGE_API_KEY set) - highest quality
    2. External API (Ollama/llama.cpp)
    3. Local sentence-transformers

    Implements the same interface as the old EmbeddingsClient for compatibility.
    """

    def __init__(
        self,
        base_url: str | None = None,
        local_model: str | None = None,
        timeout: float = 30.0,
        force_cpu: bool | None = None,  # None = auto from env
    ):
        # Default to CPU to preserve VRAM for primary inference models
        # Override with DELIA_EMBEDDINGS_GPU=1 to use GPU
        if force_cpu is None:
            use_gpu = os.getenv("DELIA_EMBEDDINGS_GPU", "").lower() in ("1", "true", "yes")
            force_cpu = not use_gpu  # Default: CPU (force_cpu=True)

        # Provider priority: Voyage AI > External (Ollama) > Local
        self.voyage_provider = VoyageEmbeddingsProvider(timeout=timeout)
        self.external_provider = ExternalEmbeddingsProvider(base_url, timeout)
        self.local_provider = LocalEmbeddingsProvider(local_model, force_cpu=force_cpu)
        self.active_provider: EmbeddingsProvider | None = None
        self._init_lock = asyncio.Lock()
        self._force_cpu = force_cpu

    async def initialize(self) -> bool:
        """Initialize and determine the best provider."""
        async with self._init_lock:
            if self.active_provider:
                return True

            # Priority 1: Try Voyage AI if API key is set
            if self.voyage_provider.api_key:
                if await self.voyage_provider.health_check():
                    self.active_provider = self.voyage_provider
                    log.info("embeddings_client_using_voyage_ai", model=self.voyage_provider.model)
                    return True
                else:
                    log.debug("voyage_ai_health_check_failed")

            # Priority 2: Try external API (detect from active backend if not set)
            if not self.external_provider.base_url:
                from .backend_manager import backend_manager
                active = backend_manager.get_active_backend()
                if active:
                    self.external_provider.base_url = active.url.rstrip("/")
                    log.debug("embeddings_url_detected_from_backend", url=active.url)

            if await self.external_provider.health_check():
                self.active_provider = self.external_provider
                await self.external_provider.detect_model()
                log.info("embeddings_client_using_external_api", model=self.external_provider.detected_model)
                return True

            # Priority 3: Fall back to local sentence-transformers
            if await self.local_provider.health_check():
                self.active_provider = self.local_provider
                log.info("embeddings_client_using_local_fallback", model=SHARED_EMBEDDING_MODEL)
                return True

            log.warning("embeddings_client_no_provider_available")
            return False

    async def embed(self, text: str) -> np.ndarray:
        """Get embedding vector for text."""
        if not self.active_provider:
            if not await self.initialize():
                raise RuntimeError("No embeddings provider available")

        try:
            return await self.active_provider.embed(text)
        except Exception as e:
            # Try fallback providers
            log.warning("embeddings_failed_trying_fallback", error=str(e), provider=type(self.active_provider).__name__)

            # If Voyage failed, try external
            if self.active_provider == self.voyage_provider:
                if await self.external_provider.health_check():
                    self.active_provider = self.external_provider
                    log.info("switched_to_external_fallback")
                    return await self.active_provider.embed(text)

            # If external failed, try local
            if self.active_provider in (self.voyage_provider, self.external_provider):
                if await self.local_provider.health_check():
                    self.active_provider = self.local_provider
                    log.info("switched_to_local_fallback")
                    return await self.active_provider.embed(text)
            raise

    async def embed_batch(
        self,
        texts: list[str],
        input_type: str = "document",
    ) -> list[np.ndarray]:
        """Batch embed multiple texts efficiently.

        Uses Voyage AI's native batch API when available, otherwise
        falls back to sequential embedding.

        Args:
            texts: List of texts to embed
            input_type: "document" for storage, "query" for search

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        if not self.active_provider:
            if not await self.initialize():
                raise RuntimeError("No embeddings provider available")

        # Use Voyage AI's efficient batch API if available
        if self.active_provider == self.voyage_provider:
            return await self.voyage_provider.embed_batch(texts, input_type)

        # Fallback: sequential embedding for other providers
        embeddings = []
        for text in texts:
            embedding = await self.active_provider.embed(text)
            embeddings.append(embedding)
        return embeddings

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with caching (optimized for search).

        Uses Voyage AI's query cache when available for repeated searches.
        Queries use input_type="query" which is optimized for retrieval matching.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector (cached if Voyage and seen before)
        """
        if not self.active_provider:
            if not await self.initialize():
                raise RuntimeError("No embeddings provider available")

        # Use Voyage's cached query embedding if available
        if self.active_provider == self.voyage_provider:
            return await self.voyage_provider.embed_query(text)

        # Fallback: regular embedding for other providers
        return await self.active_provider.embed(text)

    async def health_check(self) -> bool:
        """Check if any provider is available."""
        if self.active_provider:
            return await self.active_provider.health_check()
        return await self.initialize()

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Rerank documents using Voyage rerank-2 API if available.

        Provides a second-pass refinement after initial semantic search,
        significantly improving result quality.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by relevance.
            Falls back to original ordering if Voyage is unavailable.
        """
        if not self.active_provider:
            await self.initialize()

        # Only Voyage supports reranking
        if self.active_provider == self.voyage_provider:
            return await self.voyage_provider.rerank(query, documents, top_k)

        # Fallback: return original order with distance-based dummy scores
        return [(i, 1.0 - i * 0.05) for i in range(min(top_k, len(documents)))]

    async def close(self) -> None:
        """Clean up resources."""
        await self.voyage_provider.close()
        await self.external_provider.close()
        await self.local_provider.close()


# Alias for backward compatibility with RAG modules
EmbeddingsClient = HybridEmbeddingsClient


# =========================================================================
# SINGLETON FACTORY
# =========================================================================

_embeddings_client: HybridEmbeddingsClient | None = None
_embeddings_client_lock = asyncio.Lock()


async def get_embeddings_client() -> HybridEmbeddingsClient:
    """Get or create the global embeddings client singleton.

    Returns an initialized HybridEmbeddingsClient that's reused across
    all operations, avoiding the overhead of creating new clients.

    Returns:
        Initialized HybridEmbeddingsClient instance
    """
    global _embeddings_client

    if _embeddings_client is not None:
        return _embeddings_client

    async with _embeddings_client_lock:
        # Double-check after acquiring lock
        if _embeddings_client is not None:
            return _embeddings_client

        client = HybridEmbeddingsClient()
        await client.initialize()
        _embeddings_client = client
        log.info("embeddings_client_singleton_created")
        return _embeddings_client


def reset_embeddings_client() -> None:
    """Reset the singleton (for testing or reconfiguration)."""
    global _embeddings_client
    _embeddings_client = None


# Reference content for each category - these get embedded once and cached
REFERENCE_CONTENT = {
    # Code samples - actual code patterns
    "code_python": '''def process_data(items: list[dict]) -> pd.DataFrame:
    """Process raw items into a DataFrame."""
    filtered = [x for x in items if x.get("valid")]
    df = pd.DataFrame(filtered)
    df["timestamp"] = pd.to_datetime(df["created_at"])
    return df.sort_values("timestamp")''',

    "code_typescript": '''export async function fetchUserData(userId: string): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
    throw new ApiError(`Failed to fetch user: ${response.status}`);
  }
  const data = await response.json();
  return validateUser(data);
}''',

    "code_rust": '''impl<T: Clone + Send + Sync> Cache<T> {
    pub async fn get_or_insert<F, Fut>(&self, key: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        if let Some(value) = self.store.read().await.get(key) {
            return Ok(value.clone());
        }
        let value = f().await?;
        self.store.write().await.insert(key.to_string(), value.clone());
        Ok(value)
    }
}''',

    "code_go": '''func (s *Server) HandleRequest(ctx context.Context, req *Request) (*Response, error) {
    span, ctx := opentracing.StartSpanFromContext(ctx, "HandleRequest")
    defer span.Finish()

    if err := s.validator.Validate(req); err != nil {
        return nil, status.Error(codes.InvalidArgument, err.Error())
    }
    return s.processor.Process(ctx, req)
}''',

    # Planning/Architecture content
    "planning_architecture": '''Design a microservices architecture for an e-commerce platform that handles:
- User authentication with JWT and refresh tokens
- Product catalog with full-text search
- Shopping cart with session persistence
- Order processing with event-driven updates
- Payment integration with multiple providers
Consider scalability, fault tolerance, and data consistency requirements.''',

    "planning_system": '''We need to implement a caching strategy for our API. Current issues:
1. Database queries taking 200-500ms on average
2. Cache invalidation causing stale data
3. Memory pressure on Redis cluster
Propose a multi-tier caching approach with TTLs and invalidation strategy.''',

    "planning_migration": '''Plan the migration from our monolithic Django application to a distributed system:
- Identify service boundaries
- Design data ownership and API contracts
- Create a phased rollout strategy
- Handle backward compatibility during transition''',

    # Debugging/fixing content
    "debug_error": '''Users are reporting 500 errors when submitting the checkout form.
Stack trace shows NullPointerException in PaymentProcessor.processCard().
The error started after yesterday's deployment. Need to identify root cause and fix.''',

    "debug_performance": '''The API endpoint /api/reports/generate is timing out after 30 seconds.
Profiling shows 80% of time spent in database queries.
Query plan shows full table scan on orders table (10M rows).
Need to optimize the query and add appropriate indexes.''',

    # Refactoring content
    "refactor_cleanup": '''Refactor this authentication module to:
- Extract the JWT logic into a separate service
- Replace callbacks with async/await
- Add proper error types instead of string errors
- Implement the repository pattern for user storage''',

    "refactor_patterns": '''Convert this class-based React component to a functional component with hooks.
Replace componentDidMount with useEffect, this.state with useState,
and extract the data fetching logic into a custom hook.''',

    # Simple queries/questions
    "simple_explanation": '''What does this function do? Can you explain the logic?''',

    "simple_howto": '''How do I add a new route to the Express server?''',

    "simple_lookup": '''What's the syntax for a Python list comprehension with a condition?''',

    # Review/analysis content
    "review_code": '''Please review this pull request for:
- Potential bugs or edge cases
- Performance implications
- Security vulnerabilities
- Code style and best practices
- Test coverage gaps''',

    "review_security": '''Audit this authentication implementation for security issues:
- SQL injection vulnerabilities
- XSS attack vectors
- CSRF protection
- Session management flaws
- Password storage practices''',
}

# Category mapping for routing decisions
CATEGORY_TO_TIER = {
    "code_python": "coder",
    "code_typescript": "coder",
    "code_rust": "coder",
    "code_go": "coder",
    "planning_architecture": "moe",
    "planning_system": "moe",
    "planning_migration": "moe",
    "debug_error": "coder",
    "debug_performance": "coder",
    "refactor_cleanup": "coder",
    "refactor_patterns": "coder",
    "simple_explanation": "quick",
    "simple_howto": "quick",
    "simple_lookup": "quick",
    "review_code": "coder",
    "review_security": "moe",
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticRouter:
    """Semantic content classification using embeddings with hybrid fallback."""

    def __init__(
        self,
        embeddings_url: str | None = None,
        local_model: str | None = None,
        cache_references: bool = True,
    ):
        self.client = HybridEmbeddingsClient(embeddings_url, local_model)
        self.reference_embeddings: dict[str, np.ndarray] = {}
        self._initialized = False
        self._cache_references = cache_references
        self._init_lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize reference embeddings (lazy, called once)."""
        async with self._init_lock:
            if self._initialized:
                return True

            # Try to load cached embeddings first
            if self._cache_references and self._load_cached_embeddings():
                # Still check if we can get a provider
                if await self.client.initialize():
                    self._initialized = True
                    log.info("semantic_router_loaded_cache", categories=len(self.reference_embeddings))
                    return True

            # Check if embeddings service is available
            if not await self.client.health_check():
                log.warning("semantic_router_embeddings_unavailable")
                return False

            # Generate reference embeddings
            try:
                log.info("semantic_router_generating_references", count=len(REFERENCE_CONTENT))

                for category, text in REFERENCE_CONTENT.items():
                    embedding = await self.client.embed(text)
                    self.reference_embeddings[category] = embedding
                    log.debug("reference_embedding_generated", category=category)

                # Cache for future use
                if self._cache_references:
                    self._save_cached_embeddings()

                self._initialized = True
                log.info("semantic_router_initialized", categories=len(self.reference_embeddings))
                return True

            except Exception as e:
                log.error("semantic_router_init_failed", error=str(e))
                return False

    def _get_cache_path(self) -> Path:
        return DATA_DIR / REFERENCE_EMBEDDINGS_FILE

    def _load_cached_embeddings(self) -> bool:
        """Load reference embeddings from cache."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False

        try:
            with open(cache_path) as f:
                data = json.load(f)

            # Verify all categories are present
            if not all(cat in data for cat in REFERENCE_CONTENT.keys()):
                log.info("semantic_router_cache_outdated")
                return False

            self.reference_embeddings = {
                k: np.array(v, dtype=np.float32) for k, v in data.items()
            }
            return True

        except Exception as e:
            log.warning("semantic_router_cache_load_failed", error=str(e))
            return False

    def _save_cached_embeddings(self) -> None:
        """Save reference embeddings to cache."""
        cache_path = self._get_cache_path()
        try:
            data = {k: v.tolist() for k, v in self.reference_embeddings.items()}
            with open(cache_path, "w") as f:
                json.dump(data, f)
            log.debug("semantic_router_cache_saved")
        except Exception as e:
            log.warning("semantic_router_cache_save_failed", error=str(e))

    async def classify(self, content: str, max_length: int = 4000) -> ContentClassification:
        """Classify content semantically.

        Args:
            content: The content to classify
            max_length: Maximum content length to embed (truncates if longer)

        Returns:
            ContentClassification with category, confidence, and scores
        """
        if not self._initialized:
            if not await self.initialize():
                # Fallback to basic classification
                return ContentClassification(
                    category="simple_explanation",
                    confidence=0.0,
                    reasoning="Embeddings unavailable, defaulting to simple",
                )

        # Truncate content if needed
        if len(content) > max_length:
            content = content[:max_length]

        try:
            # Get embedding for input content
            content_embedding = await self.client.embed(content)

            # Compute similarity to all reference embeddings
            scores: dict[str, float] = {}
            for category, ref_embedding in self.reference_embeddings.items():
                scores[category] = cosine_similarity(content_embedding, ref_embedding)

            # Find best match
            best_category = max(scores, key=scores.get)
            best_score = scores[best_category]

            # Determine subcategory (e.g., language for code)
            subcategory = None
            if best_category.startswith("code_"):
                subcategory = best_category.split("_")[1]

            # Generate reasoning
            top_3 = sorted(scores.items(), key=lambda x: -x[1])[:3]
            reasoning = f"Top matches: {', '.join(f'{k}={v:.2f}' for k, v in top_3)}"

            return ContentClassification(
                category=best_category,
                confidence=best_score,
                subcategory=subcategory,
                reasoning=reasoning,
                all_scores=scores,
            )

        except Exception as e:
            log.warning("semantic_classify_failed", error=str(e))
            return ContentClassification(
                category="simple_explanation",
                confidence=0.0,
                reasoning=f"Classification failed: {e}",
            )

    async def get_recommended_tier(self, content: str) -> tuple[str, float, str]:
        """Get recommended model tier for content.

        Returns:
            (tier, confidence, reasoning) - tier is "quick", "coder", or "moe"
        """
        classification = await self.classify(content)

        tier = CATEGORY_TO_TIER.get(classification.category, "quick")

        return tier, classification.confidence, classification.reasoning

    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()


# Global singleton
_semantic_router: SemanticRouter | None = None


def get_semantic_router(
    embeddings_url: str | None = None, 
    local_model: str | None = None
) -> SemanticRouter:
    """Get or create the global semantic router."""
    global _semantic_router
    if _semantic_router is None:
        from .config import config
        url = embeddings_url or config.embeddings_url
        model = local_model or config.embedding_model
        _semantic_router = SemanticRouter(url, model)
    return _semantic_router


async def classify_content(content: str) -> ContentClassification:
    """Convenience function to classify content."""
    router = get_semantic_router()
    return await router.classify(content)


async def get_tier_for_content(content: str) -> tuple[str, float, str]:
    """Convenience function to get recommended tier."""
    router = get_semantic_router()
    return await router.get_recommended_tier(content)