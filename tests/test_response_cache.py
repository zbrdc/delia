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
Tests for response cache functionality.

Run with: uv run pytest tests/test_response_cache.py -v
"""

import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path):
    """Use a temp directory for test data."""
    os.environ["DELIA_DATA_DIR"] = str(tmp_path)

    # Clear cached modules
    modules_to_clear = ["delia.response_cache", "delia.paths", "delia"]
    for mod in list(sys.modules.keys()):
        if any(mod.startswith(m) or mod == m for m in modules_to_clear):
            del sys.modules[mod]

    yield

    os.environ.pop("DELIA_DATA_DIR", None)


@pytest.fixture
def cache(tmp_path):
    """Provide a fresh ResponseCache instance for each test."""
    from delia.response_cache import ResponseCache
    return ResponseCache(max_entries=10, ttl_seconds=60, cache_dir=tmp_path / "cache")


class TestCacheHitMiss:
    """Test basic cache hit and miss scenarios."""

    def test_cache_hit_exact_match(self, cache):
        """Same params should return cached response."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        response = "The answer is 4."
        tokens = 100

        # Store response
        cache.put(model, prompt, response, tokens)

        # Retrieve should hit
        cached = cache.get(model, prompt)
        assert cached == response

        # Stats should show 1 hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss_different_prompt(self, cache):
        """Different prompt should be a cache miss."""
        prompt1 = "What is 2+2?"
        prompt2 = "What is 3+3?"
        model = "qwen2.5:14b"
        response = "The answer is 4."
        tokens = 100

        # Store response for prompt1
        cache.put(model, prompt1, response, tokens)

        # Retrieve with different prompt should miss
        cached = cache.get(model, prompt2)
        assert cached is None

        # Stats should show 1 miss
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_cache_miss_different_model(self, cache):
        """Different model should be a cache miss."""
        prompt = "What is 2+2?"
        model1 = "qwen2.5:14b"
        model2 = "llama3.1:70b"
        response = "The answer is 4."
        tokens = 100

        # Store response for model1
        cache.put(model1, prompt, response, tokens)

        # Retrieve with different model should miss
        cached = cache.get(model2, prompt)
        assert cached is None

        # Stats should show 1 miss
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1

    def test_cache_miss_different_system(self, cache):
        """Different system prompt should be a cache miss."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        system1 = "You are a helpful assistant."
        system2 = "You are a math tutor."
        response = "The answer is 4."
        tokens = 100

        # Store response with system1
        cache.put(model, prompt, response, tokens, system=system1)

        # Retrieve with different system should miss
        cached = cache.get(model, prompt, system=system2)
        assert cached is None

        # Stats should show 1 miss
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 1


class TestTTLExpiration:
    """Test TTL expiration behavior."""

    def test_ttl_expiration(self, monkeypatch, tmp_path):
        """Expired entries should not be returned."""
        from delia.response_cache import ResponseCache, CachedResponse

        # Create cache with 2-second TTL
        cache = ResponseCache(max_entries=10, ttl_seconds=2, cache_dir=tmp_path / "cache")

        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        response = "The answer is 4."
        tokens = 100

        # Store response at time T
        cache.put(model, prompt, response, tokens)

        # Should hit immediately
        cached = cache.get(model, prompt)
        assert cached == response

        # Mock CachedResponse.is_expired to return True
        original_is_expired = CachedResponse.is_expired

        def mock_is_expired(self):
            # First call (during get) should return expired
            return True

        monkeypatch.setattr(CachedResponse, "is_expired", mock_is_expired)

        # Should miss due to expiration
        cached = cache.get(model, prompt)
        assert cached is None

        # Stats should show 1 hit, 1 miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_ttl_not_expired(self, tmp_path):
        """Non-expired entries should be returned."""
        from delia.response_cache import ResponseCache

        # Create cache with 60-second TTL
        cache = ResponseCache(max_entries=10, ttl_seconds=60, cache_dir=tmp_path / "cache")

        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        response = "The answer is 4."
        tokens = 100

        # Store response
        cache.put(model, prompt, response, tokens)

        # Should still hit (60 seconds hasn't passed)
        cached = cache.get(model, prompt)
        assert cached == response

        # Stats should show 1 hit, 0 misses
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0


class TestLRUEviction:
    """Test LRU eviction behavior."""

    def test_lru_eviction(self, tmp_path):
        """Cache should evict when exceeding max_entries."""
        from delia.response_cache import ResponseCache

        # Create cache with max 3 entries
        cache = ResponseCache(max_entries=3, ttl_seconds=60, cache_dir=tmp_path / "cache")

        # Add 3 entries
        cache.put("model1", "prompt1", "response1", 100)
        cache.put("model2", "prompt2", "response2", 100)
        cache.put("model3", "prompt3", "response3", 100)

        # All should be retrievable
        assert cache.get("model1", "prompt1") == "response1"
        assert cache.get("model2", "prompt2") == "response2"
        assert cache.get("model3", "prompt3") == "response3"

        # Add a 4th entry - should trigger eviction
        cache.put("model4", "prompt4", "response4", 100)

        # At least one should be evicted
        stats = cache.get_stats()
        assert stats["evictions"] >= 1
        assert stats["entries"] <= 3

    def test_lru_access_order(self, tmp_path):
        """Most recently accessed entries should not be evicted first."""
        from delia.response_cache import ResponseCache

        # Create cache with max 3 entries
        cache = ResponseCache(max_entries=3, ttl_seconds=60, cache_dir=tmp_path / "cache")

        # Add 3 entries
        cache.put("model1", "prompt1", "response1", 100)
        cache.put("model2", "prompt2", "response2", 100)
        cache.put("model3", "prompt3", "response3", 100)

        # Access prompt1 multiple times (increases hit count)
        for _ in range(5):
            cache.get("model1", "prompt1")

        # Add a 4th entry - should evict based on hits+recency
        cache.put("model4", "prompt4", "response4", 100)

        # prompt1 should still be accessible (high hit count)
        assert cache.get("model1", "prompt1") == "response1"

    def test_eviction_past_max_entries(self, tmp_path):
        """Filling cache past max_entries should evict correctly."""
        from delia.response_cache import ResponseCache

        # Create small cache
        cache = ResponseCache(max_entries=5, ttl_seconds=60, cache_dir=tmp_path / "cache")

        # Fill past max (add 10 entries)
        for i in range(10):
            cache.put(f"model{i}", f"prompt{i}", f"response{i}", 100)

        # Should have evictions
        stats = cache.get_stats()
        assert stats["evictions"] > 0
        assert stats["entries"] <= 5


class TestThreadSafety:
    """Test thread safety of cache operations."""

    def test_thread_safety(self, tmp_path):
        """Concurrent reads/writes should complete without errors."""
        from delia.response_cache import ResponseCache

        cache = ResponseCache(max_entries=100, ttl_seconds=60, cache_dir=tmp_path / "cache")
        errors = []

        def writer(thread_id: int):
            """Write entries to cache."""
            try:
                for i in range(50):
                    cache.put(
                        f"model{thread_id}",
                        f"prompt{thread_id}_{i}",
                        f"response{thread_id}_{i}",
                        100,
                    )
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int):
            """Read entries from cache."""
            try:
                for i in range(50):
                    cache.get(f"model{thread_id}", f"prompt{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        # Create threads
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0

        # Cache should be in valid state
        stats = cache.get_stats()
        assert stats["entries"] >= 0
        assert stats["entries"] <= cache.max_entries


class TestStats:
    """Test cache statistics."""

    def test_get_stats(self, tmp_path):
        """Stats should reflect hits/misses/evictions correctly."""
        from delia.response_cache import ResponseCache

        cache = ResponseCache(max_entries=2, ttl_seconds=60, cache_dir=tmp_path / "cache")

        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["evictions"] == 0
        assert stats["entries"] == 0
        assert stats["max_entries"] == 2
        assert stats["hit_rate"] == 0.0
        assert stats["ttl_seconds"] == 60

        # Add entry and hit it
        cache.put("model1", "prompt1", "response1", 100)
        cache.get("model1", "prompt1")  # Hit

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0
        assert stats["entries"] == 1
        assert stats["hit_rate"] == 100.0

        # Miss
        cache.get("model2", "prompt2")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

        # Eviction
        cache.put("model2", "prompt2", "response2", 100)
        cache.put("model3", "prompt3", "response3", 100)  # Triggers eviction

        stats = cache.get_stats()
        assert stats["evictions"] >= 1
        assert stats["entries"] <= 2

    def test_stats_after_clear(self, tmp_path):
        """clear() should reset cache but preserve stats."""
        from delia.response_cache import ResponseCache

        cache = ResponseCache(max_entries=10, ttl_seconds=60, cache_dir=tmp_path / "cache")

        # Generate some stats
        cache.put("model1", "prompt1", "response1", 100)
        cache.get("model1", "prompt1")  # Hit
        cache.get("model2", "prompt2")  # Miss

        # Clear cache
        cache.clear()

        stats = cache.get_stats()
        # Stats are preserved after clear
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        # Cache size should be 0
        assert stats["entries"] == 0


class TestCacheKeyConsistency:
    """Test cache key generation consistency."""

    def test_cache_key_consistency(self, cache):
        """Same params should always produce same key."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        system = "You are helpful."

        # Generate key multiple times
        key1 = cache._compute_key(model, prompt, system)
        key2 = cache._compute_key(model, prompt, system)
        key3 = cache._compute_key(model, prompt, system)

        assert key1 == key2 == key3

    def test_cache_key_different_params(self, cache):
        """Different params should produce different keys."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"

        key1 = cache._compute_key(model, prompt, None)
        key2 = cache._compute_key(model, prompt, "system")
        key3 = cache._compute_key("other_model", prompt, None)
        key4 = cache._compute_key(model, "other_prompt", None)

        # All keys should be different
        keys = [key1, key2, key3, key4]
        assert len(keys) == len(set(keys))


class TestNoneSystemPrompt:
    """Test handling of None system prompt."""

    def test_none_system_prompt(self, cache):
        """None system prompt should be handled correctly."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        response = "The answer is 4."
        tokens = 100

        # Store with None system prompt
        cache.put(model, prompt, response, tokens, system=None)

        # Retrieve with None system prompt
        cached = cache.get(model, prompt, system=None)
        assert cached == response

        # Retrieve without specifying system (defaults to None)
        cached = cache.get(model, prompt)
        assert cached == response

    def test_none_vs_empty_system_prompt(self, cache):
        """None and empty string should be treated the same."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        response = "The answer is 4."
        tokens = 100

        # Store with None
        cache.put(model, prompt, response, tokens, system=None)

        # Retrieve with empty string should hit
        cached = cache.get(model, prompt, system="")
        assert cached == response

    def test_system_prompt_vs_none(self, cache):
        """Non-empty system prompt should not match None."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        response = "The answer is 4."
        tokens = 100

        # Store with None
        cache.put(model, prompt, response, tokens, system=None)

        # Retrieve with non-empty system should miss
        cached = cache.get(model, prompt, system="You are helpful.")
        assert cached is None


class TestCacheUpdate:
    """Test cache update behavior."""

    def test_update_existing_entry(self, cache):
        """Updating existing entry should replace response."""
        prompt = "What is 2+2?"
        model = "qwen2.5:14b"
        response1 = "The answer is 4."
        response2 = "2 + 2 = 4"
        tokens = 100

        # Store initial response
        cache.put(model, prompt, response1, tokens)
        assert cache.get(model, prompt) == response1

        # Update with new response
        cache.put(model, prompt, response2, tokens)
        assert cache.get(model, prompt) == response2

        # Size should still be 1 (no eviction from update)
        stats = cache.get_stats()
        assert stats["entries"] == 1


class TestCacheClear:
    """Test cache clearing."""

    def test_clear(self, cache):
        """clear() should remove all entries."""
        # Add several entries
        cache.put("model1", "prompt1", "response1", 100)
        cache.put("model2", "prompt2", "response2", 100)
        cache.put("model3", "prompt3", "response3", 100)

        # Verify they're there
        assert cache.get("model1", "prompt1") == "response1"

        # Clear cache
        cache.clear()

        # All should be gone
        assert cache.get("model1", "prompt1") is None
        assert cache.get("model2", "prompt2") is None
        assert cache.get("model3", "prompt3") is None

        stats = cache.get_stats()
        assert stats["entries"] == 0


class TestPersistence:
    """Test disk persistence."""

    def test_save_to_disk(self, tmp_path):
        """save_to_disk() should persist cache to file."""
        from delia.response_cache import ResponseCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        cache = ResponseCache(max_entries=10, ttl_seconds=60, cache_dir=cache_dir)

        # Add entries
        cache.put("model1", "prompt1", "response1", 100)
        cache.put("model2", "prompt2", "response2", 200)

        # Save to disk
        cache.save_to_disk()

        # File should exist
        cache_file = cache_dir / "response_cache.json"
        assert cache_file.exists()

    def test_load_from_disk(self, tmp_path):
        """load_from_disk() should restore cache from file."""
        from delia.response_cache import ResponseCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create and populate cache
        cache1 = ResponseCache(max_entries=10, ttl_seconds=60, cache_dir=cache_dir)
        cache1.put("model1", "prompt1", "response1", 100)
        cache1.save_to_disk()

        # Create new cache and load
        cache2 = ResponseCache(max_entries=10, ttl_seconds=60, cache_dir=cache_dir)
        cache2.load_from_disk()

        # Should have loaded entry
        assert cache2.get("model1", "prompt1") == "response1"

    def test_load_nonexistent_file(self, tmp_path):
        """load_from_disk() should handle missing file gracefully."""
        from delia.response_cache import ResponseCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        cache = ResponseCache(max_entries=10, ttl_seconds=60, cache_dir=cache_dir)

        # Should not raise
        cache.load_from_disk()

        stats = cache.get_stats()
        assert stats["entries"] == 0


class TestGlobalCache:
    """Test global cache instance."""

    def test_get_cache_singleton(self, tmp_path):
        """get_cache() should return singleton instance."""
        from delia.response_cache import get_cache, _global_cache

        # Import will auto-initialize global cache
        cache1 = get_cache()
        cache2 = get_cache()

        # Should be same instance
        assert cache1 is cache2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
