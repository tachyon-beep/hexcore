# src/knowledge/cache_manager.py
import threading
import time
import logging
from datetime import datetime, timezone, UTC
from typing import Dict, Any, Optional, Set, List, Tuple

logger = logging.getLogger(__name__)


class KnowledgeGraphCache:
    """
    Thread-safe caching system for knowledge graph queries with time-based
    and content-based invalidation mechanisms.

    This cache reduces the computational load on the knowledge graph by storing
    query results and invalidating them based on either time (TTL) or content changes.
    It also provides metrics for cache performance monitoring.
    """

    def __init__(self, max_cache_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the knowledge graph cache.

        Args:
            max_cache_size: Maximum number of entries in cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, Any] = {}
        self.cache_access_times: Dict[str, datetime] = {}
        self.cache_creation_times: Dict[str, datetime] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        self.max_cache_size = max_cache_size
        self.ttl_seconds = ttl_seconds
        self.cache_lock = threading.RLock()  # Reentrant lock for thread safety

        # Metrics tracking
        self.metrics = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "evictions": 0,
            "size": 0,
            "created_at": datetime.now(UTC),
            "last_reset": datetime.now(UTC),
        }

        logger.info(
            f"Initialized knowledge graph cache (max_size={max_cache_size}, ttl={ttl_seconds}s)"
        )

    def get(self, query_key: str) -> Optional[Any]:
        """
        Get a result from cache if it exists and is valid.

        Args:
            query_key: Unique key for the query result

        Returns:
            Cached result or None if not found or expired
        """
        with self.cache_lock:
            if query_key not in self.cache:
                self.metrics["misses"] += 1
                logger.debug(f"Cache miss: {query_key}")
                return None

            # Check if entry has expired
            creation_time = self.cache_creation_times[query_key]
            age_seconds = (datetime.now(UTC) - creation_time).total_seconds()

            if age_seconds > self.ttl_seconds:
                self._invalidate_key(query_key)
                self.metrics["misses"] += 1
                logger.debug(f"Cache expired: {query_key} (age: {age_seconds:.1f}s)")
                return None

            # Update access time and hit count
            self.cache_access_times[query_key] = datetime.now(UTC)
            self.metrics["hits"] += 1

            logger.debug(f"Cache hit: {query_key} (age: {age_seconds:.1f}s)")
            return self.cache[query_key]

    def set(
        self, query_key: str, result: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store a result in the cache.

        Args:
            query_key: Unique key for the query result
            result: Result to cache
            metadata: Optional metadata for content-based invalidation
        """
        with self.cache_lock:
            # Evict entries if at capacity and this is a new key
            if len(self.cache) >= self.max_cache_size and query_key not in self.cache:
                self._evict_least_recently_used()

            # Store result
            current_time = datetime.now(UTC)
            self.cache[query_key] = result
            self.cache_access_times[query_key] = current_time

            # For new entries, set creation time
            if query_key not in self.cache_creation_times:
                self.cache_creation_times[query_key] = current_time

            # Store metadata for content-based invalidation if provided
            if metadata:
                self.cache_metadata[query_key] = metadata
            elif query_key in self.cache_metadata:
                # Remove metadata if not provided but existed before
                del self.cache_metadata[query_key]

            # Update size metric
            self.metrics["size"] = len(self.cache)

            logger.debug(f"Cache set: {query_key} (total: {len(self.cache)})")

    def invalidate(
        self,
        query_key: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries based on key or affected entities.

        Args:
            query_key: Optional specific key to invalidate
            entity_type: Optional entity type for content-based invalidation
            entity_id: Optional entity ID for content-based invalidation

        Returns:
            Number of entries invalidated
        """
        with self.cache_lock:
            invalidated = 0

            # Case 1: Invalidate by key
            if query_key is not None:
                if self._invalidate_key(query_key):
                    invalidated += 1

            # Case 2: Content-based invalidation
            elif entity_type is not None:
                # Find all entries affected by this entity
                keys_to_invalidate = self._find_affected_keys(entity_type, entity_id)

                for key in keys_to_invalidate:
                    if self._invalidate_key(key):
                        invalidated += 1

            # Case 3: Invalidate all (if no parameters provided)
            else:
                keys = list(self.cache.keys())
                for key in keys:
                    if self._invalidate_key(key):
                        invalidated += 1

            # Update metrics
            self.metrics["invalidations"] += invalidated
            self.metrics["size"] = len(self.cache)

            if invalidated > 0:
                logger.info(f"Invalidated {invalidated} cache entries")

            return invalidated

    def _invalidate_key(self, query_key: str) -> bool:
        """
        Invalidate a specific cache entry.

        Args:
            query_key: Key to invalidate

        Returns:
            True if entry was invalidated, False if not found
        """
        if query_key not in self.cache:
            return False

        del self.cache[query_key]
        del self.cache_access_times[query_key]
        del self.cache_creation_times[query_key]

        if query_key in self.cache_metadata:
            del self.cache_metadata[query_key]

        return True

    def _find_affected_keys(
        self, entity_type: str, entity_id: Optional[str] = None
    ) -> List[str]:
        """
        Find cache keys affected by an entity type or specific entity.

        Args:
            entity_type: Entity type
            entity_id: Optional specific entity ID

        Returns:
            List of affected cache keys
        """
        affected_keys = []

        for key, metadata in self.cache_metadata.items():
            # Skip entries without metadata
            if not metadata:
                continue

            # Check if this entry depends on the entity type
            is_affected = False

            # Case 1: Check for entity types in the metadata
            if "entity_types" in metadata and entity_type in metadata["entity_types"]:
                is_affected = True

            # Case 2: Check for specific entities in the metadata
            if not is_affected and "entities" in metadata:
                for entity in metadata.get("entities", []):
                    if entity.get("type") == entity_type and (
                        entity_id is None or entity.get("id") == entity_id
                    ):
                        is_affected = True
                        break

            if is_affected:
                affected_keys.append(key)

        return affected_keys

    def _evict_least_recently_used(self) -> None:
        """
        Evict the least recently used cache entry to make room for new entries.
        """
        if not self.cache_access_times:
            return  # Cache is empty

        # Find LRU entry
        lru_key = min(self.cache_access_times.items(), key=lambda x: x[1])[0]

        # Evict it
        self._invalidate_key(lru_key)
        self.metrics["evictions"] += 1

        logger.debug(f"Evicted LRU entry: {lru_key}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.

        Returns:
            Dictionary of metrics
        """
        with self.cache_lock:
            total_requests = self.metrics["hits"] + self.metrics["misses"]
            hit_rate = (
                (self.metrics["hits"] / total_requests * 100)
                if total_requests > 0
                else 0
            )

            metrics = {
                "hit_rate_percent": hit_rate,
                "size": len(self.cache),
                "max_size": self.max_cache_size,
                "ttl_seconds": self.ttl_seconds,
                "usage_percent": (
                    (len(self.cache) / self.max_cache_size * 100)
                    if self.max_cache_size > 0
                    else 0
                ),
                "created_at": self.metrics["created_at"].isoformat(),
                "last_reset": self.metrics["last_reset"].isoformat(),
                "age_seconds": (
                    datetime.now(UTC) - self.metrics["created_at"]
                ).total_seconds(),
                **self.metrics,
            }

            return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics except creation time."""
        with self.cache_lock:
            created_at = self.metrics["created_at"]
            self.metrics = {
                "hits": 0,
                "misses": 0,
                "invalidations": 0,
                "evictions": 0,
                "size": len(self.cache),
                "created_at": created_at,
                "last_reset": datetime.now(UTC),
            }

            logger.info("Cache metrics reset")

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with self.cache_lock:
            count = len(self.cache)
            self.cache = {}
            self.cache_access_times = {}
            self.cache_creation_times = {}
            self.cache_metadata = {}

            # Update metrics
            self.metrics["size"] = 0
            self.metrics["invalidations"] += count

            logger.info(f"Cache cleared ({count} entries)")
            return count

    def generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.

        Args:
            *args: Positional arguments to include in the key
            **kwargs: Keyword arguments to include in the key

        Returns:
            Cache key string
        """
        # Create a canonical string representation of args and kwargs
        key_parts = []

        # Add args
        for arg in args:
            key_parts.append(str(arg))

        # Add sorted kwargs
        for k in sorted(kwargs.keys()):
            key_parts.append(f"{k}={kwargs[k]}")

        # Join with a delimiter unlikely to appear in the values
        return "||".join(key_parts)
