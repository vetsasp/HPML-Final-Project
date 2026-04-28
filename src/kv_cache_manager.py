"""Tiered KV cache metadata and residency manager."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    from . import config
except ImportError:
    import config


logger = logging.getLogger("rag_pipeline")

MB = 1024 * 1024


@dataclass
class CacheEntry:
    """Metadata for a cacheable prompt block."""

    key: str
    block_type: str
    text: str
    token_count: int
    estimated_kv_bytes: int
    tier: str = "disk"
    access_count: int = 0
    last_access_ts: float = 0.0
    disk_path: Optional[str] = None


class TieredKVManager:
    """Tracks prompt block residency across GPU, CPU, and disk tiers."""

    def __init__(self, cfg=None):
        self.config = cfg or config.config
        self.tier_config = self.config.tiered_kv
        self.kv_dir = Path(self.config.paths.kv_cache_dir)
        self.entries: Dict[str, CacheEntry] = {}
        self.gpu_bytes = 0
        self.cpu_bytes = 0
        self.disk_bytes = 0
        self.hits = {"gpu": 0, "cpu": 0, "disk": 0}
        self.misses = 0
        self.promotions = 0
        self.demotions = 0

    def estimate_kv_size(self, token_count: int) -> int:
        """Estimate KV size in bytes for a block."""
        hidden_factor = 8192
        return max(token_count, 1) * hidden_factor

    def _entry_path(self, key: str) -> Path:
        return self.kv_dir / f"{key}.json"

    def _write_disk_entry(self, entry: CacheEntry) -> None:
        disk_path = self._entry_path(entry.key)
        payload = asdict(entry)
        payload["text"] = entry.text
        disk_path.write_text(json.dumps(payload), encoding="utf-8")
        entry.disk_path = str(disk_path)

    def _set_tier(self, entry: CacheEntry, new_tier: str) -> None:
        if entry.tier == new_tier:
            return

        if entry.tier == "gpu":
            self.gpu_bytes = max(0, self.gpu_bytes - entry.estimated_kv_bytes)
        elif entry.tier == "cpu":
            self.cpu_bytes = max(0, self.cpu_bytes - entry.estimated_kv_bytes)
        elif entry.tier == "disk":
            self.disk_bytes = max(0, self.disk_bytes - entry.estimated_kv_bytes)

        entry.tier = new_tier

        if new_tier == "gpu":
            self.gpu_bytes += entry.estimated_kv_bytes
        elif new_tier == "cpu":
            self.cpu_bytes += entry.estimated_kv_bytes
        elif new_tier == "disk":
            self.disk_bytes += entry.estimated_kv_bytes

    def _ensure_budget(self, tier: str) -> None:
        if tier == "gpu":
            budget = self.tier_config.gpu_kv_budget_mb * MB
            usage = self.gpu_bytes
            next_tier = "cpu"
        elif tier == "cpu":
            budget = self.tier_config.cpu_kv_budget_mb * MB
            usage = self.cpu_bytes
            next_tier = "disk"
        else:
            budget = self.tier_config.disk_kv_budget_mb * MB
            usage = self.disk_bytes
            next_tier = None

        if usage <= budget or next_tier is None:
            return

        candidates = [entry for entry in self.entries.values() if entry.tier == tier]
        candidates.sort(key=lambda entry: (entry.access_count, entry.last_access_ts))

        while usage > budget and candidates:
            victim = candidates.pop(0)
            self._move_entry(victim, next_tier)
            usage = self.gpu_bytes if tier == "gpu" else self.cpu_bytes

    def _move_entry(self, entry: CacheEntry, new_tier: str) -> None:
        if entry.tier == new_tier:
            return
        if new_tier == "disk":
            self._write_disk_entry(entry)
        self._set_tier(entry, new_tier)
        if new_tier in {"gpu", "cpu"}:
            self.promotions += 1
        else:
            self.demotions += 1

    def get_or_create_entry(self, block, token_count: int) -> CacheEntry:
        """Get or create a cache entry for a prompt block."""
        entry = self.entries.get(block.key)
        if entry is None:
            entry = CacheEntry(
                key=block.key,
                block_type=block.block_type,
                text=block.text,
                token_count=token_count,
                estimated_kv_bytes=self.estimate_kv_size(token_count),
            )
            self.entries[block.key] = entry
            self._write_disk_entry(entry)
            self._set_tier(entry, "disk")
            self.misses += 1
        return entry

    def record_access(self, entry: CacheEntry) -> None:
        """Update access statistics and tier residency for an entry."""
        if entry.tier in self.hits:
            self.hits[entry.tier] += 1

        entry.access_count += 1
        entry.last_access_ts = time.time()

        if entry.access_count >= self.tier_config.min_reuse_count_for_gpu:
            self._move_entry(entry, "gpu")
            self._ensure_budget("gpu")
        elif entry.access_count >= 1:
            self._move_entry(entry, "cpu")
            self._ensure_budget("cpu")

    def prepare_entry(self, block, token_count: int) -> CacheEntry:
        """Register and update residency for a block."""
        entry = self.get_or_create_entry(block, token_count)
        self.record_access(entry)
        return entry

    def get_stats(self) -> dict:
        """Return cache residency and usage statistics."""
        counts = {"gpu": 0, "cpu": 0, "disk": 0}
        for entry in self.entries.values():
            counts[entry.tier] += 1

        return {
            "entries": len(self.entries),
            "entry_counts": counts,
            "hits": dict(self.hits),
            "misses": self.misses,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "gpu_bytes": self.gpu_bytes,
            "cpu_bytes": self.cpu_bytes,
            "disk_bytes": self.disk_bytes,
            "gpu_budget_mb": self.tier_config.gpu_kv_budget_mb,
            "cpu_budget_mb": self.tier_config.cpu_kv_budget_mb,
            "disk_budget_mb": self.tier_config.disk_kv_budget_mb,
            "cache_policy": self.tier_config.cache_policy,
        }
