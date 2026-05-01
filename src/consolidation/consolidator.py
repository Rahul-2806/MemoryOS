import numpy as np
import threading
import time
from datetime import datetime, timezone
from src.memory.memory_store import MemoryStore
from src.memory.decay import DecayFunction


class MemoryConsolidator:
    """
    Async background consolidation — like the brain during sleep.

    Does 3 things:
    1. Merges redundant memories (similar meaning = one stronger memory)
    2. Prunes forgotten memories (low retention = deleted)
    3. Boosts highly accessed memories (important = stronger)

    Runs in a background thread automatically.
    """

    def __init__(self, store: MemoryStore, interval_seconds: int = 300):
        self.store = store
        self.decay = DecayFunction()
        self.interval = interval_seconds
        self._running = False
        self._thread = None
        self.stats = {
            "last_run": None,
            "total_runs": 0,
            "total_merged": 0,
            "total_pruned": 0
        }
        print("[MemoryOS] Consolidator initialized ✓")

    def _cosine_similarity(self, vec_a: list, vec_b: list) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _merge_memories(self, mem_a: dict, mem_b: dict) -> dict:
        """
        Merge two similar memories into one stronger memory.
        Keeps the more important text, boosts importance score.
        """
        # Keep text from more important memory
        if mem_a["importance_score"] >= mem_b["importance_score"]:
            primary, secondary = mem_a, mem_b
        else:
            primary, secondary = mem_b, mem_a

        # Merge associative keys
        merged_keys = list(set(
            primary["associative_keys"] + secondary["associative_keys"]
        ))[:15]

        # Average vectors
        vec_a = np.array(primary["semantic_vector"])
        vec_b = np.array(secondary["semantic_vector"])
        merged_vector = ((vec_a + vec_b) / 2).tolist()

        # Boost importance
        merged_importance = min(1.0, (
            primary["importance_score"] * 0.7 +
            secondary["importance_score"] * 0.3 +
            0.05  # merge bonus
        ))

        merged = primary.copy()
        merged["semantic_vector"] = merged_vector
        merged["importance_score"] = merged_importance
        merged["associative_keys"] = merged_keys
        merged["access_count"] = (
            primary["access_count"] + secondary["access_count"]
        )
        merged["metadata"]["merged_from"] = secondary["memory_id"]

        return merged

    def consolidate_once(self) -> dict:
        """
        Run one consolidation cycle.
        Returns stats about what happened.
        """
        run_stats = {"merged": 0, "pruned": 0, "checked": 0}

        all_memories = self.store.get_top_memories(limit=500)
        run_stats["checked"] = len(all_memories)

        # Step 1 — Find and merge redundant memories
        merged_ids = set()
        for i, mem_a in enumerate(all_memories):
            if mem_a["memory_id"] in merged_ids:
                continue
            for mem_b in all_memories[i+1:]:
                if mem_b["memory_id"] in merged_ids:
                    continue
                sim = self._cosine_similarity(
                    mem_a["semantic_vector"],
                    mem_b["semantic_vector"]
                )
                # If >85% similar, merge them
                if sim > 0.85:
                    merged = self._merge_memories(mem_a, mem_b)
                    self.store.save(merged)
                    # Delete the secondary memory
                    self._delete_memory(mem_b["memory_id"])
                    merged_ids.add(mem_b["memory_id"])
                    run_stats["merged"] += 1

        # Step 2 — Apply decay and prune forgotten memories
        decay_stats = self.store.apply_decay_all()
        run_stats["pruned"] = decay_stats.get("forgotten", 0)

        # Update global stats
        self.stats["last_run"] = datetime.now(timezone.utc).isoformat()
        self.stats["total_runs"] += 1
        self.stats["total_merged"] += run_stats["merged"]
        self.stats["total_pruned"] += run_stats["pruned"]

        return run_stats

    def _delete_memory(self, memory_id: str):
        """Delete a memory from the store."""
        import sqlite3
        try:
            with sqlite3.connect(self.store.db_path) as conn:
                conn.execute(
                    "DELETE FROM memories WHERE memory_id = ?",
                    (memory_id,)
                )
                conn.commit()
        except Exception as e:
            print(f"[Consolidator] Delete error: {e}")

    def _run_loop(self):
        """Background thread loop."""
        while self._running:
            try:
                stats = self.consolidate_once()
                print(
                    f"[Consolidator] Run #{self.stats['total_runs']} — "
                    f"merged={stats['merged']}, pruned={stats['pruned']}"
                )
            except Exception as e:
                print(f"[Consolidator] Error: {e}")
            time.sleep(self.interval)

    def start(self):
        """Start background consolidation thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[MemoryOS] Consolidator running every {self.interval}s ✓")

    def stop(self):
        """Stop background consolidation."""
        self._running = False
        print("[MemoryOS] Consolidator stopped.")