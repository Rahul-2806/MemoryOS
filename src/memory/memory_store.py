import sqlite3
import json
import numpy as np
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from src.memory.decay import DecayFunction


DB_PATH = "C:/PROJECT/MemoryOS/data/memory.db"


class MemoryStore:
    """
    Persistent SQLite memory store.

    Handles:
    - Saving memories to disk
    - Retrieving memories by ID or keyword
    - Applying decay to all memories
    - Pruning forgotten memories
    - Boosting recalled memories
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.decay = DecayFunction()
        self._init_db()
        print(f"[MemoryOS] Memory Store initialized at {db_path} ✓")

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    semantic_vector TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    surprise_score REAL DEFAULT 0.5,
                    emotional_weight REAL DEFAULT 0.5,
                    associative_keys TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    decay_factor REAL DEFAULT 1.0,
                    session_id TEXT DEFAULT 'default',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance
                ON memories(importance_score DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON memories(session_id)
            """)
            conn.commit()

    def save(self, memory: dict, session_id: str = "default") -> bool:
        """Save a single memory to the store."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    memory["memory_id"],
                    memory["text"],
                    memory.get("role", "user"),
                    json.dumps(memory["semantic_vector"]),
                    memory["importance_score"],
                    memory.get("surprise_score", 0.5),
                    memory.get("emotional_weight", 0.5),
                    json.dumps(memory["associative_keys"]),
                    memory["created_at"],
                    memory["last_accessed"],
                    memory.get("access_count", 0),
                    memory.get("decay_factor", 1.0),
                    session_id,
                    json.dumps(memory.get("metadata", {}))
                ))
                conn.commit()
            return True
        except Exception as e:
            print(f"[MemoryStore] Save error: {e}")
            return False

    def save_many(self, memories: list[dict], session_id: str = "default") -> int:
        """Save multiple memories at once. Returns count saved."""
        saved = 0
        for memory in memories:
            if self.save(memory, session_id):
                saved += 1
        return saved

    def get(self, memory_id: str) -> Optional[dict]:
        """Retrieve a memory by ID and boost its importance on access."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?",
                (memory_id,)
            ).fetchone()

            if not row:
                return None

            memory = self._row_to_dict(row)

            # Boost importance on access (spaced repetition)
            new_importance = self.decay.boost_on_access(
                memory["importance_score"],
                memory["access_count"]
            )
            new_access_count = memory["access_count"] + 1

            conn.execute("""
                UPDATE memories
                SET importance_score = ?, access_count = ?, last_accessed = ?
                WHERE memory_id = ?
            """, (
                new_importance,
                new_access_count,
                datetime.now(timezone.utc).isoformat(),
                memory_id
            ))
            conn.commit()

            memory["importance_score"] = new_importance
            memory["access_count"] = new_access_count
            return memory

    def search_by_keys(self, keys: list[str], limit: int = 10) -> list[dict]:
        """Find memories that share associative keys."""
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for key in keys:
                rows = conn.execute("""
                    SELECT * FROM memories
                    WHERE associative_keys LIKE ?
                    ORDER BY importance_score DESC
                    LIMIT ?
                """, (f'%"{key}"%', limit)).fetchall()
                for row in rows:
                    memory = self._row_to_dict(row)
                    if memory not in results:
                        results.append(memory)

        # Sort by importance
        results.sort(key=lambda x: x["importance_score"], reverse=True)
        return results[:limit]

    def get_all_by_session(self, session_id: str) -> list[dict]:
        """Get all memories for a session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM memories
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,)).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_top_memories(self, limit: int = 20) -> list[dict]:
        """Get highest importance memories across all sessions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM memories
                ORDER BY importance_score DESC
                LIMIT ?
            """, (limit,)).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def apply_decay_all(self) -> dict:
        """
        Apply decay to all memories.
        Called periodically (e.g., every hour or on startup).
        Returns stats about what decayed.
        """
        stats = {"updated": 0, "forgotten": 0, "total": 0}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memories").fetchall()
            stats["total"] = len(rows)

            for row in rows:
                memory = self._row_to_dict(row)
                retention = self.decay.compute_retention(
                    memory["importance_score"],
                    memory["created_at"],
                    memory["last_accessed"],
                    memory["access_count"]
                )

                if self.decay.should_forget(
                    memory["importance_score"],
                    memory["created_at"],
                    memory["last_accessed"],
                    memory["access_count"]
                ):
                    # Delete forgotten memory
                    conn.execute(
                        "DELETE FROM memories WHERE memory_id = ?",
                        (memory["memory_id"],)
                    )
                    stats["forgotten"] += 1
                else:
                    # Update decay factor
                    conn.execute("""
                        UPDATE memories SET decay_factor = ?
                        WHERE memory_id = ?
                    """, (retention, memory["memory_id"]))
                    stats["updated"] += 1

            conn.commit()

        return stats

    def get_stats(self) -> dict:
        """Get memory store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            avg_importance = conn.execute(
                "SELECT AVG(importance_score) FROM memories"
            ).fetchone()[0] or 0
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM memories"
            ).fetchone()[0]

        return {
            "total_memories": total,
            "avg_importance": round(avg_importance, 4),
            "total_sessions": sessions,
            "db_path": self.db_path
        }

    def _row_to_dict(self, row) -> dict:
        """Convert SQLite row to memory dict."""
        d = dict(row)
        d["semantic_vector"] = json.loads(d["semantic_vector"])
        d["associative_keys"] = json.loads(d["associative_keys"])
        d["metadata"] = json.loads(d["metadata"])
        return d