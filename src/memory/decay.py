import numpy as np
from datetime import datetime, timezone


class DecayFunction:
    """
    Biologically-inspired memory decay.

    Based on Ebbinghaus Forgetting Curve:
    R = e^(-t/S)
    Where:
        R = memory retention (0 to 1)
        t = time elapsed (in hours)
        S = stability (importance score — more important = slower decay)

    High importance memories decay slowly.
    Low importance memories decay fast.
    Accessed memories get their stability boosted (spaced repetition effect).
    """

    def __init__(self, base_decay_rate: float = 0.01):
        self.base_decay_rate = base_decay_rate

    def compute_retention(
        self,
        importance_score: float,
        created_at: str,
        last_accessed: str,
        access_count: int
    ) -> float:
        """
        Compute current retention strength of a memory.

        Returns:
            retention: float between 0.0 (forgotten) and 1.0 (perfect recall)
        """
        now = datetime.now(timezone.utc)

        # Parse timestamps
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        accessed = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))

        # Make timezone aware if naive
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if accessed.tzinfo is None:
            accessed = accessed.replace(tzinfo=timezone.utc)

        # Hours since last access
        hours_elapsed = (now - accessed).total_seconds() / 3600

        # Stability = importance + access bonus
        # More important + more accessed = much slower decay
        access_bonus = min(0.5, access_count * 0.05)
        stability = importance_score + access_bonus

        # Ebbinghaus decay: R = e^(-t / (S * scale))
        # Scale of 100 means a 0.5 importance memory lasts ~50 hours at full strength
        scale = 100.0
        retention = np.exp(-hours_elapsed / (stability * scale + 1e-6))

        return float(np.clip(retention, 0.0, 1.0))

    def should_forget(
        self,
        importance_score: float,
        created_at: str,
        last_accessed: str,
        access_count: int,
        threshold: float = 0.05
    ) -> bool:
        """
        Returns True if memory retention has fallen below threshold.
        These memories get pruned during consolidation.
        """
        retention = self.compute_retention(
            importance_score, created_at, last_accessed, access_count
        )
        return retention < threshold

    def boost_on_access(self, current_importance: float, access_count: int) -> float:
        """
        When a memory is accessed, its importance slightly increases.
        Simulates spaced repetition — recalled memories become stronger.
        """
        boost = 0.02 * (1 / (access_count + 1))
        return float(min(1.0, current_importance + boost))