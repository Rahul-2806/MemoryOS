import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import Optional
import hashlib
import uuid

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[MemoryOS] Encoder running on: {DEVICE}")


class ImportanceScorer(nn.Module):
    """
    Neural network that scores how important a memory is.
    Importance = f(surprise, emotional_weight, relevance)
    Runs on GTX 1650 GPU.
    """

    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.to(DEVICE)

    def forward(self, current_embed: torch.Tensor, context_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([current_embed, context_embed], dim=-1)
        return self.scorer(combined)


class EpisodicEncoder:
    """
    Core encoder that converts raw text into episodic memories.

    Each memory contains:
    - semantic_vector: 384-dim embedding (what it means)
    - importance_score: 0-1 (how important it is)
    - temporal_marker: timestamp with decay metadata
    - associative_keys: keywords for graph linking
    - memory_id: unique identifier
    """

    def __init__(self):
        print("[MemoryOS] Loading sentence transformer...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEVICE))
        self.importance_scorer = ImportanceScorer(embedding_dim=384)
        self.context_buffer = []  # Last N embeddings as context
        self.context_window = 5
        print("[MemoryOS] Episodic Encoder ready ✓")

    def _get_context_embedding(self) -> torch.Tensor:
        """Average of recent memory embeddings as context."""
        if not self.context_buffer:
            return torch.zeros(384).to(DEVICE)
        stacked = torch.stack(self.context_buffer[-self.context_window:])
        return stacked.mean(dim=0)

    def _extract_associative_keys(self, text: str) -> list[str]:
        """Extract keywords for building associative links."""
        stopwords = {
            "the", "a", "an", "is", "it", "in", "on", "at", "to",
            "for", "of", "and", "or", "but", "i", "you", "we", "they",
            "was", "are", "be", "been", "have", "has", "do", "did",
            "this", "that", "with", "from", "by", "as", "my", "your"
        }
        words = text.lower().split()
        keys = [w.strip(".,!?;:") for w in words
                if w.strip(".,!?;:") not in stopwords
                and len(w.strip(".,!?;:")) > 2]
        return list(set(keys))[:10]  # Max 10 keys per memory

    def _compute_surprise(
        self,
        current_embed: torch.Tensor,
        context_embed: torch.Tensor
    ) -> float:
        """
        Surprise = how different this memory is from recent context.
        High surprise = high importance (novel information).
        """
        if context_embed.sum() == 0:
            return 0.5  # Neutral surprise for first memory

        cosine_sim = nn.functional.cosine_similarity(
            current_embed.unsqueeze(0),
            context_embed.unsqueeze(0)
        ).item()

        # Surprise is inverse of similarity
        # Very similar = low surprise, very different = high surprise
        return 1.0 - abs(cosine_sim)

    def encode(
        self,
        text: str,
        role: str = "user",  # "user" or "assistant"
        metadata: Optional[dict] = None
    ) -> dict:
        """
        Main encoding function.
        Takes raw text → returns a complete episodic memory object.

        Args:
            text: The text to encode
            role: Who said it (user/assistant)
            metadata: Any extra info (session_id, etc.)

        Returns:
            memory: Complete memory dict ready to store
        """
        # 1. Generate semantic embedding on GPU
        raw_embed = self.embedder.encode(
            text,
            convert_to_tensor=True,
            device=str(DEVICE)
        )

        # 2. Get context from recent memories
        context_embed = self._get_context_embedding()

        # 3. Compute surprise score
        surprise = self._compute_surprise(raw_embed, context_embed)

        # 4. Score importance via neural network
        with torch.no_grad():
            importance = self.importance_scorer(
                raw_embed.unsqueeze(0),
                context_embed.unsqueeze(0)
            ).item()

        # 5. Blend surprise into importance
        # Memories that are surprising OR important by content get high scores
        final_importance = (importance * 0.6) + (surprise * 0.4)

        # 6. Emotional weight — length + punctuation as proxy
        emotional_weight = min(1.0, (
            len(text) / 500 +
            text.count("!") * 0.1 +
            text.count("?") * 0.05
        ))

        # 7. Final importance blended with emotional weight
        final_importance = (final_importance * 0.7) + (emotional_weight * 0.3)
        final_importance = float(np.clip(final_importance, 0.0, 1.0))

        # 8. Update context buffer
        self.context_buffer.append(raw_embed.detach())
        if len(self.context_buffer) > 20:
            self.context_buffer.pop(0)

        # 9. Build memory object
        memory = {
            "memory_id": str(uuid.uuid4()),
            "text": text,
            "role": role,
            "semantic_vector": raw_embed.cpu().numpy().tolist(),
            "importance_score": final_importance,
            "surprise_score": float(surprise),
            "emotional_weight": float(emotional_weight),
            "associative_keys": self._extract_associative_keys(text),
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": 0,
            "decay_factor": 1.0,  # Starts full, decays over time
            "metadata": metadata or {},
            "checksum": hashlib.md5(text.encode()).hexdigest()
        }

        return memory

    def encode_conversation(self, messages: list[dict]) -> list[dict]:
        """
        Encode a full conversation at once.

        Args:
            messages: [{"role": "user", "content": "..."}, ...]

        Returns:
            List of memory objects
        """
        memories = []
        for msg in messages:
            memory = self.encode(
                text=msg.get("content", ""),
                role=msg.get("role", "user")
            )
            memories.append(memory)
        return memories