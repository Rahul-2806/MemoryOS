import networkx as nx
import numpy as np
from datetime import datetime, timezone


class AssociativeGraph:
    """
    Graph where memories are nodes and associations are weighted edges.

    Unlike RAG (flat vector search), this builds a NETWORK of memories
    where related memories link to each other through:
    1. Shared associative keys (keyword overlap)
    2. Semantic similarity (vector cosine similarity)
    3. Temporal proximity (memories close in time)
    4. Co-access patterns (memories retrieved together)

    Retrieval = graph traversal, not search.
    One memory triggers a chain of related memories.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        print("[MemoryOS] Associative Graph initialized ✓")

    def add_memory(self, memory: dict, session_id: str = None):
        """Add a memory as a node in the graph."""
        self.graph.add_node(
            memory["memory_id"],
            text=memory["text"],
            role=memory.get("role", "user"),
            importance=memory["importance_score"],
            keys=memory["associative_keys"],
            vector=memory["semantic_vector"],
            created_at=memory["created_at"],
            access_count=memory.get("access_count", 0),
            session_id=session_id or memory.get("metadata", {}).get("session_id", "default")
        )

    def add_memories(self, memories: list, session_id: str = None):
        """Add multiple memories and auto-build associations."""
        for memory in memories:
            self.add_memory(memory, session_id=session_id)
        self._build_associations(memories, session_id=session_id)

    def _build_associations(self, memories: list, session_id: str = None):
        """
        Build edges between memories based on 3 signals:
        1. Keyword overlap
        2. Semantic similarity
        3. Temporal proximity
        Only links memories within the same session.
        """
        for i, mem_a in enumerate(memories):
            for j, mem_b in enumerate(memories):
                if i >= j:
                    continue

                weight = 0.0

                # Signal 1 — Keyword overlap
                keys_a = set(mem_a["associative_keys"])
                keys_b = set(mem_b["associative_keys"])
                overlap = keys_a & keys_b
                if overlap:
                    keyword_score = len(overlap) / max(len(keys_a | keys_b), 1)
                    weight += keyword_score * 0.4

                # Signal 2 — Semantic similarity
                vec_a = np.array(mem_a["semantic_vector"])
                vec_b = np.array(mem_b["semantic_vector"])
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                if norm_a > 0 and norm_b > 0:
                    cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                    weight += max(0.0, cosine_sim) * 0.4

                # Signal 3 — Temporal proximity
                try:
                    t_a = datetime.fromisoformat(mem_a["created_at"].replace("Z", "+00:00"))
                    t_b = datetime.fromisoformat(mem_b["created_at"].replace("Z", "+00:00"))
                    if t_a.tzinfo is None:
                        t_a = t_a.replace(tzinfo=timezone.utc)
                    if t_b.tzinfo is None:
                        t_b = t_b.replace(tzinfo=timezone.utc)
                    time_diff = abs((t_a - t_b).total_seconds())
                    temporal_score = max(0.0, 1.0 - (time_diff / 60.0))
                    weight += temporal_score * 0.2
                except Exception:
                    pass

                if weight > 0.15:
                    self.graph.add_edge(
                        mem_a["memory_id"], mem_b["memory_id"],
                        weight=weight, overlap_keys=list(overlap)
                    )
                    self.graph.add_edge(
                        mem_b["memory_id"], mem_a["memory_id"],
                        weight=weight, overlap_keys=list(overlap)
                    )

    def strengthen_edge(self, memory_id_a: str, memory_id_b: str, boost: float = 0.05):
        """
        When two memories are retrieved together, strengthen their link.
        Simulates Hebbian learning — neurons that fire together wire together.
        """
        if self.graph.has_edge(memory_id_a, memory_id_b):
            current = self.graph[memory_id_a][memory_id_b]["weight"]
            self.graph[memory_id_a][memory_id_b]["weight"] = min(1.0, current + boost)
        else:
            self.graph.add_edge(memory_id_a, memory_id_b, weight=boost, overlap_keys=[])

    def get_node_stats(self) -> dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "avg_degree": round(
                sum(d for _, d in self.graph.degree()) / max(self.graph.number_of_nodes(), 1),
                2
            )
        }

    def export_for_viz(self) -> dict:
        """Export graph structure for frontend D3 visualization."""
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "text": data.get("text", "")[:80],
                "importance": data.get("importance", 0.5),
                "role": data.get("role", "user"),
                "keys": data.get("keys", []),
                "session_id": data.get("session_id", "default")
            })

        edges = []
        for src, dst, data in self.graph.edges(data=True):
            edges.append({
                "source": src,
                "target": dst,
                "weight": round(data.get("weight", 0.0), 3),
                "keys": data.get("overlap_keys", [])
            })

        return {"nodes": nodes, "edges": edges}