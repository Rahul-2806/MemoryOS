import numpy as np
from src.retrieval.associative_graph import AssociativeGraph
from src.memory.memory_store import MemoryStore


class GraphWalker:
    """
    Associative retrieval via graph traversal.

    Instead of: "find the closest vector" (RAG)
    This does:  "start from relevant nodes, walk the graph,
                 collect everything strongly connected"

    Result: Retrieves memories that are CONTEXTUALLY related,
            not just semantically similar. Like human recall.
    """

    def __init__(self, graph: AssociativeGraph, store: MemoryStore):
        self.graph = graph
        self.store = store

    def _get_seed_nodes(self, query_vector: list, top_k: int = 3,
                        session_filter: set = None) -> list:
        query = np.array(query_vector)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        scores = []
        for node_id, data in self.graph.graph.nodes(data=True):
            if session_filter and data.get("session_id") not in session_filter:
                continue
            node_vec = np.array(data.get("vector", []))
            if len(node_vec) == 0:
                continue
            node_norm = np.linalg.norm(node_vec)
            if node_norm == 0:
                continue
            sim = float(np.dot(query, node_vec) / (query_norm * node_norm))
            scores.append((node_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in scores[:top_k]]

    def walk(
        self,
        query_vector: list,
        query_keys: list,
        max_hops: int = 2,
        max_results: int = 5,
        min_weight: float = 0.15,
        session_id: str = None
    ) -> list:
        """
        Core graph walk retrieval — session-isolated, similarity-first.

        Primary signal  (70%): direct cosine similarity
        Secondary signal(15%): graph traversal bonus
        Tertiary signal ( 8%): keyword overlap boost
        """
        if self.graph.graph.number_of_nodes() == 0:
            return []

        session_filter = {session_id} if session_id else None

        # Step 1 — Direct cosine similarity for ALL nodes (PRIMARY signal)
        query = np.array(query_vector)
        query_norm = np.linalg.norm(query)

        visited = {}
        if query_norm > 0:
            for node_id, data in self.graph.graph.nodes(data=True):
                if session_filter and data.get("session_id") not in session_filter:
                    continue
                node_vec = np.array(data.get("vector", []))
                if len(node_vec) == 0:
                    continue
                node_norm = np.linalg.norm(node_vec)
                if node_norm == 0:
                    continue
                sim = float(np.dot(query, node_vec) / (query_norm * node_norm))
                node_importance = data.get("importance", 0.5)
                visited[node_id] = (sim * 0.7) + (node_importance * 0.3)

        # Step 2 — Graph traversal bonus (SECONDARY signal, max +15%)
        seeds = self._get_seed_nodes(query_vector, top_k=3,
                                     session_filter=session_filter)
        for seed in seeds:
            frontier = [(seed, 1.0, 0)]
            while frontier:
                current, path_weight, hop = frontier.pop(0)
                if hop >= max_hops:
                    continue
                for neighbor in self.graph.graph.successors(current):
                    nbr_data = self.graph.graph.nodes.get(neighbor, {})
                    if session_filter and nbr_data.get("session_id") not in session_filter:
                        continue
                    edge_weight = self.graph.graph[current][neighbor].get("weight", 0.0)
                    if edge_weight < min_weight:
                        continue
                    cumulative = path_weight * edge_weight * (0.7 ** hop)
                    bonus = cumulative * 0.15
                    if neighbor in visited:
                        visited[neighbor] = min(1.0, visited[neighbor] + bonus)
                    frontier.append((neighbor, cumulative, hop + 1))

        # Step 3 — Keyword boost (TERTIARY signal, +8% per overlap)
        query_key_set = set(query_keys)
        for node_id, data in self.graph.graph.nodes(data=True):
            if session_filter and data.get("session_id") not in session_filter:
                continue
            node_keys = set(data.get("keys", []))
            overlap = node_keys & query_key_set
            if overlap and node_id in visited:
                boost = len(overlap) * 0.08
                visited[node_id] = min(1.0, visited[node_id] + boost)

        # Step 4 — Sort and return top results
        ranked = sorted(visited.items(), key=lambda x: x[1], reverse=True)
        top_ids = [node_id for node_id, _ in ranked[:max_results]]

        results = []
        for memory_id in top_ids:
            memory = self.store.get(memory_id)
            if memory:
                results.append(memory)

        # Hebbian learning
        for i in range(len(top_ids)):
            for j in range(i + 1, min(len(top_ids), 4)):
                self.graph.strengthen_edge(top_ids[i], top_ids[j])

        return results