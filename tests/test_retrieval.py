import sys
sys.path.append("C:/PROJECT/MemoryOS")

from src.encoder.episodic_encoder import EpisodicEncoder
from src.memory.memory_store import MemoryStore
from src.retrieval.associative_graph import AssociativeGraph
from src.retrieval.graph_walker import GraphWalker

def test_retrieval():
    # Setup
    encoder = EpisodicEncoder()
    store = MemoryStore()
    graph = AssociativeGraph()
    walker = GraphWalker(graph, store)

    # Load existing memories from DB into graph
    print("\n--- Building Graph from stored memories ---")
    all_memories = store.get_top_memories(limit=100)
    graph.add_memories(all_memories)
    stats = graph.get_node_stats()
    print(f"Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges, avg_degree={stats['avg_degree']}")

    # Query 1 — Career related
    print("\n--- Query 1: Career & FAANG ---")
    query = "I want to work at a top tech company"
    query_memory = encoder.encode(query, role="user")
    results = walker.walk(
        query_vector=query_memory["semantic_vector"],
        query_keys=query_memory["associative_keys"],
        max_hops=3,
        max_results=5
    )
    print(f"Retrieved {len(results)} memories:")
    for r in results:
        print(f"  [{r['importance_score']:.3f}] {r['text'][:70]}...")

    # Query 2 — Personal info
    print("\n--- Query 2: Personal Background ---")
    query2 = "Tell me about where you are from"
    query_memory2 = encoder.encode(query2, role="user")
    results2 = walker.walk(
        query_vector=query_memory2["semantic_vector"],
        query_keys=query_memory2["associative_keys"],
        max_hops=3,
        max_results=5
    )
    print(f"Retrieved {len(results2)} memories:")
    for r in results2:
        print(f"  [{r['importance_score']:.3f}] {r['text'][:70]}...")

    # Graph export for visualization
    print("\n--- Graph Export (for D3 frontend) ---")
    export = graph.export_for_viz()
    print(f"Nodes: {len(export['nodes'])}, Edges: {len(export['edges'])}")

    print("\n✅ Associative Graph + GraphWalker working!")

if __name__ == "__main__":
    test_retrieval()
    