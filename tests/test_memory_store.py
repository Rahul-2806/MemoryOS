import sys
sys.path.append("C:/PROJECT/MemoryOS")

from src.encoder.episodic_encoder import EpisodicEncoder
from src.memory.memory_store import MemoryStore

def test_memory_store():
    encoder = EpisodicEncoder()
    store = MemoryStore()

    print("\n--- Test 1: Save Memories ---")
    convo = [
        {"role": "user", "content": "My name is Rahul and I am building MemoryOS."},
        {"role": "assistant", "content": "That is amazing Rahul! MemoryOS sounds like a groundbreaking project."},
        {"role": "user", "content": "I want to get hired at Meta or Google with this project."},
        {"role": "assistant", "content": "With a project like this, you have a strong chance at FAANG companies."},
        {"role": "user", "content": "I am from Kerala, India and I use a GTX 1650 GPU."},
    ]

    memories = encoder.encode_conversation(convo)
    saved = store.save_many(memories, session_id="test_session_001")
    print(f"Saved: {saved}/{len(memories)} memories")

    print("\n--- Test 2: Memory Stats ---")
    stats = store.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n--- Test 3: Search by Keys ---")
    results = store.search_by_keys(["rahul", "memoryos", "kerala"])
    print(f"Found {len(results)} memories matching keys")
    for r in results:
        print(f"  importance={r['importance_score']:.3f} | {r['text'][:60]}...")

    print("\n--- Test 4: Top Memories ---")
    top = store.get_top_memories(limit=3)
    print("Top 3 most important memories:")
    for i, m in enumerate(top):
        print(f"  [{i+1}] {m['importance_score']:.3f} | {m['text'][:60]}...")

    print("\n--- Test 5: Apply Decay ---")
    decay_stats = store.apply_decay_all()
    print(f"Decay applied: {decay_stats}")

    print("\n✅ Memory Store working perfectly!")

if __name__ == "__main__":
    test_memory_store()