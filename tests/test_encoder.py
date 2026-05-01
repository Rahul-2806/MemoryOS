import sys
sys.path.append("C:/PROJECT/MemoryOS")

from src.encoder.episodic_encoder import EpisodicEncoder

def test_encoder():
    encoder = EpisodicEncoder()

    # Test single memory
    print("\n--- Test 1: Single Memory ---")
    memory = encoder.encode(
        text="I am building MemoryOS, a neurologically inspired memory system for LLMs.",
        role="user"
    )
    print(f"Memory ID     : {memory['memory_id']}")
    print(f"Importance    : {memory['importance_score']:.4f}")
    print(f"Surprise      : {memory['surprise_score']:.4f}")
    print(f"Emotional Wt  : {memory['emotional_weight']:.4f}")
    print(f"Assoc. Keys   : {memory['associative_keys']}")
    print(f"Vector Shape  : {len(memory['semantic_vector'])} dims")

    # Test conversation
    print("\n--- Test 2: Full Conversation ---")
    convo = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Tell me about the Eiffel Tower."},
        {"role": "assistant", "content": "The Eiffel Tower is an iron lattice tower built in 1889."},
        {"role": "user", "content": "I want to visit Paris next summer with my family!"},
    ]
    memories = encoder.encode_conversation(convo)
    print(f"Encoded {len(memories)} memories")
    for i, m in enumerate(memories):
        print(f"  [{i+1}] importance={m['importance_score']:.3f} | keys={m['associative_keys'][:3]}")

    print("\n✅ Encoder working perfectly!")

if __name__ == "__main__":
    test_encoder()