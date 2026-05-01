import sys
sys.path.append("C:/PROJECT/MemoryOS")

import numpy as np
import time
import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch

from src.encoder.episodic_encoder import EpisodicEncoder
from src.memory.memory_store import MemoryStore
from src.retrieval.associative_graph import AssociativeGraph
from src.retrieval.graph_walker import GraphWalker

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaselineRAG:
    """Standard RAG — flat cosine similarity, no graph, no decay."""

    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        self.store = []

    def add(self, text: str, session_id: str = "default"):
        vec = self.embedder.encode(text, convert_to_tensor=False).tolist()
        self.store.append({"text": text, "vector": vec, "session_id": session_id})

    def retrieve(self, query: str, top_k: int = 5, session_id: str = None) -> List[str]:
        candidates = [
            item for item in self.store
            if session_id is None or item["session_id"] == session_id
        ]
        if not candidates:
            return []
        query_vec = np.array(self.embedder.encode(query, convert_to_tensor=False))
        scores = []
        for item in candidates:
            vec = np.array(item["vector"])
            sim = float(np.dot(query_vec, vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-9
            ))
            scores.append((item["text"], sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scores[:top_k]]


BENCHMARK_CONVERSATIONS = [
    {
        "session": "user_001",
        "memories": [
            "My name is Arjun and I work as a machine learning engineer at a startup.",
            "I graduated from IIT Bombay with a degree in computer science in 2021.",
            "My favorite programming language is Python and I use PyTorch for all my ML work.",
            "I am currently working on a recommendation system for e-commerce.",
            "I love playing chess and reading about quantum computing in my free time.",
            "My biggest challenge right now is scaling my model to handle 10 million users.",
            "I previously worked at Infosys for two years before joining the startup.",
            "I am planning to apply for a PhD program at MIT next year.",
            "My research interest is in reinforcement learning and multi-agent systems.",
            "I use a MacBook Pro M2 with 32GB RAM for all my development work.",
        ],
        "queries": [
            {
                "question": "What is my educational background?",
                "relevant_keywords": ["iit", "bombay", "computer", "science", "graduated", "2021"]
            },
            {
                "question": "What are my hobbies and interests?",
                "relevant_keywords": ["chess", "quantum", "computing", "free", "time"]
            },
            {
                "question": "What challenges am I facing at work?",
                "relevant_keywords": ["scaling", "model", "million", "users", "challenge"]
            },
            {
                "question": "What are my career plans?",
                "relevant_keywords": ["phd", "mit", "apply", "reinforcement", "research"]
            },
            {
                "question": "What tools and hardware do I use?",
                "relevant_keywords": ["macbook", "pytorch", "python", "32gb", "development"]
            },
        ]
    },
    {
        "session": "user_002",
        "memories": [
            "I am Sofia and I am a data scientist at a healthcare company in Berlin.",
            "I specialize in natural language processing and medical text analysis.",
            "My team is building a clinical decision support system using BERT.",
            "I have published three papers on biomedical NLP at top conferences.",
            "I am originally from Portugal and moved to Germany five years ago.",
            "I struggle with class imbalance problems in medical datasets.",
            "My company uses AWS for cloud infrastructure and Kubernetes for deployment.",
            "I am learning Rust in my spare time to improve my systems programming skills.",
            "My biggest achievement was reducing diagnostic error rates by 23% with my model.",
            "I mentor junior data scientists and run a weekly ML reading group.",
        ],
        "queries": [
            {
                "question": "What is this person's area of expertise?",
                "relevant_keywords": ["nlp", "natural", "language", "medical", "biomedical", "bert"]
            },
            {
                "question": "What technical challenges does this person face?",
                "relevant_keywords": ["class", "imbalance", "medical", "datasets", "struggle"]
            },
            {
                "question": "What is the person's biggest professional achievement?",
                "relevant_keywords": ["diagnostic", "error", "23", "reducing", "achievement"]
            },
            {
                "question": "What is this person's background and origin?",
                "relevant_keywords": ["portugal", "germany", "berlin", "moved", "originally"]
            },
            {
                "question": "What is this person currently learning?",
                "relevant_keywords": ["rust", "systems", "programming", "learning", "spare"]
            }
        ]
    },
    {
        "session": "user_003",
        "memories": [
            "I am Priya and I am a product manager at a fintech startup.",
            "Our product helps small businesses manage their cash flow and invoicing.",
            "I studied economics at Delhi University and then got an MBA from ISB.",
            "My team consists of 8 people including 3 engineers and 2 designers.",
            "We recently raised a Series A round of 5 million dollars.",
            "Our biggest competitor is a company called ClearBooks based in London.",
            "I am struggling with user retention — our 30-day retention is only 34%.",
            "I use Notion for all my documentation and Linear for project tracking.",
            "My co-founder handles all the technical architecture decisions.",
            "We are planning to expand to Southeast Asia in Q3 of this year.",
            "I previously worked at Paytm for three years in the growth team.",
            "Our current MRR is 180,000 dollars and growing at 15% month over month.",
            "I am most proud of closing our first enterprise deal with HDFC Bank.",
            "We use a freemium model with paid plans starting at 49 dollars per month.",
            "My biggest challenge is hiring senior engineers in the current market.",
        ],
        "queries": [
            {
                "question": "What is the company's financial performance?",
                "relevant_keywords": ["mrr", "180000", "series", "million", "growing", "15"]
            },
            {
                "question": "What are the biggest challenges this person faces?",
                "relevant_keywords": ["retention", "34", "hiring", "engineers", "challenge", "struggling"]
            },
            {
                "question": "What is the person's educational and work background?",
                "relevant_keywords": ["delhi", "isb", "mba", "economics", "paytm", "growth"]
            },
            {
                "question": "What tools does this person use for work?",
                "relevant_keywords": ["notion", "linear", "documentation", "tracking"]
            },
            {
                "question": "What are the company's expansion and competitive plans?",
                "relevant_keywords": ["southeast", "asia", "clearbooks", "competitor", "expand", "q3"]
            }
        ]
    }
]


def precision_at_k(retrieved: List[str], relevant_keywords: List[str], k: int = 5) -> float:
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(
        1 for text in retrieved_k
        if any(kw.lower() in text.lower() for kw in relevant_keywords)
    )
    return hits / len(retrieved_k)


def recall_at_k(retrieved: List[str], all_memories: List[str],
                relevant_keywords: List[str], k: int = 5) -> float:
    relevant_memories = [
        m for m in all_memories
        if any(kw.lower() in m.lower() for kw in relevant_keywords)
    ]
    if not relevant_memories:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(
        1 for m in relevant_memories
        if any(m.lower() in r.lower() or r.lower() in m.lower() for r in retrieved_k)
    )
    return min(1.0, hits / len(relevant_memories))


def mrr_score(retrieved: List[str], relevant_keywords: List[str]) -> float:
    for i, text in enumerate(retrieved):
        if any(kw.lower() in text.lower() for kw in relevant_keywords):
            return 1.0 / (i + 1)
    return 0.0


class BenchmarkRunner:

    def __init__(self):
        self.encoder = EpisodicEncoder()
        self.store = MemoryStore()
        self.graph = AssociativeGraph()
        self.walker = GraphWalker(self.graph, self.store)
        self.rag = BaselineRAG()

    def run(self) -> Dict:
        print("\n" + "="*60)
        print("  MEMORYOS vs BASELINE RAG — BENCHMARK")
        print("="*60)

        memoryos_scores = {"precision": [], "recall": [], "mrr": [], "latency": []}
        rag_scores      = {"precision": [], "recall": [], "mrr": [], "latency": []}
        memoryos_wins = rag_wins = ties = 0

        for session_data in BENCHMARK_CONVERSATIONS:
            session_id    = session_data["session"]
            memories_text = session_data["memories"]
            queries       = session_data["queries"]

            print(f"\n[Session: {session_id}] — {len(memories_text)} memories")

            # Load into MemoryOS with session_id on every node
            encoded = self.encoder.encode_conversation([
                {"role": "user", "content": t} for t in memories_text
            ])
            self.store.save_many(encoded, session_id=session_id)
            self.graph.add_memories(encoded, session_id=session_id)

            # Load into RAG with session tag
            for text in memories_text:
                self.rag.add(text, session_id=session_id)

            for q in queries:
                question = q["question"]
                keywords = q["relevant_keywords"]

                # ── MemoryOS ──
                t0 = time.perf_counter()
                q_memory = self.encoder.encode(question, role="user")
                results_m = self.walker.walk(
                    query_vector=q_memory["semantic_vector"],
                    query_keys=q_memory["associative_keys"],
                    max_hops=2,
                    max_results=5,
                    session_id=session_id          # ← session isolation
                )
                t1 = time.perf_counter()
                texts_m = [m["text"] for m in results_m]
                lat_m   = (t1 - t0) * 1000

                p_m   = precision_at_k(texts_m, keywords)
                r_m   = recall_at_k(texts_m, memories_text, keywords)
                mrr_m = mrr_score(texts_m, keywords)

                memoryos_scores["precision"].append(p_m)
                memoryos_scores["recall"].append(r_m)
                memoryos_scores["mrr"].append(mrr_m)
                memoryos_scores["latency"].append(lat_m)

                # ── RAG ──
                t2 = time.perf_counter()
                texts_r = self.rag.retrieve(question, top_k=5, session_id=session_id)
                t3 = time.perf_counter()
                lat_r   = (t3 - t2) * 1000

                p_r   = precision_at_k(texts_r, keywords)
                r_r   = recall_at_k(texts_r, memories_text, keywords)
                mrr_r = mrr_score(texts_r, keywords)

                rag_scores["precision"].append(p_r)
                rag_scores["recall"].append(r_r)
                rag_scores["mrr"].append(mrr_r)
                rag_scores["latency"].append(lat_r)

                if mrr_m > mrr_r:
                    winner = "MemoryOS ✅"; memoryos_wins += 1
                elif mrr_r > mrr_m:
                    winner = "RAG ❌";      rag_wins += 1
                else:
                    winner = "Tie ➖";      ties += 1

                print(f"  Q: {question[:52]}...")
                print(f"     MemoryOS → P={p_m:.2f} R={r_m:.2f} MRR={mrr_m:.2f} ({lat_m:.1f}ms)")
                print(f"     RAG      → P={p_r:.2f} R={r_r:.2f} MRR={mrr_r:.2f} ({lat_r:.1f}ms)")
                print(f"     Winner   → {winner}")

        def avg(lst): return round(sum(lst)/len(lst), 4) if lst else 0

        results = {
            "MemoryOS":    {
                "precision@5":    avg(memoryos_scores["precision"]),
                "recall@5":       avg(memoryos_scores["recall"]),
                "mrr":            avg(memoryos_scores["mrr"]),
                "avg_latency_ms": round(avg(memoryos_scores["latency"]), 2)
            },
            "BaselineRAG": {
                "precision@5":    avg(rag_scores["precision"]),
                "recall@5":       avg(rag_scores["recall"]),
                "mrr":            avg(rag_scores["mrr"]),
                "avg_latency_ms": round(avg(rag_scores["latency"]), 2)
            }
        }

        p_gain   = round((results["MemoryOS"]["precision@5"] - results["BaselineRAG"]["precision@5"])
                         / max(results["BaselineRAG"]["precision@5"], 0.001) * 100, 1)
        r_gain   = round((results["MemoryOS"]["recall@5"] - results["BaselineRAG"]["recall@5"])
                         / max(results["BaselineRAG"]["recall@5"], 0.001) * 100, 1)
        mrr_gain = round((results["MemoryOS"]["mrr"] - results["BaselineRAG"]["mrr"])
                         / max(results["BaselineRAG"]["mrr"], 0.001) * 100, 1)

        total = memoryos_wins + rag_wins + ties

        print("\n" + "="*60)
        print("  FINAL RESULTS")
        print("="*60)
        print(f"\n{'Metric':<20} {'MemoryOS':>12} {'RAG':>12} {'Gain':>10}")
        print("-"*56)
        print(f"{'Precision@5':<20} {results['MemoryOS']['precision@5']:>12.4f} {results['BaselineRAG']['precision@5']:>12.4f} {p_gain:>+9.1f}%")
        print(f"{'Recall@5':<20} {results['MemoryOS']['recall@5']:>12.4f} {results['BaselineRAG']['recall@5']:>12.4f} {r_gain:>+9.1f}%")
        print(f"{'MRR':<20} {results['MemoryOS']['mrr']:>12.4f} {results['BaselineRAG']['mrr']:>12.4f} {mrr_gain:>+9.1f}%")
        print(f"{'Latency (ms)':<20} {results['MemoryOS']['avg_latency_ms']:>12.2f} {results['BaselineRAG']['avg_latency_ms']:>12.2f}")
        print("="*60)
        print(f"\nQuery wins → MemoryOS: {memoryos_wins}/{total} | RAG: {rag_wins}/{total} | Ties: {ties}/{total}")
        print("="*60)

        with open("C:/PROJECT/MemoryOS/data/benchmark_results.json", "w") as f:
            json.dump({
                "results": results,
                "gains": {"precision": p_gain, "recall": r_gain, "mrr": mrr_gain},
                "query_wins": {"memoryos": memoryos_wins, "rag": rag_wins,
                               "ties": ties, "total": total}
            }, f, indent=2)
        print("\n✅ Results saved to data/benchmark_results.json")
        return results


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run()