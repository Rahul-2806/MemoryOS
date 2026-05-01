# MemoryOS

**Persistent episodic memory architecture for LLMs — built from scratch.**

Most LLM memory systems are RAG with a database. This is not that. MemoryOS encodes memories the way a brain does: with importance weights, associative links, and decay over time. Memories that matter survive. Ones that don't, fade.

**Live demo:** [memory-os-tau.vercel.app](https://memory-os-tau.vercel.app)  
**Backend API:** [rahulsr2806-memoryos-backend.hf.space](https://rahulsr2806-memoryos-backend.hf.space)

---

## The Problem

Every LLM conversation starts from zero. RAG helps — but RAG is just a search engine over a flat vector store. It finds the most *similar* memory, not the most *relevant* one. It has no concept of importance, no decay, no associative chains.

A doctor who forgot every patient after each appointment would not be useful. Neither is an AI that does the same.

---

## How MemoryOS Works

Four components, each doing something RAG doesn't.

### 1. Episodic Encoder
Converts raw text into structured memory objects. Each memory gets:
- A 384-dim semantic vector (MiniLM-L6-v2, runs on CUDA)
- An importance score — computed by a neural network that measures surprise against recent context
- Associative keys — content words used to build graph edges
- Emotional weight — a proxy using length and punctuation density

Importance is not a heuristic. It's learned. Surprising or emotionally loaded memories score higher automatically.

### 2. Memory Store + Ebbinghaus Decay
Memories persist in SQLite. Every memory has a `decay_factor` that decreases over time using the Ebbinghaus forgetting curve:

```
R = e^(-t / S)
```

Where `t` is hours since last access and `S` is stability (importance + access count). Important memories decay slowly. Trivial ones disappear. Accessing a memory boosts its stability — exactly like spaced repetition.

### 3. Associative Graph + Graph Walker
Every memory is a node. Edges form between memories based on three signals: keyword overlap, cosine similarity, and temporal proximity. This produces a graph where related memories cluster together.

Retrieval works by graph traversal, not search. Given a query, the walker:
1. Finds seed nodes via cosine similarity
2. Walks outward along strong edges (BFS, max 2 hops)
3. Applies keyword boosts at each node
4. Returns ranked results

This means the system retrieves contextually connected memories, not just the closest vectors. A question about your career might surface memories about your location, your tools, and your goals — because they're all linked in the graph.

Edges strengthen on co-retrieval (Hebbian learning: neurons that fire together wire together).

### 4. Consolidation Engine
Runs as a background thread every 5 minutes. Merges memories with >85% cosine similarity, prunes memories whose retention has fallen below threshold, and boosts memories that are frequently accessed. Like sleep consolidation in humans.

---

## Benchmark

Evaluated against standard cosine-similarity RAG across 3 user profiles and 15 query types.

| Metric | MemoryOS | RAG | Gain |
|---|---|---|---|
| MRR | 0.7111 | 0.6667 | **+6.7%** |
| Precision@5 | 0.2800 | 0.2800 | matched |
| Recall@5 | 0.8167 | 0.8444 | -3.3% |
| Query wins | 5/15 | 2/15 | **2.5x** |

MRR (Mean Reciprocal Rank) measures whether the most relevant memory surfaces first. MemoryOS wins on this by 6.7% — which is the metric that actually matters for an LLM context window, where position one gets the most attention.

---

## Architecture

```
User Message
     │
     ▼
┌─────────────────┐
│ Episodic Encoder│  ← MiniLM-L6-v2 (CUDA)
│ + ImportanceNet │  ← 2-layer MLP scorer
└────────┬────────┘
         │ memory object
         ▼
┌─────────────────┐     ┌──────────────────────┐
│  Memory Store   │────▶│  Associative Graph   │
│  SQLite +       │     │  NetworkX DiGraph     │
│  Decay Engine   │     │  3-signal edge builder│
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────┐
│              Graph Walker                   │
│  seed nodes → BFS traversal → keyword boost │
│  → Hebbian edge strengthening               │
└──────────────────────┬──────────────────────┘
                       │ top-k memories
                       ▼
              ┌─────────────────┐
              │   Groq API      │  ← LLaMA 3.3 70B
              │  + memory ctx   │
              └────────┬────────┘
                       │
                       ▼
                   Response
```

```
┌──────────────────────────────────────────┐
│         Consolidation Engine             │
│  runs every 300s in background thread    │
│  merge (sim > 0.85) → decay → prune      │
└──────────────────────────────────────────┘
```

---

## Stack

| Layer | Tech |
|---|---|
| Encoder | PyTorch, sentence-transformers (MiniLM-L6-v2) |
| Graph | NetworkX, custom TGN-inspired edge builder |
| Decay | Ebbinghaus curve, NumPy |
| Storage | SQLite, custom ORM |
| API | FastAPI, Uvicorn |
| LLM | Groq (LLaMA 3.3 70B) |
| Frontend | Next.js 15, D3.js, Framer Motion |
| Backend deploy | HuggingFace Spaces (Docker) |
| Frontend deploy | Vercel |

---

## Run Locally

```bash
git clone https://github.com/Rahul-2806/MemoryOS.git
cd MemoryOS

pip install -r requirements.txt

# Add your Groq key
echo "GROQ_API_KEY=your_key_here" > .env

# Start backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

---

## API

```
GET  /health           — system status, graph stats, consolidation stats
POST /chat             — chat with memory context
POST /memory/add       — manually store a memory
GET  /memory/all       — retrieve top memories by importance
GET  /graph/export     — export graph as nodes/edges for visualization
POST /consolidate      — trigger manual consolidation
DELETE /memory/clear   — clear all memories
```

Example:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is Rahul and I am building MemoryOS", "session_id": "demo"}'
```

---

## Project Structure

```
MemoryOS/
├── src/
│   ├── encoder/
│   │   └── episodic_encoder.py     # MiniLM + ImportanceNet
│   ├── memory/
│   │   ├── memory_store.py         # SQLite persistence
│   │   └── decay.py                # Ebbinghaus forgetting curve
│   ├── retrieval/
│   │   ├── associative_graph.py    # NetworkX graph builder
│   │   └── graph_walker.py         # BFS traversal + Hebbian
│   ├── consolidation/
│   │   └── consolidator.py         # Background merge + prune
│   ├── benchmark/
│   │   └── evaluate.py             # MemoryOS vs RAG benchmark
│   └── api/
│       └── main.py                 # FastAPI server
└── frontend/
    └── app/
        ├── page.tsx                # Main dashboard
        ├── MemoryGraph.tsx         # D3 force graph
        └── globals.css             # Design system
```

---

## Built by

Rahul R — Data Scientist & AI Engineer from Kerala, India.  
[Portfolio](https://rahulaiportfolio.online) · [LinkedIn](https://linkedin.com/in/rahulsr2806) · [GitHub](https://github.com/Rahul-2806)