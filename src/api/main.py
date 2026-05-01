import sys
sys.path.append("C:/PROJECT/MemoryOS")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv
from groq import Groq

from src.encoder.episodic_encoder import EpisodicEncoder
from src.memory.memory_store import MemoryStore
from src.retrieval.associative_graph import AssociativeGraph
from src.retrieval.graph_walker import GraphWalker
from src.consolidation.consolidator import MemoryConsolidator

load_dotenv("C:/PROJECT/MemoryOS/.env")

# ── Init all components ──────────────────────────────────────────
app = FastAPI(
    title="MemoryOS",
    description="Neurologically-inspired persistent memory for LLMs",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

encoder   = EpisodicEncoder()
store     = MemoryStore()
graph     = AssociativeGraph()
walker    = GraphWalker(graph, store)
consolidator = MemoryConsolidator(store, interval_seconds=300)
groq_client  = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load existing memories into graph on startup
_existing = store.get_top_memories(limit=1000)
if _existing:
    graph.add_memories(_existing)
    print(f"[MemoryOS] Loaded {len(_existing)} memories into graph ✓")

consolidator.start()


# ── Request/Response Models ───────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    user_id: str = "user"

class ChatResponse(BaseModel):
    reply: str
    memories_used: int
    memory_ids: list[str]
    graph_stats: dict
    session_id: str

class MemoryRequest(BaseModel):
    text: str
    role: str = "user"
    session_id: str = "default"

class ConsolidateRequest(BaseModel):
    run_now: bool = True


# ── Routes ────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "system": "MemoryOS",
        "status": "online",
        "version": "1.0.0",
        "description": "Neurologically-inspired persistent memory for LLMs"
    }

@app.get("/health")
def health():
    stats = store.get_stats()
    graph_stats = graph.get_node_stats()
    return {
        "status": "healthy",
        "memory_store": stats,
        "graph": graph_stats,
        "consolidator": consolidator.stats
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    1. Encode user message
    2. Retrieve relevant memories via graph walk
    3. Build context-aware prompt
    4. Call Groq LLaMA 3.3 70B
    5. Save both user message and AI reply as memories
    """
    try:
        # Step 1 — Encode user message
        user_memory = encoder.encode(req.message, role="user")

        # Step 2 — Retrieve relevant memories
        relevant = walker.walk(
            query_vector=user_memory["semantic_vector"],
            query_keys=user_memory["associative_keys"],
            max_hops=3,
            max_results=6
        )

        # Step 3 — Build memory context string
        memory_context = ""
        if relevant:
            memory_context = "\n\n[MEMORY CONTEXT — what I remember about you]\n"
            for i, mem in enumerate(relevant, 1):
                memory_context += f"{i}. [{mem['role']}]: {mem['text']}\n"
            memory_context += "[END MEMORY]\n"

        # Step 4 — Build system prompt
        system_prompt = f"""You are MemoryOS — an AI assistant with genuine persistent memory.
You remember past conversations and use them naturally in your responses.
You never say "as an AI I don't have memory" — you DO have memory.
When relevant memories exist, reference them naturally like a human would.
Be concise, intelligent, and helpful.
{memory_context}"""

        # Step 5 — Call Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message}
            ],
            max_tokens=1024,
            temperature=0.7
        )
        reply = response.choices[0].message.content

        # Step 6 — Save user message to memory
        store.save(user_memory, session_id=req.session_id)
        graph.add_memory(user_memory)

        # Step 7 — Save AI reply to memory
        ai_memory = encoder.encode(reply, role="assistant")
        store.save(ai_memory, session_id=req.session_id)
        graph.add_memory(ai_memory)

        # Step 8 — Build associations for new memories
        graph._build_associations([user_memory, ai_memory])

        return ChatResponse(
            reply=reply,
            memories_used=len(relevant),
            memory_ids=[m["memory_id"] for m in relevant],
            graph_stats=graph.get_node_stats(),
            session_id=req.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/add")
async def add_memory(req: MemoryRequest):
    """Manually add a memory."""
    memory = encoder.encode(req.text, role=req.role)
    saved = store.save(memory, session_id=req.session_id)
    graph.add_memory(memory)
    return {
        "saved": saved,
        "memory_id": memory["memory_id"],
        "importance": memory["importance_score"]
    }


@app.get("/memory/all")
async def get_all_memories(limit: int = 50):
    """Get top memories."""
    memories = store.get_top_memories(limit=limit)
    return {
        "count": len(memories),
        "memories": [
            {
                "memory_id": m["memory_id"],
                "text": m["text"][:100],
                "importance": m["importance_score"],
                "role": m["role"],
                "created_at": m["created_at"]
            }
            for m in memories
        ]
    }


@app.get("/graph/export")
async def export_graph():
    """Export graph for D3 visualization."""
    return graph.export_for_viz()


@app.post("/consolidate")
async def consolidate(req: ConsolidateRequest):
    """Manually trigger consolidation."""
    if req.run_now:
        stats = consolidator.consolidate_once()
        return {"status": "done", "stats": stats}
    return {"status": "skipped"}


@app.delete("/memory/clear")
async def clear_memories():
    """Clear all memories (dev use only)."""
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DELETE FROM memories")
        conn.commit()
    graph.graph.clear()
    return {"status": "cleared"}