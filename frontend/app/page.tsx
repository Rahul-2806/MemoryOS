"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import dynamic from "next/dynamic";

const MemoryGraph = dynamic(() => import("./MemoryGraph"), { ssr: false });

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Message  { role: "user"|"assistant"; content: string; memories_used?: number; }
interface Memory   { memory_id: string; text: string; importance: number; role: string; created_at: string; }
interface GraphData{ nodes: any[]; edges: any[]; }
interface Stats    { total_memories: number; avg_importance: number; total_sessions: number; }
interface Health   { memory_store: Stats; graph: { total_nodes: number; total_edges: number; avg_degree: number }; consolidator: { total_runs: number; total_merged: number } }

export default function Page() {
  const [messages,   setMessages]   = useState<Message[]>([]);
  const [input,      setInput]      = useState("");
  const [loading,    setLoading]    = useState(false);
  const [memories,   setMemories]   = useState<Memory[]>([]);
  const [graph,      setGraph]      = useState<GraphData>({ nodes: [], edges: [] });
  const [health,     setHealth]     = useState<Health|null>(null);
  const [sessionId, setSessionId] = useState("session_init");
  const [tab,        setTab]        = useState<"graph"|"memories"|"benchmark">("graph");
  const [benchmark,  setBenchmark]  = useState<any>(null);
  const [apiOnline,  setApiOnline]  = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Poll health + graph
  const refresh = useCallback(async () => {
    try {
      const [h, g, m] = await Promise.all([
        axios.get(`${API}/health`),
        axios.get(`${API}/graph/export`),
        axios.get(`${API}/memory/all?limit=30`),
      ]);
      setHealth(h.data);
      setGraph(g.data);
      setMemories(m.data.memories || []);
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load benchmark results
  useEffect(() => {
    fetch("/benchmark.json").then(r => r.ok ? r.json() : null).then(d => d && setBenchmark(d)).catch(() => {});
  }, []);

  useEffect(() => {
  setSessionId(`session_${Math.random().toString(36).slice(2, 10)}`);
}, []);

  const send = async () => {
    if (!input.trim() || loading) return;
    const text = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: text }]);
    setLoading(true);
    try {
      const res = await axios.post(`${API}/chat`, { message: text, session_id: sessionId });
      setMessages(prev => [...prev, {
        role: "assistant",
        content: res.data.reply,
        memories_used: res.data.memories_used
      }]);
      refresh();
    } catch (e: any) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: "⚠ API offline. Start the server: `python -m uvicorn src.api.main:app --port 8000`"
      }]);
    }
    setLoading(false);
  };

  const importanceColor = (v: number) =>
    v > 0.6 ? "var(--gold)" : v > 0.35 ? "var(--cyan)" : "var(--text3)";

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)", display: "flex", flexDirection: "column" }}>

      {/* ── Header ── */}
      <header style={{
        borderBottom: "1px solid var(--border)",
        padding: "0 32px",
        height: 56,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "var(--bg2)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Logo mark */}
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="10" stroke="var(--gold)" strokeWidth="1" opacity="0.4"/>
            <circle cx="12" cy="12" r="4" fill="var(--gold)" opacity="0.8"/>
            <line x1="12" y1="2" x2="12" y2="7" stroke="var(--gold)" strokeWidth="1" opacity="0.4"/>
            <line x1="12" y1="17" x2="12" y2="22" stroke="var(--gold)" strokeWidth="1" opacity="0.4"/>
            <line x1="2" y1="12" x2="7" y2="12" stroke="var(--gold)" strokeWidth="1" opacity="0.4"/>
            <line x1="17" y1="12" x2="22" y2="12" stroke="var(--gold)" strokeWidth="1" opacity="0.4"/>
          </svg>
          <span style={{ fontFamily: "Syne", fontWeight: 800, fontSize: 17, letterSpacing: "0.08em", color: "var(--gold)", textShadow: "0 0 20px rgba(201,168,76,0.3)" }}>
            MEMORY<span style={{ color: "var(--text2)" }}>OS</span>
          </span>
          <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text3)", letterSpacing: "0.15em" }}>v1.0</span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
          {health && (
            <div style={{ display: "flex", gap: 16, fontFamily: "JetBrains Mono", fontSize: 11, color: "var(--text2)" }}>
              <span><span style={{ color: "var(--gold)" }}>{health.memory_store.total_memories}</span> memories</span>
              <span><span style={{ color: "var(--cyan)" }}>{health.graph.total_nodes}</span> nodes</span>
              <span><span style={{ color: "var(--cyan)" }}>{health.graph.total_edges}</span> edges</span>
            </div>
          )}
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div className="pulse" style={{
              width: 6, height: 6, borderRadius: "50%",
              background: apiOnline ? "var(--green)" : "var(--red)"
            }}/>
            <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: apiOnline ? "var(--green)" : "var(--red)" }}>
              {apiOnline ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </div>
      </header>

      {/* ── Main grid ── */}
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 420px", gap: 0, height: "calc(100vh - 56px)" }}>

        {/* ── LEFT: Chat ── */}
        <div style={{ display: "flex", flexDirection: "column", borderRight: "1px solid var(--border)", height: "calc(100vh - 56px)" }}>

          {/* Chat header */}
          <div style={{ padding: "16px 24px", borderBottom: "1px solid var(--border)", background: "var(--bg2)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text3)", letterSpacing: "0.15em" }}>CHAT INTERFACE</span>
              <div style={{ flex: 1, height: 1, background: "var(--border)" }}/>
              <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text3)" }}>{sessionId.slice(-8)}</span>
            </div>
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "24px", display: "flex", flexDirection: "column", gap: 16 }}>
            {messages.length === 0 && (
              <div style={{ margin: "auto", textAlign: "center", maxWidth: 380 }}>
                <div style={{ fontSize: 32, marginBottom: 16, opacity: 0.3 }}>◎</div>
                <p style={{ fontFamily: "Syne", fontSize: 15, color: "var(--text2)", lineHeight: 1.7 }}>
                  MemoryOS remembers everything you tell it — across sessions, across time.
                </p>
                <p style={{ fontFamily: "JetBrains Mono", fontSize: 11, color: "var(--text3)", marginTop: 12 }}>
                  Try: "My name is Rahul and I'm building this system"
                </p>
              </div>
            )}

            <AnimatePresence initial={false}>
              {messages.map((m, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                  style={{ display: "flex", justifyContent: m.role === "user" ? "flex-end" : "flex-start" }}
                >
                  <div style={{
                    maxWidth: "78%",
                    padding: "12px 16px",
                    borderRadius: 2,
                    background: m.role === "user" ? "var(--gold-dim)" : "var(--bg3)",
                    border: `1px solid ${m.role === "user" ? "rgba(201,168,76,0.2)" : "var(--border)"}`,
                    position: "relative",
                  }}>
                    {m.role === "assistant" && m.memories_used !== undefined && (
                      <div style={{ marginBottom: 8, display: "flex", alignItems: "center", gap: 6 }}>
                        <div style={{ width: 4, height: 4, borderRadius: "50%", background: "var(--cyan)" }}/>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--cyan)", letterSpacing: "0.12em" }}>
                          {m.memories_used} MEMORIES RETRIEVED
                        </span>
                      </div>
                    )}
                    <p style={{ fontFamily: "Syne", fontSize: 14, lineHeight: 1.7, color: "var(--text)", whiteSpace: "pre-wrap" }}>
                      {m.content}
                    </p>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {loading && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ display: "flex", gap: 4, padding: "0 4px" }}>
                {[0,1,2].map(i => (
                  <div key={i} style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: "var(--gold)",
                    animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`
                  }}/>
                ))}
              </motion.div>
            )}
            <div ref={bottomRef}/>
          </div>

          {/* Input */}
          <div style={{ padding: "16px 24px", borderTop: "1px solid var(--border)", background: "var(--bg2)", display: "flex", gap: 12 }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !e.shiftKey && send()}
              placeholder="Send a message — MemoryOS will remember it..."
              style={{
                flex: 1, padding: "12px 16px", borderRadius: 2, fontSize: 13,
                background: "var(--bg3)", border: "1px solid var(--border)",
                color: "var(--text)", fontFamily: "JetBrains Mono",
              }}
            />
            <button
              onClick={send}
              disabled={loading || !input.trim()}
              style={{
                padding: "12px 20px", borderRadius: 2, cursor: "pointer",
                background: loading ? "transparent" : "var(--gold-dim)",
                border: "1px solid var(--gold)",
                color: "var(--gold)", fontFamily: "Syne", fontWeight: 600,
                fontSize: 12, letterSpacing: "0.1em",
                opacity: loading || !input.trim() ? 0.4 : 1,
                transition: "all 0.2s",
              }}
            >
              SEND
            </button>
          </div>
        </div>

        {/* ── RIGHT: Panels ── */}
        <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 56px)", overflow: "hidden" }}>

          {/* Tab bar */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--border)", background: "var(--bg2)" }}>
            {(["graph","memories","benchmark"] as const).map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                flex: 1, padding: "13px 0",
                fontFamily: "JetBrains Mono", fontSize: 10, letterSpacing: "0.12em",
                color: tab === t ? "var(--gold)" : "var(--text3)",
                background: "transparent", border: "none", cursor: "pointer",
                borderBottom: tab === t ? "1px solid var(--gold)" : "1px solid transparent",
                transition: "all 0.2s",
              }}>
                {t.toUpperCase()}
              </button>
            ))}
          </div>

          <div style={{ flex: 1, overflow: "hidden" }}>

            {/* ── GRAPH TAB ── */}
            {tab === "graph" && (
              <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
                <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between" }}>
                  <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text3)" }}>ASSOCIATIVE MEMORY GRAPH</span>
                  <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text2)" }}>
                    {graph.nodes.length}N · {graph.edges.length}E
                  </span>
                </div>

                {graph.nodes.length === 0 ? (
                  <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <p style={{ fontFamily: "JetBrains Mono", fontSize: 11, color: "var(--text3)", textAlign: "center" }}>
                      No memories yet.<br/>Start chatting to build the graph.
                    </p>
                  </div>
                ) : (
                  <div style={{ flex: 1 }}>
                    <MemoryGraph data={graph}/>
                  </div>
                )}

                {/* Legend */}
                <div style={{ padding: "10px 16px", borderTop: "1px solid var(--border)", display: "flex", gap: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--gold)", opacity: 0.6 }}/>
                    <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)" }}>User</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--cyan)", opacity: 0.6 }}/>
                    <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)" }}>Assistant</span>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <div style={{ width: 16, height: 1, background: "var(--gold)", opacity: 0.4 }}/>
                    <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)" }}>Association</span>
                  </div>
                </div>
              </div>
            )}

            {/* ── MEMORIES TAB ── */}
            {tab === "memories" && (
              <div style={{ height: "100%", overflowY: "auto", padding: 0 }}>
                <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--border)", position: "sticky", top: 0, background: "var(--bg2)", zIndex: 1 }}>
                  <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text3)" }}>
                    TOP MEMORIES BY IMPORTANCE
                  </span>
                </div>
                {memories.length === 0 ? (
                  <div style={{ padding: 32, textAlign: "center" }}>
                    <p style={{ fontFamily: "JetBrains Mono", fontSize: 11, color: "var(--text3)" }}>No memories stored yet.</p>
                  </div>
                ) : (
                  memories.map((m, i) => (
                    <motion.div
                      key={m.memory_id}
                      initial={{ opacity: 0, x: 8 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.03 }}
                      style={{
                        padding: "14px 16px",
                        borderBottom: "1px solid var(--border)",
                        display: "flex", flexDirection: "column", gap: 6,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{
                          fontFamily: "JetBrains Mono", fontSize: 9,
                          color: m.role === "user" ? "var(--gold)" : "var(--cyan)",
                          letterSpacing: "0.1em"
                        }}>
                          {m.role.toUpperCase()}
                        </span>
                        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <div style={{
                            width: 40, height: 3, borderRadius: 2,
                            background: "var(--bg3)",
                            overflow: "hidden",
                          }}>
                            <div style={{
                              width: `${m.importance * 100}%`,
                              height: "100%",
                              background: importanceColor(m.importance),
                              transition: "width 0.5s",
                            }}/>
                          </div>
                          <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: importanceColor(m.importance) }}>
                            {m.importance.toFixed(3)}
                          </span>
                        </div>
                      </div>
                      <p style={{ fontFamily: "Syne", fontSize: 12, color: "var(--text2)", lineHeight: 1.5 }}>
                        {m.text.slice(0, 120)}{m.text.length > 120 ? "…" : ""}
                      </p>
                    </motion.div>
                  ))
                )}
              </div>
            )}

            {/* ── BENCHMARK TAB ── */}
            {tab === "benchmark" && (
              <div style={{ height: "100%", overflowY: "auto", padding: "16px" }}>
                <div style={{ marginBottom: 16 }}>
                  <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text3)", letterSpacing: "0.12em" }}>
                    MEMORYOS vs BASELINE RAG
                  </span>
                </div>

                {/* Live stats */}
                {health && (
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 16 }}>
                    {[
                      { label: "TOTAL MEMORIES", value: health.memory_store.total_memories, color: "var(--gold)" },
                      { label: "GRAPH NODES",    value: health.graph.total_nodes, color: "var(--cyan)" },
                      { label: "GRAPH EDGES",    value: health.graph.total_edges, color: "var(--cyan)" },
                      { label: "CONSOLIDATIONS", value: health.consolidator.total_runs, color: "var(--green)" },
                    ].map(s => (
                      <div key={s.label} style={{
                        padding: "12px", background: "var(--bg3)",
                        border: "1px solid var(--border)", borderRadius: 2,
                      }}>
                        <div style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)", marginBottom: 4 }}>{s.label}</div>
                        <div style={{ fontFamily: "Syne", fontSize: 22, fontWeight: 700, color: s.color }}>{s.value}</div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Benchmark results */}
                <div style={{ background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: 2, padding: 16 }}>
                  <div style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text3)", marginBottom: 16, letterSpacing: "0.1em" }}>
                    BENCHMARK RESULTS — 3 SESSIONS · 15 QUERIES
                  </div>

                  {[
                    { metric: "MRR",        memoryos: 0.7111, rag: 0.6667, gain: "+6.7%", better: true },
                    { metric: "Precision@5", memoryos: 0.2800, rag: 0.2800, gain: "0.0%",  better: null },
                    { metric: "Recall@5",   memoryos: 0.8167, rag: 0.8444, gain: "-3.3%", better: false },
                  ].map(row => (
                    <div key={row.metric} style={{ marginBottom: 14 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 10, color: "var(--text2)" }}>{row.metric}</span>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 10,
                          color: row.better === true ? "var(--green)" : row.better === false ? "var(--red)" : "var(--text3)" }}>
                          {row.gain}
                        </span>
                      </div>
                      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
                        {/* MemoryOS bar */}
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--gold)", width: 60 }}>MemOS</span>
                        <div style={{ flex: 1, height: 6, background: "var(--bg2)", borderRadius: 1, overflow: "hidden" }}>
                          <div style={{ width: `${row.memoryos * 100}%`, height: "100%", background: "var(--gold)", borderRadius: 1 }}/>
                        </div>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--gold)", width: 36, textAlign: "right" }}>
                          {row.memoryos.toFixed(3)}
                        </span>
                      </div>
                      <div style={{ display: "flex", gap: 6, alignItems: "center", marginTop: 4 }}>
                        {/* RAG bar */}
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)", width: 60 }}>RAG</span>
                        <div style={{ flex: 1, height: 6, background: "var(--bg2)", borderRadius: 1, overflow: "hidden" }}>
                          <div style={{ width: `${row.rag * 100}%`, height: "100%", background: "var(--text3)", borderRadius: 1 }}/>
                        </div>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)", width: 36, textAlign: "right" }}>
                          {row.rag.toFixed(3)}
                        </span>
                      </div>
                    </div>
                  ))}

                  {/* Win rate */}
                  <div style={{ marginTop: 20, padding: "12px", background: "var(--bg2)", borderRadius: 2, border: "1px solid var(--border)" }}>
                    <div style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)", marginBottom: 8 }}>QUERY WIN RATE</div>
                    <div style={{ display: "flex", gap: 0, height: 16, borderRadius: 1, overflow: "hidden" }}>
                      <div style={{ width: `${(5/15)*100}%`, background: "var(--gold)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 8, color: "#000" }}>5</span>
                      </div>
                      <div style={{ width: `${(8/15)*100}%`, background: "var(--bg3)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 8, color: "var(--text3)" }}>8</span>
                      </div>
                      <div style={{ width: `${(2/15)*100}%`, background: "var(--text3)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <span style={{ fontFamily: "JetBrains Mono", fontSize: 8, color: "#000" }}>2</span>
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
                      <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--gold)" }}>■ MemoryOS 5/15</span>
                      <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text3)" }}>■ Ties 8/15</span>
                      <span style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--text2)" }}>■ RAG 2/15</span>
                    </div>
                  </div>

                  {/* Architecture note */}
                  <div style={{ marginTop: 16, padding: "10px 12px", background: "var(--gold-glow)", border: "1px solid rgba(201,168,76,0.15)", borderRadius: 2 }}>
                    <p style={{ fontFamily: "JetBrains Mono", fontSize: 9, color: "var(--gold)", lineHeight: 1.7 }}>
                      MemoryOS uses associative graph traversal + importance-weighted episodic encoding + Ebbinghaus decay — not flat cosine similarity. MRR +6.7% means the most relevant memory surfaces first more often.
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}